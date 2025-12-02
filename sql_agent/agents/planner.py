"""
Planner Agent - The Strategist

Uses RAG to understand business context and maps out the data retrieval strategy.
Outputs a structured QueryPlan or ClarificationRequest.

Uses Qwen2.5-Coder base model for planning.
"""

from typing import Literal
from datetime import datetime
from langgraph.types import Command, interrupt

from sql_agent.utils.state import AgentState, QueryPlan, ClarificationRequest
from sql_agent.utils.tools import RAGTools, DatabaseTools
from sql_agent.utils.models import get_qwen_model


# ============================================================================
# Planner Node Functions
# ============================================================================

def create_planner_nodes(
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    rag_tools: RAGTools = None,
    db_tools: DatabaseTools = None
):
    """Factory function to create planner nodes with injected dependencies."""
    
    # Use base Qwen model for planning (no LoRA)
    llm = get_qwen_model(model_name=base_model, lora_path=None)
    
    # -------------------------------------------------------------------------
    # Node 1: Retrieve Context (RAG)
    # -------------------------------------------------------------------------
    def retrieve_context(state: AgentState) -> dict:
        """Retrieve schema, glossary, and cookbook context using RAG."""
        question = state["user_question"]
        
        schema_context = rag_tools.retrieve_schema_context(question, k=8)
        glossary_context = rag_tools.retrieve_glossary_context(question, k=3)
        cookbook_context = rag_tools.retrieve_cookbook_examples(question, k=2)
        
        return {
            "schema_context": schema_context,
            "glossary_context": glossary_context,
            "cookbook_context": cookbook_context,
            "current_date": datetime.now().strftime("%Y-%m-%d"),
            "status": "planning"
        }
    
    # -------------------------------------------------------------------------
    # Node 2: Analyze and Plan
    # -------------------------------------------------------------------------
    def analyze_and_plan(state: AgentState) -> Command[Literal["verify_values", "request_clarification", "synthesizer"]]:
        """
        Analyze the question and create a query plan.
        Routes to value verification, clarification, or directly to synthesizer.
        """
        
        # Build the planning prompt
        planning_prompt = f"""You are a SQL query planner for a music database (Chinook).
Your job is to analyze the user's question and create a structured query plan.

## Current Date (for temporal queries)
{state["current_date"]}

## Available Schema Context
{state["schema_context"]}

## Business Glossary
{state["glossary_context"]}

## Similar Query Examples
{state["cookbook_context"]}

## User Question
{state["user_question"]}

## Previous Error Feedback (if any)
{state.get("error_feedback", "None")}

## Instructions
1. Identify the primary entity (main table) the user is asking about
2. List ALL tables needed to answer the question
3. Determine the join path between tables
4. Extract filter conditions - convert relative dates to absolute (e.g., "last month" -> specific dates)
5. Identify any aggregations needed (COUNT, SUM, AVG, etc.)
6. Note any ordering requirements
7. Set an appropriate LIMIT (default 50 unless user specifies)

## Important Rules
- If you see categorical filters (Genre, Artist, Country), note them for value verification
- If the question is ambiguous (could mean multiple things), set needs_clarification=true
- Convert ALL relative dates to absolute ISO dates using the current date
- Explain your reasoning clearly

Respond with a QueryPlan JSON object.
"""
        
        # Get structured output
        structured_llm = llm.with_structured_output(QueryPlan)
        
        try:
            plan = structured_llm.invoke(planning_prompt)
            plan_dict = plan.model_dump()
            
            # Check if we need to verify categorical values
            categorical_filters = [f for f in plan.filters if any(
                term in f.lower() for term in ["genre", "artist", "country", "city", "name"]
            )]
            
            if categorical_filters:
                # Route to value verification
                return Command(
                    update={
                        "query_plan": plan_dict,
                        "needs_clarification": False,
                    },
                    goto="verify_values"
                )
            else:
                # No categorical filters, go directly to synthesizer
                return Command(
                    update={
                        "query_plan": plan_dict,
                        "needs_clarification": False,
                        "status": "synthesizing"
                    },
                    goto="synthesizer"
                )
                
        except Exception as e:
            # If planning fails, request clarification
            return Command(
                update={
                    "needs_clarification": True,
                    "clarification_request": {
                        "ambiguous_term": "query",
                        "possible_interpretations": ["Unable to parse question"],
                        "question": f"I had trouble understanding your question. Could you rephrase it? Error: {str(e)}"
                    },
                    "status": "clarifying"
                },
                goto="request_clarification"
            )
    
    # -------------------------------------------------------------------------
    # Node 3: Verify Values (High-Cardinality Lookup)
    # -------------------------------------------------------------------------
    def verify_values(state: AgentState) -> Command[Literal["synthesizer", "request_clarification"]]:
        """
        Verify that categorical filter values exist in the database.
        Updates the plan with exact DB values.
        """
        plan = state["query_plan"]
        resolved_values = {}
        unresolved = []
        
        for filter_str in plan.get("filters", []):
            # Parse simple filters like "Genre.Name = 'Rock'"
            if "=" in filter_str or "LIKE" in filter_str.upper():
                parts = filter_str.replace("'", "").replace('"', "").split("=")
                if len(parts) == 2:
                    column_part = parts[0].strip()
                    value_part = parts[1].strip().replace("LIKE", "").replace("%", "").strip()
                    
                    # Extract table and column
                    if "." in column_part:
                        table, column = column_part.split(".")
                    else:
                        # Guess the table from context
                        table = plan.get("primary_entity", "")
                        column = column_part
                    
                    if table and column and value_part:
                        # Search for matching values
                        matches = db_tools.search_column_values(table, column, value_part)
                        
                        if matches and not matches[0].startswith("Error"):
                            # Found matches - use the closest one
                            resolved_values[f"{table}.{column}"] = matches[0]
                        else:
                            unresolved.append({
                                "filter": filter_str,
                                "searched": value_part,
                                "table": table,
                                "column": column
                            })
        
        if unresolved:
            # Get available values for clarification
            available_values = []
            for item in unresolved:
                vals = db_tools.get_distinct_values(item["table"], item["column"], limit=10)
                available_values.extend(vals[:5])
            
            return Command(
                update={
                    "needs_clarification": True,
                    "clarification_request": {
                        "ambiguous_term": unresolved[0]["searched"],
                        "possible_interpretations": available_values,
                        "question": f"I couldn't find '{unresolved[0]['searched']}' in the database. Did you mean one of these: {', '.join(available_values[:5])}?"
                    },
                    "status": "clarifying"
                },
                goto="request_clarification"
            )
        
        # Update plan with resolved values
        updated_filters = []
        for filter_str in plan.get("filters", []):
            for key, value in resolved_values.items():
                if key.split(".")[1].lower() in filter_str.lower():
                    # Update the filter with the exact value
                    table, col = key.split(".")
                    updated_filters.append(f"{table}.{col} = '{value}'")
                    break
            else:
                updated_filters.append(filter_str)
        
        plan["filters"] = updated_filters
        
        return Command(
            update={
                "query_plan": plan,
                "resolved_values": resolved_values,
                "status": "synthesizing"
            },
            goto="synthesizer"
        )
    
    # -------------------------------------------------------------------------
    # Node 4: Request Clarification (Human-in-the-Loop)
    # -------------------------------------------------------------------------
    def request_clarification(state: AgentState) -> Command[Literal["analyze_and_plan"]]:
        """
        Pause execution and ask the user for clarification.
        Uses LangGraph's interrupt() for human-in-the-loop.
        """
        clarification = state["clarification_request"]
        
        # This will pause the graph and wait for user input
        user_response = interrupt({
            "type": "clarification_needed",
            "question": clarification["question"],
            "options": clarification["possible_interpretations"],
            "original_term": clarification["ambiguous_term"]
        })
        
        # User has responded - update the question with clarification
        updated_question = f"{state['user_question']} (Clarification: {user_response})"
        
        return Command(
            update={
                "user_question": updated_question,
                "needs_clarification": False,
                "clarification_request": None,
                "status": "planning"
            },
            goto="analyze_and_plan"
        )
    
    return {
        "retrieve_context": retrieve_context,
        "analyze_and_plan": analyze_and_plan,
        "verify_values": verify_values,
        "request_clarification": request_clarification,
    }
