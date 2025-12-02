"""
Controller Agent - The Supervisor

Routes workflow based on execution results.
Handles retries, error classification, and final answer generation.

Uses Qwen2.5-Coder base model for answer generation.
"""

from typing import Literal
from langgraph.types import Command
from langgraph.graph import END

from sql_agent.utils.state import AgentState
from sql_agent.utils.models import get_qwen_model


def create_controller_node(base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"):
    """Factory function to create the controller node."""
    
    # Use base Qwen model for answer generation (no LoRA)
    llm = get_qwen_model(model_name=base_model, lora_path=None)
    
    def control_flow(state: AgentState) -> Command[Literal["synthesizer", "analyze_and_plan", "__end__"]]:
        """
        Analyze execution results and decide next action.
        Routes to retry, re-plan, or finalize.
        """
        result = state["execution_result"]
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 3)
        
        # Case 1: Successful execution
        if result and result.get("success"):
            data = result.get("data", [])
            columns = result.get("columns", [])
            row_count = result.get("row_count", 0)
            
            # Check for empty results (potential logic error)
            if row_count == 0:
                if retry_count < max_retries:
                    return Command(
                        update={
                            "retry_count": retry_count + 1,
                            "error_feedback": "Query returned 0 results. Check if filter values exist in the database or if join conditions are correct.",
                            "status": "planning"
                        },
                        goto="analyze_and_plan"
                    )
                else:
                    # Max retries reached, return empty result
                    final_answer = _generate_final_answer(
                        llm, state, "The query executed successfully but returned no results. "
                        "This might mean the data you're looking for doesn't exist in the database."
                    )
                    return Command(
                        update={
                            "final_answer": final_answer,
                            "status": "complete"
                        },
                        goto=END
                    )
            
            # Semantic validation: check if columns match expected entity
            plan = state.get("query_plan", {})
            primary_entity = plan.get("primary_entity", "").lower()
            columns_lower = [c.lower() for c in columns]
            
            # Basic semantic check
            if primary_entity and not any(primary_entity in c for c in columns_lower):
                # Columns don't seem to match the expected entity
                if retry_count < max_retries:
                    return Command(
                        update={
                            "retry_count": retry_count + 1,
                            "error_feedback": f"Semantic mismatch: User asked about '{primary_entity}' but query returned columns: {columns}. Please adjust the SELECT clause.",
                            "status": "synthesizing"
                        },
                        goto="synthesizer"
                    )
            
            # Success! Generate final answer
            final_answer = _generate_final_answer(llm, state, None)
            
            # Add truncation note if results were limited
            if result.get("truncated"):
                final_answer += f"\n\n_Note: Results limited to top {result.get('row_count', 50)} rows for performance._"
            
            return Command(
                update={
                    "final_answer": final_answer,
                    "status": "complete"
                },
                goto=END
            )
        
        # Case 2: Execution failed
        error_type = result.get("error_type") if result else "unknown"
        error_message = result.get("error_message", "Unknown error") if result else "No result"
        
        if retry_count >= max_retries:
            # Max retries reached
            final_answer = (
                f"I was unable to answer your question after {max_retries} attempts.\n\n"
                f"Last error: {error_message}\n\n"
                f"Last SQL attempted:\n```sql\n{state.get('generated_sql', 'N/A')}\n```"
            )
            return Command(
                update={
                    "final_answer": final_answer,
                    "status": "failed"
                },
                goto=END
            )
        
        # Route based on error type
        if error_type == "security":
            # Security violation - go back to synthesizer with strict warning
            return Command(
                update={
                    "retry_count": retry_count + 1,
                    "error_feedback": f"SECURITY ALERT: {error_message}. You MUST generate only SELECT statements. No INSERT, UPDATE, DELETE, or DROP.",
                    "status": "synthesizing"
                },
                goto="synthesizer"
            )
        
        elif error_type == "syntax":
            # Syntax error - go back to synthesizer
            return Command(
                update={
                    "retry_count": retry_count + 1,
                    "error_feedback": f"SQL Syntax Error: {error_message}. Please fix the query.",
                    "status": "synthesizing"
                },
                goto="synthesizer"
            )
        
        elif error_type == "logic":
            # Logic error - go back to planner
            return Command(
                update={
                    "retry_count": retry_count + 1,
                    "error_feedback": f"Logic Error: {error_message}. Please reconsider the query plan.",
                    "status": "planning"
                },
                goto="analyze_and_plan"
            )
        
        else:
            # Unknown error - try synthesizer first
            return Command(
                update={
                    "retry_count": retry_count + 1,
                    "error_feedback": f"Error: {error_message}",
                    "status": "synthesizing"
                },
                goto="synthesizer"
            )
    
    return control_flow


def _generate_final_answer(llm, state: AgentState, override_message: str = None) -> str:
    """Generate a natural language answer from the results."""
    
    if override_message:
        return override_message
    
    result = state["execution_result"]
    columns = result.get("columns", [])
    data = result.get("data", [])
    row_count = result.get("row_count", 0)
    sql = state.get("generated_sql", "")
    question = state["user_question"]
    
    # For small results, include the data in the prompt
    if row_count <= 20:
        data_preview = "\n".join([str(row) for row in data])
    else:
        data_preview = "\n".join([str(row) for row in data[:10]])
        data_preview += f"\n... and {row_count - 10} more rows"
    
    answer_prompt = f"""Based on the SQL query results, provide a clear, natural language answer to the user's question.

## User Question
{question}

## SQL Query Executed
```sql
{sql}
```

## Results
Columns: {columns}
Rows Returned: {row_count}
{f"(Truncated from {result.get('total_row_count', row_count)} total rows)" if result.get('truncated') else ""}

Data:
{data_preview}

## Instructions
1. Answer the question directly and concisely
2. Include relevant numbers/data from the results
3. If the data is a list, format it nicely
4. Mention the total count if relevant
5. Keep the answer focused on what the user asked
"""
    
    response = llm.invoke(answer_prompt)
    return response.content
