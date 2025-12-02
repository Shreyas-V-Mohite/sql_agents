"""
SQL Synthesizer Agent - The Specialist

Translates the Planner's strategy into executable SQLite SQL.
Enforces safety and SQLite dialect rules.

Uses fine-tuned Qwen2.5-Coder model for SQL generation.
"""

from typing import Literal, Optional
from langgraph.types import Command

from sql_agent.utils.state import AgentState
from sql_agent.utils.models import get_qwen_model, QwenModel


def create_synthesizer_node(
    lora_path: Optional[str] = None,
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
):
    """
    Factory function to create the synthesizer node.
    
    Args:
        lora_path: Path to fine-tuned LoRA weights (e.g., "SQL_Synthesizer_Weights_v2")
        base_model: Base Qwen model name
    """
    # Load fine-tuned model for SQL synthesis
    llm = get_qwen_model(model_name=base_model, lora_path=lora_path)
    
    def synthesize_sql(state: AgentState) -> Command[Literal["executor"]]:
        """
        Generate SQL from the query plan.
        Enforces SQLite dialect and safety rules.
        """
        plan = state["query_plan"]
        error_feedback = state.get("error_feedback")
        
        synthesis_prompt = f"""You are a SQL code generator for SQLite databases.
Generate a SELECT query based on the provided plan.

## Query Plan
- Primary Entity: {plan.get('primary_entity')}
- Tables Involved: {', '.join(plan.get('tables_involved', []))}
- Join Path: {plan.get('join_path')}
- Filters: {plan.get('filters')}
- Aggregations: {plan.get('aggregations')}
- Ordering: {plan.get('ordering')}
- Limit: {plan.get('limit', 50)}
- Reasoning: {plan.get('reasoning')}

## Schema Context
{state.get('schema_context', 'No schema context available')}

## Previous Error (if retrying)
{error_feedback if error_feedback else 'None - first attempt'}

## SQLite-Specific Rules (CRITICAL)
1. Use || for string concatenation (NOT CONCAT)
2. Use strftime('%Y', date_column) for year extraction (NOT YEAR())
3. Use date('now') for current date (NOT NOW())
4. Use julianday() for date differences (NOT DATEDIFF)
5. SQLite is case-insensitive for keywords but case-sensitive for identifiers
6. Use double quotes for identifiers with spaces, single quotes for strings

## Safety Rules (CRITICAL)
1. Generate ONLY SELECT statements
2. NEVER generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE
3. Always include a LIMIT clause (use {plan.get('limit', 50)} if not specified)
4. Use table aliases for clarity in JOINs

## Output Format
Return ONLY the SQL query, no explanations or markdown.
The query must be executable as-is.

## Defensive SQL Patterns
- Wrap nullable columns: COALESCE(column, default_value)
- Use explicit JOINs (not implicit comma joins)
- Prefer CTEs (WITH clause) for complex queries
"""
        
        response = llm.invoke(synthesis_prompt)
        sql = response.content.strip()
        
        # Clean up any markdown formatting
        if sql.startswith("```"):
            sql = sql.split("```")[1]
            if sql.startswith("sql"):
                sql = sql[3:]
            sql = sql.strip()
        
        # Basic safety check
        sql_upper = sql.upper()
        dangerous_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE"]
        for keyword in dangerous_keywords:
            if keyword in sql_upper and "SELECT" not in sql_upper[:20]:
                return Command(
                    update={
                        "generated_sql": None,
                        "error_feedback": f"SECURITY: Attempted to generate {keyword} statement. Only SELECT is allowed.",
                        "status": "failed"
                    },
                    goto="controller"
                )
        
        return Command(
            update={
                "generated_sql": sql,
                "status": "executing"
            },
            goto="executor"
        )
    
    return synthesize_sql
