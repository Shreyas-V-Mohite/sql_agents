"""
State definitions for the SQL Agent pipeline.
All agents share this common state schema.
"""

from typing import TypedDict, Literal, Optional, List, Any
from pydantic import BaseModel, Field


# ============================================================================
# Pydantic Models for Structured LLM Output
# ============================================================================

class QueryPlan(BaseModel):
    """Structured output from the Planner agent."""
    primary_entity: str = Field(description="Main table/entity the user is asking about")
    tables_involved: List[str] = Field(description="All tables needed for the query")
    join_path: str = Field(description="How tables connect, e.g., 'Track -> Album -> Artist'")
    filters: List[str] = Field(description="Filter conditions, e.g., ['Artist.Name = AC/DC']")
    aggregations: Optional[str] = Field(default=None, description="Aggregation needed, e.g., 'COUNT', 'SUM'")
    ordering: Optional[str] = Field(default=None, description="ORDER BY clause if needed")
    limit: Optional[int] = Field(default=50, description="Result limit")
    reasoning: str = Field(description="Explanation of why this plan was chosen")


class ClarificationRequest(BaseModel):
    """Returned when the Planner needs user clarification."""
    ambiguous_term: str = Field(description="The term that is unclear")
    possible_interpretations: List[str] = Field(description="Possible meanings")
    question: str = Field(description="Question to ask the user")


class ExecutionResult(BaseModel):
    """Result from the Executor."""
    success: bool
    columns: Optional[List[str]] = None
    data: Optional[List[Any]] = None
    row_count: int = 0
    total_row_count: int = 0  # Actual count before truncation
    truncated: bool = False   # True if results were limited
    error_message: Optional[str] = None
    error_type: Optional[Literal["syntax", "logic", "security", "timeout"]] = None


# ============================================================================
# Shared Agent State
# ============================================================================

class AgentState(TypedDict):
    """Shared state across all agents in the pipeline."""
    
    # User input
    user_question: str
    
    # Planner outputs
    schema_context: Optional[str]          # Retrieved schema info from RAG
    glossary_context: Optional[str]        # Retrieved business terms
    cookbook_context: Optional[str]        # Similar query examples
    query_plan: Optional[dict]             # QueryPlan as dict
    needs_clarification: bool
    clarification_request: Optional[dict]  # ClarificationRequest as dict
    resolved_values: Optional[dict]        # Verified DB values (e.g., {"Genre": "Rock And Roll"})
    current_date: str                      # For temporal grounding
    
    # Synthesizer outputs
    generated_sql: Optional[str]
    
    # Executor outputs
    execution_result: Optional[dict]       # ExecutionResult as dict
    
    # Controller state
    retry_count: int
    max_retries: int
    error_feedback: Optional[str]          # Feedback for retry loops
    final_answer: Optional[str]
    status: Literal["planning", "synthesizing", "executing", "validating", "complete", "failed", "clarifying"]
