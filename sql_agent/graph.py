"""
Main Graph Definition - SQL Agent Pipeline

Assembles all agents into a single LangGraph workflow.
Uses Qwen2.5-Coder models (base + fine-tuned for Synthesizer).
"""

from typing import Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from sql_agent.utils.state import AgentState
from sql_agent.utils.tools import RAGTools, DatabaseTools, create_vector_store
from sql_agent.agents.planner import create_planner_nodes
from sql_agent.agents.synthesizer import create_synthesizer_node
from sql_agent.agents.executor import create_executor_node
from sql_agent.agents.controller import create_controller_node


def create_sql_agent(
    db_path: str,
    chroma_persist_dir: str = "./chroma_db",
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    synthesizer_lora_path: Optional[str] = None,
    use_checkpointer: bool = True
):
    """
    Create the complete SQL Agent pipeline.
    
    Args:
        db_path: Path to the SQLite database
        chroma_persist_dir: Directory for ChromaDB persistence
        base_model: Base Qwen model for all agents
        synthesizer_lora_path: Path to fine-tuned LoRA weights for Synthesizer
        use_checkpointer: Whether to enable persistence (required for interrupt)
    
    Returns:
        Compiled LangGraph agent
    """
    
    # Initialize tools
    vector_store = create_vector_store(chroma_persist_dir)
    rag_tools = RAGTools(vector_store)
    db_tools = DatabaseTools(db_path)
    
    # Create agent nodes
    # Planner and Controller use base model
    planner_nodes = create_planner_nodes(base_model, rag_tools, db_tools)
    controller_node = create_controller_node(base_model)
    
    # Synthesizer uses fine-tuned model (if provided)
    synthesizer_node = create_synthesizer_node(
        lora_path=synthesizer_lora_path,
        base_model=base_model
    )
    
    # Executor is deterministic (no LLM)
    executor_node = create_executor_node(db_path)
    
    # Build the graph
    workflow = StateGraph(AgentState)
    
    # Add Planner nodes
    workflow.add_node("retrieve_context", planner_nodes["retrieve_context"])
    workflow.add_node("analyze_and_plan", planner_nodes["analyze_and_plan"])
    workflow.add_node("verify_values", planner_nodes["verify_values"])
    workflow.add_node("request_clarification", planner_nodes["request_clarification"])
    
    # Add other agent nodes
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("controller", controller_node)
    
    # Define edges
    # Entry point
    workflow.add_edge(START, "retrieve_context")
    workflow.add_edge("retrieve_context", "analyze_and_plan")
    
    # Note: analyze_and_plan, verify_values, request_clarification, synthesizer, 
    # executor, and controller all use Command for routing, so we don't need 
    # explicit edges for them - they handle their own routing.
    
    # Compile with checkpointer for interrupt support
    if use_checkpointer:
        checkpointer = MemorySaver()
        agent = workflow.compile(checkpointer=checkpointer)
    else:
        agent = workflow.compile()
    
    return agent


def get_initial_state(question: str) -> AgentState:
    """Create initial state for a new query."""
    return {
        "user_question": question,
        "schema_context": None,
        "glossary_context": None,
        "cookbook_context": None,
        "query_plan": None,
        "needs_clarification": False,
        "clarification_request": None,
        "resolved_values": None,
        "current_date": "",
        "generated_sql": None,
        "execution_result": None,
        "retry_count": 0,
        "max_retries": 3,
        "error_feedback": None,
        "final_answer": None,
        "status": "planning"
    }
