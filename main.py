"""
Main Entry Point - SQL Agent

Run the multi-agent SQL pipeline interactively or programmatically.
Uses Qwen2.5-Coder models (local inference).
"""

import os
from uuid import uuid4

from sql_agent.graph import create_sql_agent, get_initial_state
from langgraph.types import Command

# Default paths
DEFAULT_SYNTHESIZER_LORA = "SQL_Synthesizer_Weights_v2"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"


def run_query(agent, question: str, thread_id: str = None):
    """
    Run a single query through the agent pipeline.
    
    Args:
        agent: Compiled SQL agent
        question: Natural language question
        thread_id: Thread ID for persistence (auto-generated if None)
    
    Returns:
        Final answer string
    """
    if thread_id is None:
        thread_id = str(uuid4())
    
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = get_initial_state(question)
    
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}\n")
    
    # Stream through the agent
    for event in agent.stream(initial_state, config, stream_mode="updates"):
        # Print progress
        for node_name, node_output in event.items():
            status = node_output.get("status", "")
            if status:
                print(f"[{node_name}] Status: {status}")
            
            # Show generated SQL
            if "generated_sql" in node_output and node_output["generated_sql"]:
                print(f"\n📝 Generated SQL:\n{node_output['generated_sql']}\n")
            
            # Show execution result summary
            if "execution_result" in node_output and node_output["execution_result"]:
                result = node_output["execution_result"]
                if result.get("success"):
                    print(f"✓ Query executed: {result.get('row_count', 0)} rows returned")
                else:
                    print(f"✗ Error: {result.get('error_message', 'Unknown')}")
    
    # Get final state
    final_state = agent.get_state(config)
    final_answer = final_state.values.get("final_answer", "No answer generated")
    
    print(f"\n{'='*60}")
    print("FINAL ANSWER:")
    print(f"{'='*60}")
    print(final_answer)
    
    return final_answer


def run_with_clarification(agent, question: str, thread_id: str = None):
    """
    Run a query that might need clarification (human-in-the-loop).
    
    Args:
        agent: Compiled SQL agent
        question: Natural language question
        thread_id: Thread ID for persistence
    
    Returns:
        Final answer string
    """
    if thread_id is None:
        thread_id = str(uuid4())
    
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = get_initial_state(question)
    
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}\n")
    
    # First run - might pause for clarification
    result = agent.invoke(initial_state, config)
    
    # Check if we need clarification
    state = agent.get_state(config)
    
    while state.values.get("status") == "clarifying":
        # Get the clarification request from the interrupt
        tasks = state.tasks
        if tasks and hasattr(tasks[0], 'interrupts') and tasks[0].interrupts:
            interrupt_data = tasks[0].interrupts[0].value
            
            print("\n🤔 Clarification needed:")
            print(f"   {interrupt_data.get('question', 'Please clarify')}")
            if interrupt_data.get('options'):
                print(f"   Options: {interrupt_data['options']}")
            
            # Get user input
            user_response = input("\nYour response: ").strip()
            
            # Resume with the clarification
            result = agent.invoke(
                Command(resume=user_response),
                config
            )
            
            # Check state again
            state = agent.get_state(config)
        else:
            break
    
    final_answer = state.values.get("final_answer", "No answer generated")
    
    print(f"\n{'='*60}")
    print("FINAL ANSWER:")
    print(f"{'='*60}")
    print(final_answer)
    
    return final_answer


def interactive_mode(agent):
    """Run the agent in interactive mode."""
    print("\n" + "="*60)
    print("SQL Agent - Interactive Mode")
    print("="*60)
    print("Ask questions about the Chinook music database.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    thread_id = str(uuid4())
    
    while True:
        try:
            question = input("\n🎵 Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            run_with_clarification(agent, question, thread_id)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SQL Agent - Natural Language to SQL (Qwen)")
    parser.add_argument("--setup", action="store_true", help="Run initial setup (download DB, create vector store)")
    parser.add_argument("--query", "-q", type=str, help="Run a single query")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--db", type=str, default="Chinook.db", help="Path to SQLite database")
    parser.add_argument("--chroma", type=str, default="./chroma_db", help="ChromaDB persist directory")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL, help="Base Qwen model")
    parser.add_argument("--lora", type=str, default=DEFAULT_SYNTHESIZER_LORA, help="Path to Synthesizer LoRA weights")
    parser.add_argument("--no-lora", action="store_true", help="Don't use fine-tuned model for Synthesizer")
    
    args = parser.parse_args()
    
    if args.setup:
        from sql_agent.setup_data import setup_all
        setup_all(args.db, args.chroma)
        return
    
    # Determine LoRA path
    lora_path = None if args.no_lora else args.lora
    
    # Create the agent
    print("=" * 60)
    print("Initializing SQL Agent (Qwen2.5-Coder)")
    print("=" * 60)
    print(f"Base Model: {args.base_model}")
    print(f"Synthesizer LoRA: {lora_path or 'None (using base model)'}")
    print("Loading models... (this may take a moment)")
    
    agent = create_sql_agent(
        db_path=args.db,
        chroma_persist_dir=args.chroma,
        base_model=args.base_model,
        synthesizer_lora_path=lora_path
    )
    print("✓ Agent ready!\n")
    
    if args.query:
        run_with_clarification(agent, args.query)
    elif args.interactive:
        interactive_mode(agent)
    else:
        # Default: interactive mode
        interactive_mode(agent)


if __name__ == "__main__":
    main()
