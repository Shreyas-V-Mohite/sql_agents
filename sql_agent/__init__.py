# SQL Agent - Multi-Agent Text-to-SQL System
"""
A LangGraph-based multi-agent system for converting natural language to SQL.

Components:
- Planner: RAG-based query planning with schema understanding
- Synthesizer: SQL code generation
- Executor: Safe SQL execution
- Controller: Workflow orchestration and validation
"""

from sql_agent.graph import create_sql_agent

__all__ = ["create_sql_agent"]
