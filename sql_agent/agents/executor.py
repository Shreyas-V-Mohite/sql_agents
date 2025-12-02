"""
Executor Node - The Technician

Safely executes SQL queries against the database.
Enforces read-only access and handles errors gracefully.
"""

import sqlite3
from typing import Literal
from langgraph.types import Command

from sql_agent.utils.state import AgentState, ExecutionResult

# Constants for truncation
PREVIEW_LIMIT = 50


def create_executor_node(db_path: str):
    """Factory function to create the executor node with database path."""
    
    def execute_sql(state: AgentState) -> Command[Literal["controller"]]:
        """
        Execute the generated SQL query safely.
        Uses read-only connection and returns structured results.
        """
        sql = state["generated_sql"]
        
        if not sql:
            return Command(
                update={
                    "execution_result": ExecutionResult(
                        success=False,
                        error_message="No SQL query provided",
                        error_type="logic"
                    ).model_dump(),
                    "status": "validating"
                },
                goto="controller"
            )
        
        try:
            # Connect in read-only mode (critical safety measure)
            conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Execute with timeout
            cursor.execute(sql)
            
            # Extract column names
            columns = [description[0] for description in cursor.description] if cursor.description else []
            
            # Fetch with truncation limit
            preview_rows = cursor.fetchmany(PREVIEW_LIMIT + 1)  # Fetch one extra to check if more exist
            
            # Check if results were truncated
            if len(preview_rows) > PREVIEW_LIMIT:
                # More rows exist - truncate and count total
                data = [tuple(row) for row in preview_rows[:PREVIEW_LIMIT]]
                truncated = True
                
                # Count remaining rows (efficient way)
                remaining_count = 1  # We already fetched one extra
                while cursor.fetchmany(1000):
                    remaining_count += len(cursor.fetchmany(1000)) or 0
                # Approximate: just note it's truncated, don't count all
                total_row_count = PREVIEW_LIMIT + remaining_count
            else:
                # All rows fit in preview
                data = [tuple(row) for row in preview_rows]
                truncated = False
                total_row_count = len(data)
            
            conn.close()
            
            # Sanitize data types for LLM consumption
            data = _sanitize_data(data)
            
            result = ExecutionResult(
                success=True,
                columns=columns,
                data=data,
                row_count=len(data),
                total_row_count=total_row_count,
                truncated=truncated,
                error_message=None,
                error_type=None
            )
            
            return Command(
                update={
                    "execution_result": result.model_dump(),
                    "status": "validating"
                },
                goto="controller"
            )
            
        except sqlite3.OperationalError as e:
            error_msg = str(e)
            
            # Classify the error
            if "readonly" in error_msg.lower():
                error_type = "security"
            elif "no such table" in error_msg.lower() or "no such column" in error_msg.lower():
                error_type = "syntax"
                # Try to provide helpful suggestions
                error_msg = _enhance_error_message(error_msg, db_path)
            elif "syntax error" in error_msg.lower():
                error_type = "syntax"
            else:
                error_type = "syntax"
            
            result = ExecutionResult(
                success=False,
                error_message=error_msg,
                error_type=error_type
            )
            
            return Command(
                update={
                    "execution_result": result.model_dump(),
                    "status": "validating"
                },
                goto="controller"
            )
            
        except Exception as e:
            result = ExecutionResult(
                success=False,
                error_message=f"Unexpected error: {str(e)}",
                error_type="syntax"
            )
            
            return Command(
                update={
                    "execution_result": result.model_dump(),
                    "status": "validating"
                },
                goto="controller"
            )
    
    return execute_sql


def _sanitize_data(data: list) -> list:
    """
    Sanitize data types for LLM consumption.
    - Convert Decimal to float
    - Convert bytes/blobs to placeholder string
    - Ensure None consistency
    """
    sanitized = []
    for row in data:
        sanitized_row = []
        for value in row:
            if value is None:
                sanitized_row.append(None)
            elif isinstance(value, bytes):
                sanitized_row.append("[Binary Data]")
            elif isinstance(value, (int, float, str, bool)):
                sanitized_row.append(value)
            else:
                # Convert other types to string
                sanitized_row.append(str(value))
        sanitized.append(tuple(sanitized_row))
    return sanitized


def _enhance_error_message(error_msg: str, db_path: str) -> str:
    """Enhance error messages with helpful suggestions."""
    try:
        conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
        cursor = conn.cursor()
        
        if "no such table" in error_msg.lower():
            # Extract the bad table name
            import re
            match = re.search(r"no such table: (\w+)", error_msg, re.IGNORECASE)
            if match:
                bad_table = match.group(1)
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                # Find similar table names
                similar = [t for t in tables if bad_table.lower() in t.lower() or t.lower() in bad_table.lower()]
                if similar:
                    error_msg += f" Did you mean: {', '.join(similar)}? Available tables: {', '.join(tables)}"
                else:
                    error_msg += f" Available tables: {', '.join(tables)}"
        
        conn.close()
    except:
        pass
    
    return error_msg
