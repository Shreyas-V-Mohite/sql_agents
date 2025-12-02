"""
Tools for the SQL Agent pipeline.
Includes RAG retrieval and database value lookup.
"""

import sqlite3
from typing import List, Tuple, Optional
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


# ============================================================================
# Vector Store Setup (ChromaDB)
# ============================================================================

def create_vector_store(persist_directory: str = "./chroma_db") -> Chroma:
    """Create or load ChromaDB vector store."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vector_store = Chroma(
        collection_name="chinook_schema",
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    return vector_store


def initialize_vector_store(
    schema_cards_path: str,
    glossary_path: str,
    cookbook_path: str,
    persist_directory: str = "./chroma_db"
) -> Chroma:
    """Initialize vector store with schema, glossary, and cookbook data."""
    import pandas as pd
    from langchain_core.documents import Document
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docs = []
    
    # Load schema cards
    schema_df = pd.read_csv(schema_cards_path)
    for _, row in schema_df.iterrows():
        content = (
            f"Table: {row['table_name']}\n"
            f"Column: {row['column_name']}\n"
            f"Data type: {row['data_type']}\n"
            f"Is Primary Key: {row['is_primary_key']}\n"
            f"Foreign Key Reference: {row['foreign_key_reference']}\n"
            f"Sample value: {row['sample_value']}"
        )
        docs.append(Document(
            page_content=content,
            metadata={"source": "schema", "table": row["table_name"], "column": row["column_name"]}
        ))
    
    # Load glossary
    glossary_df = pd.read_csv(glossary_path)
    for _, row in glossary_df.iterrows():
        content = (
            f"Term: {row['term']}\n"
            f"Definition: {row['definition']}\n"
            f"Example: {row['example_usage']}"
        )
        docs.append(Document(
            page_content=content,
            metadata={"source": "glossary", "term": row["term"]}
        ))
    
    # Load cookbook
    cookbook_df = pd.read_csv(cookbook_path)
    for _, row in cookbook_df.iterrows():
        content = (
            f"Query Description: {row['description']}\n"
            f"SQL: {row['sql_query']}\n"
            f"Verified: {row['verified']}"
        )
        docs.append(Document(
            page_content=content,
            metadata={"source": "cookbook", "query_id": row["query_id"]}
        ))
    
    # Create vector store
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="chinook_schema",
        persist_directory=persist_directory,
    )
    
    print(f"Indexed {len(docs)} documents into ChromaDB")
    return vector_store


# ============================================================================
# RAG Retrieval Tools
# ============================================================================

class RAGTools:
    """RAG tools for schema and context retrieval."""
    
    def __init__(self, vector_store: Chroma):
        self.vector_store = vector_store
    
    def retrieve_schema_context(self, query: str, k: int = 5) -> str:
        """Retrieve relevant schema information."""
        results = self.vector_store.similarity_search(
            query,
            k=k,
            filter={"source": "schema"}
        )
        if not results:
            return "No schema context found."
        return "\n\n".join([doc.page_content for doc in results])
    
    def retrieve_glossary_context(self, query: str, k: int = 3) -> str:
        """Retrieve relevant business terms."""
        results = self.vector_store.similarity_search(
            query,
            k=k,
            filter={"source": "glossary"}
        )
        if not results:
            return "No glossary context found."
        return "\n\n".join([doc.page_content for doc in results])
    
    def retrieve_cookbook_examples(self, query: str, k: int = 2) -> str:
        """Retrieve similar query examples."""
        results = self.vector_store.similarity_search(
            query,
            k=k,
            filter={"source": "cookbook"}
        )
        if not results:
            return "No similar queries found."
        return "\n\n".join([doc.page_content for doc in results])


# ============================================================================
# Database Value Lookup Tool
# ============================================================================

class DatabaseTools:
    """Tools for direct database interaction."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def search_column_values(
        self, 
        table: str, 
        column: str, 
        search_term: str,
        limit: int = 10
    ) -> List[str]:
        """
        Search for values in a column that match the search term.
        Uses LIKE for fuzzy matching.
        """
        try:
            conn = sqlite3.connect(f'file:{self.db_path}?mode=ro', uri=True)
            cursor = conn.cursor()
            
            # Sanitize inputs (basic protection)
            safe_table = table.replace('"', '""')
            safe_column = column.replace('"', '""')
            
            query = f'''
                SELECT DISTINCT "{safe_column}" 
                FROM "{safe_table}" 
                WHERE "{safe_column}" LIKE ? 
                LIMIT ?
            '''
            cursor.execute(query, (f'%{search_term}%', limit))
            results = [row[0] for row in cursor.fetchall() if row[0]]
            conn.close()
            return results
        except Exception as e:
            return [f"Error: {str(e)}"]
    
    def get_distinct_values(
        self, 
        table: str, 
        column: str, 
        limit: int = 20
    ) -> List[str]:
        """Get distinct values from a column."""
        try:
            conn = sqlite3.connect(f'file:{self.db_path}?mode=ro', uri=True)
            cursor = conn.cursor()
            
            safe_table = table.replace('"', '""')
            safe_column = column.replace('"', '""')
            
            query = f'SELECT DISTINCT "{safe_column}" FROM "{safe_table}" LIMIT ?'
            cursor.execute(query, (limit,))
            results = [row[0] for row in cursor.fetchall() if row[0]]
            conn.close()
            return results
        except Exception as e:
            return [f"Error: {str(e)}"]
    
    def get_table_names(self) -> List[str]:
        """Get all table names in the database."""
        try:
            conn = sqlite3.connect(f'file:{self.db_path}?mode=ro', uri=True)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()
            return tables
        except Exception as e:
            return [f"Error: {str(e)}"]
