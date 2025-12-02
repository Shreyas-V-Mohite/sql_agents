# -*- coding: utf-8 -*-
"""
SQL Agent - Multi-Agent Text-to-SQL System
Google Colab Version

Run this notebook cell by cell in Google Colab.
Requirements: GPU runtime (T4 free tier works)

To use:
1. Open Google Colab: https://colab.research.google.com
2. Go to Runtime > Change runtime type > Select GPU (T4)
3. Copy each cell (marked with # %% [markdown] or # %%) and run
"""

# %% [markdown]
"""
# 🚀 SQL Agent Setup
Multi-agent system using Qwen2.5-Coder for Text-to-SQL

**Runtime Required:** GPU (T4 or better)
"""

# %% [markdown]
"""
## Cell 1: Install Dependencies
"""

# %%
# Install required packages
!pip install -q torch transformers accelerate peft
!pip install -q langgraph langchain-core langchain-chroma chromadb
!pip install -q pandas pydantic requests

print("✓ Dependencies installed")

# %% [markdown]
"""
## Cell 2: Download Chinook Database
"""

# %%
import requests
import pathlib

url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
local_path = pathlib.Path("Chinook.db")

if local_path.exists():
    print(f"✓ {local_path} already exists")
else:
    response = requests.get(url)
    if response.status_code == 200:
        local_path.write_bytes(response.content)
        print(f"✓ Downloaded Chinook.db")
    else:
        raise RuntimeError(f"Failed to download: {response.status_code}")

# %% [markdown]
"""
## Cell 3: Generate Schema CSV Files
"""

# %%
import sqlite3
import csv

conn = sqlite3.connect("Chinook.db")
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# Get tables
cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
tables = [row["name"] for row in cur.fetchall()]
print(f"Tables: {tables}")

# Schema cards
with open("schema_cards.csv", "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["table_name", "column_name", "data_type", "is_primary_key", "foreign_key_reference", "sample_value"])
    
    for table in tables:
        cur.execute(f"PRAGMA table_info('{table}');")
        cols = cur.fetchall()
        cur.execute(f"PRAGMA foreign_key_list('{table}');")
        fks = cur.fetchall()
        fk_map = {fk["from"]: f"{fk['table']}({fk['to']})" for fk in fks}
        
        for col in cols:
            col_name = col["name"]
            try:
                cur.execute(f"SELECT [{col_name}] FROM [{table}] WHERE [{col_name}] IS NOT NULL LIMIT 1;")
                sample = cur.fetchone()
                sample_val = sample[0] if sample else ""
            except:
                sample_val = ""
            writer.writerow([table, col_name, col["type"], bool(col["pk"]), fk_map.get(col_name, ""), sample_val])

print("✓ Generated schema_cards.csv")

# Glossary
with open("glossary.csv", "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["term", "definition", "example_usage"])
    for table in tables:
        writer.writerow([table, f"Table containing {table} data", f"SELECT * FROM {table}"])
    # Business terms
    business_terms = [
        ("Track", "A single song or audio recording", "SELECT Name FROM Track WHERE AlbumId = 1"),
        ("Album", "A collection of tracks released together", "SELECT Title FROM Album WHERE ArtistId = 1"),
        ("Artist", "A musician or band", "SELECT Name FROM Artist WHERE Name LIKE '%AC/DC%'"),
        ("Genre", "Music category like Rock, Jazz, Pop", "SELECT Name FROM Genre"),
        ("Invoice", "A sales transaction record", "SELECT Total FROM Invoice WHERE CustomerId = 1"),
        ("Total Sales", "Sum of Invoice.Total for revenue calculations", "SELECT SUM(Total) FROM Invoice"),
    ]
    for term, definition, example in business_terms:
        writer.writerow([term, definition, example])

print("✓ Generated glossary.csv")

# Query cookbook
with open("query_cookbook.csv", "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["query_id", "description", "sql_query", "dialect", "verified"])
    cookbook_queries = [
        ("Q001", "Total sales by country", "SELECT BillingCountry, SUM(Total) AS TotalSales FROM Invoice GROUP BY BillingCountry ORDER BY TotalSales DESC;", "SQLite", "TRUE"),
        ("Q002", "Top genres by units sold", "SELECT g.Name AS Genre, COUNT(il.InvoiceLineId) AS UnitsSold FROM InvoiceLine il JOIN Track t ON il.TrackId = t.TrackId JOIN Genre g ON t.GenreId = g.GenreId GROUP BY g.Name ORDER BY UnitsSold DESC LIMIT 10;", "SQLite", "TRUE"),
        ("Q003", "Tracks by a specific artist", "SELECT t.Name AS TrackName, al.Title AS Album FROM Track t JOIN Album al ON t.AlbumId = al.AlbumId JOIN Artist ar ON al.ArtistId = ar.ArtistId WHERE ar.Name = 'AC/DC';", "SQLite", "TRUE"),
        ("Q004", "Average track length by genre", "SELECT g.Name AS Genre, AVG(t.Milliseconds)/1000.0 AS AvgSeconds FROM Track t JOIN Genre g ON t.GenreId = g.GenreId GROUP BY g.Name ORDER BY AvgSeconds DESC;", "SQLite", "TRUE"),
    ]
    for row in cookbook_queries:
        writer.writerow(row)

print("✓ Generated query_cookbook.csv")
conn.close()

# %% [markdown]
"""
## Cell 4: Initialize ChromaDB Vector Store
"""

# %%
import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma

# Use a simple embedding function for Colab (no OpenAI needed)
from chromadb.utils import embedding_functions
import chromadb

# Create documents
docs = []

# Schema cards
schema_df = pd.read_csv("schema_cards.csv")
for _, row in schema_df.iterrows():
    content = f"Table: {row['table_name']}\nColumn: {row['column_name']}\nData type: {row['data_type']}\nIs PK: {row['is_primary_key']}\nFK ref: {row['foreign_key_reference']}\nSample: {row['sample_value']}"
    docs.append({"content": content, "metadata": {"source": "schema", "table": row["table_name"]}})

# Glossary
glossary_df = pd.read_csv("glossary.csv")
for _, row in glossary_df.iterrows():
    content = f"Term: {row['term']}\nDefinition: {row['definition']}\nExample: {row['example_usage']}"
    docs.append({"content": content, "metadata": {"source": "glossary", "term": row["term"]}})

# Cookbook
cookbook_df = pd.read_csv("query_cookbook.csv")
for _, row in cookbook_df.iterrows():
    content = f"Query: {row['description']}\nSQL: {row['sql_query']}"
    docs.append({"content": content, "metadata": {"source": "cookbook", "query_id": row["query_id"]}})

# Create ChromaDB with default embeddings
client = chromadb.Client()
collection = client.create_collection(
    name="chinook_schema",
    embedding_function=embedding_functions.DefaultEmbeddingFunction()
)

# Add documents
collection.add(
    documents=[d["content"] for d in docs],
    metadatas=[d["metadata"] for d in docs],
    ids=[f"doc_{i}" for i in range(len(docs))]
)

print(f"✓ Indexed {len(docs)} documents into ChromaDB")

# %% [markdown]
"""
## Cell 5: Load Qwen Model
"""

# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"

print(f"Loading {MODEL_NAME}...")
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

print("✓ Model loaded")

# %% [markdown]
"""
## Cell 6: Define Helper Functions
"""

# %%
def generate_response(prompt: str, max_tokens: int = 1024) -> str:
    """Generate response from Qwen model."""
    messages = [
        {"role": "system", "content": "You are a helpful SQL assistant."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated = outputs[0][len(inputs.input_ids[0]):]
    return tokenizer.decode(generated, skip_special_tokens=True)


def retrieve_context(query: str, k: int = 5) -> str:
    """Retrieve relevant context from ChromaDB."""
    results = collection.query(query_texts=[query], n_results=k)
    return "\n\n".join(results["documents"][0])


def execute_sql(sql: str, db_path: str = "Chinook.db") -> dict:
    """Execute SQL safely in read-only mode."""
    try:
        conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql)
        
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchmany(50)  # Limit to 50 rows
        data = [tuple(row) for row in rows]
        
        conn.close()
        return {"success": True, "columns": columns, "data": data, "row_count": len(data)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def search_column_values(table: str, column: str, search_term: str) -> list:
    """Search for values in a database column."""
    try:
        conn = sqlite3.connect('file:Chinook.db?mode=ro', uri=True)
        cursor = conn.cursor()
        cursor.execute(f'SELECT DISTINCT "{column}" FROM "{table}" WHERE "{column}" LIKE ? LIMIT 10', (f'%{search_term}%',))
        results = [row[0] for row in cursor.fetchall() if row[0]]
        conn.close()
        return results
    except Exception as e:
        return [f"Error: {e}"]

print("✓ Helper functions defined")

# %% [markdown]
"""
## Cell 7: Define Agent Functions
"""

# %%
from datetime import datetime
import json
import re

def planner_agent(question: str) -> dict:
    """
    Planner Agent: Analyzes question and creates a query plan.
    """
    # Retrieve context
    context = retrieve_context(question)
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    prompt = f"""You are a SQL query planner for a music database (Chinook).
Analyze the user's question and create a structured query plan.

## Current Date: {current_date}

## Schema Context:
{context}

## User Question:
{question}

## Instructions:
Create a JSON query plan with these fields:
- primary_entity: Main table the user is asking about
- tables_involved: List of all tables needed
- join_path: How tables connect (e.g., "Track -> Album -> Artist")
- filters: List of WHERE conditions
- aggregations: Any COUNT, SUM, AVG needed (or null)
- ordering: ORDER BY clause (or null)
- limit: Result limit (default 50)
- reasoning: Why this plan was chosen

Return ONLY valid JSON, no other text.
"""
    
    response = generate_response(prompt)
    
    # Parse JSON from response
    try:
        # Try to extract JSON
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            plan = json.loads(json_match.group())
        else:
            plan = json.loads(response)
        return {"success": True, "plan": plan}
    except:
        return {"success": False, "error": f"Failed to parse plan: {response[:500]}"}


def synthesizer_agent(plan: dict, schema_context: str, error_feedback: str = None) -> dict:
    """
    Synthesizer Agent: Generates SQL from the query plan.
    """
    prompt = f"""You are a SQL code generator for SQLite databases.
Generate a SELECT query based on the provided plan.

## Query Plan:
{json.dumps(plan, indent=2)}

## Schema Context:
{schema_context}

## Previous Error (if retrying):
{error_feedback or 'None - first attempt'}

## SQLite Rules:
- Use || for string concatenation (NOT CONCAT)
- Use strftime('%Y', date) for year (NOT YEAR())
- Use date('now') for current date
- ONLY generate SELECT statements
- Always include LIMIT clause

Return ONLY the SQL query, no explanations.
"""
    
    sql = generate_response(prompt, max_tokens=512).strip()
    
    # Clean markdown
    if "```" in sql:
        sql = re.search(r'```(?:sql)?\s*(.*?)\s*```', sql, re.DOTALL)
        sql = sql.group(1) if sql else sql
    
    # Safety check
    sql_upper = sql.upper()
    if any(kw in sql_upper for kw in ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER"]):
        return {"success": False, "error": "Security: Only SELECT allowed"}
    
    return {"success": True, "sql": sql}


def controller_agent(question: str, plan: dict, sql: str, result: dict) -> str:
    """
    Controller Agent: Generates final answer from results.
    """
    if not result["success"]:
        return f"Error executing query: {result.get('error', 'Unknown error')}\n\nSQL attempted:\n```sql\n{sql}\n```"
    
    data = result.get("data", [])
    columns = result.get("columns", [])
    row_count = result.get("row_count", 0)
    
    if row_count == 0:
        return "The query executed successfully but returned no results."
    
    # Format data preview
    if row_count <= 10:
        data_str = "\n".join([str(row) for row in data])
    else:
        data_str = "\n".join([str(row) for row in data[:10]]) + f"\n... and {row_count - 10} more rows"
    
    prompt = f"""Based on the SQL results, provide a clear answer to the user's question.

## Question: {question}

## SQL: 
```sql
{sql}
```

## Results:
Columns: {columns}
Rows: {row_count}

Data:
{data_str}

Provide a concise, natural language answer.
"""
    
    return generate_response(prompt, max_tokens=512)

print("✓ Agent functions defined")

# %% [markdown]
"""
## Cell 8: Main Pipeline Function
"""

# %%
def ask_sql_agent(question: str, max_retries: int = 3) -> str:
    """
    Main pipeline: Question -> Plan -> SQL -> Execute -> Answer
    """
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}\n")
    
    # Step 1: Plan
    print("🔍 [Planner] Analyzing question...")
    plan_result = planner_agent(question)
    
    if not plan_result["success"]:
        return f"Planning failed: {plan_result['error']}"
    
    plan = plan_result["plan"]
    print(f"✓ Plan created: {plan.get('primary_entity')} from {plan.get('tables_involved')}")
    
    # Get schema context for synthesizer
    schema_context = retrieve_context(question)
    
    # Retry loop
    error_feedback = None
    for attempt in range(max_retries):
        # Step 2: Synthesize SQL
        print(f"\n📝 [Synthesizer] Generating SQL (attempt {attempt + 1})...")
        sql_result = synthesizer_agent(plan, schema_context, error_feedback)
        
        if not sql_result["success"]:
            error_feedback = sql_result["error"]
            print(f"✗ Synthesis failed: {error_feedback}")
            continue
        
        sql = sql_result["sql"]
        print(f"✓ SQL generated:\n{sql}\n")
        
        # Step 3: Execute
        print("⚡ [Executor] Running query...")
        exec_result = execute_sql(sql)
        
        if exec_result["success"]:
            print(f"✓ Query returned {exec_result['row_count']} rows")
            
            # Step 4: Generate answer
            print("\n💬 [Controller] Generating answer...")
            answer = controller_agent(question, plan, sql, exec_result)
            
            print(f"\n{'='*60}")
            print("ANSWER:")
            print(f"{'='*60}")
            print(answer)
            return answer
        else:
            error_feedback = exec_result["error"]
            print(f"✗ Execution failed: {error_feedback}")
    
    return f"Failed after {max_retries} attempts. Last error: {error_feedback}"

print("✓ Pipeline ready!")

# %% [markdown]
"""
## Cell 9: Test the Agent! 🎉
"""

# %%
# Test queries - run these one at a time

# Query 1: Simple count
ask_sql_agent("How many tracks are in the database?")

# %%
# Query 2: Aggregation
ask_sql_agent("What are the top 5 genres by number of tracks?")

# %%
# Query 3: Join query
ask_sql_agent("List all tracks by AC/DC")

# %%
# Query 4: Complex aggregation
ask_sql_agent("Which country has the highest total sales?")

# %%
# Query 5: Your own question!
ask_sql_agent("What is the average track length by genre?")

# %% [markdown]
"""
## Cell 10: Interactive Mode (Optional)
"""

# %%
# Run this cell for interactive mode
print("🎵 SQL Agent - Interactive Mode")
print("Type 'quit' to exit\n")

while True:
    question = input("Your question: ").strip()
    if question.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        break
    if question:
        ask_sql_agent(question)
