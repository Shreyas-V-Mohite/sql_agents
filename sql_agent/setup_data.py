"""
Data Setup Script

Downloads the Chinook database and initializes the ChromaDB vector store.
Run this once before using the agent.
"""

import requests
import pathlib
import sqlite3
import csv
import os


def download_chinook_db(local_path: str = "Chinook.db") -> str:
    """Download the Chinook database if not present."""
    url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
    path = pathlib.Path(local_path)
    
    if path.exists():
        print(f"✓ {local_path} already exists")
        return str(path)
    
    print(f"Downloading Chinook database...")
    response = requests.get(url)
    if response.status_code == 200:
        path.write_bytes(response.content)
        print(f"✓ Downloaded to {local_path}")
        return str(path)
    else:
        raise RuntimeError(f"Failed to download: {response.status_code}")


def generate_csv_files(db_path: str = "Chinook.db", output_dir: str = "."):
    """Generate schema_cards.csv, glossary.csv, and query_cookbook.csv from the database."""
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    # Get tables
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = [row["name"] for row in cur.fetchall()]
    print(f"Found tables: {tables}")
    
    # Schema cards
    schema_path = os.path.join(output_dir, "schema_cards.csv")
    with open(schema_path, "w", newline='', encoding='utf-8') as f:
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
                
                writer.writerow([
                    table, col_name, col["type"], bool(col["pk"]),
                    fk_map.get(col_name, ""), sample_val
                ])
    print(f"✓ Generated {schema_path}")
    
    # Glossary
    glossary_path = os.path.join(output_dir, "glossary.csv")
    with open(glossary_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["term", "definition", "example_usage"])
        
        # Add table definitions
        for table in tables:
            writer.writerow([table, f"Table containing {table} data", f"SELECT * FROM {table}"])
        
        # Add business terms
        business_terms = [
            ("Track", "A single song or audio recording", "SELECT Name FROM Track WHERE AlbumId = 1"),
            ("Album", "A collection of tracks released together", "SELECT Title FROM Album WHERE ArtistId = 1"),
            ("Artist", "A musician or band", "SELECT Name FROM Artist WHERE Name LIKE '%AC/DC%'"),
            ("Genre", "Music category like Rock, Jazz, Pop", "SELECT Name FROM Genre"),
            ("Invoice", "A sales transaction record", "SELECT Total FROM Invoice WHERE CustomerId = 1"),
            ("Customer", "A person who purchased music", "SELECT FirstName, LastName FROM Customer"),
            ("Employee", "A staff member of the music store", "SELECT FirstName FROM Employee WHERE Title LIKE '%Manager%'"),
            ("Playlist", "A user-created collection of tracks", "SELECT Name FROM Playlist"),
            ("MediaType", "Format of the track (MP3, AAC, etc.)", "SELECT Name FROM MediaType"),
            ("Total Sales", "Sum of Invoice.Total for revenue calculations", "SELECT SUM(Total) FROM Invoice"),
            ("Top Tracks", "Tracks ordered by sales or popularity", "SELECT t.Name, COUNT(*) FROM Track t JOIN InvoiceLine il ON t.TrackId = il.TrackId GROUP BY t.TrackId ORDER BY COUNT(*) DESC"),
        ]
        for term, definition, example in business_terms:
            writer.writerow([term, definition, example])
    print(f"✓ Generated {glossary_path}")
    
    # Query cookbook
    cookbook_path = os.path.join(output_dir, "query_cookbook.csv")
    with open(cookbook_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "description", "sql_query", "dialect", "verified"])
        
        cookbook_queries = [
            ("Q001", "Total sales by country",
             "SELECT BillingCountry, SUM(Total) AS TotalSales FROM Invoice GROUP BY BillingCountry ORDER BY TotalSales DESC;",
             "SQLite", "TRUE"),
            ("Q002", "Top genres by units sold",
             "SELECT g.Name AS Genre, COUNT(il.InvoiceLineId) AS UnitsSold FROM InvoiceLine il JOIN Track t ON il.TrackId = t.TrackId JOIN Genre g ON t.GenreId = g.GenreId GROUP BY g.Name ORDER BY UnitsSold DESC LIMIT 10;",
             "SQLite", "TRUE"),
            ("Q003", "Tracks by a specific artist",
             "SELECT t.Name AS TrackName, al.Title AS Album FROM Track t JOIN Album al ON t.AlbumId = al.AlbumId JOIN Artist ar ON al.ArtistId = ar.ArtistId WHERE ar.Name = 'AC/DC';",
             "SQLite", "TRUE"),
            ("Q004", "Average track length by genre",
             "SELECT g.Name AS Genre, AVG(t.Milliseconds)/1000.0 AS AvgSeconds FROM Track t JOIN Genre g ON t.GenreId = g.GenreId GROUP BY g.Name ORDER BY AvgSeconds DESC;",
             "SQLite", "TRUE"),
            ("Q005", "Customers who spent the most",
             "SELECT c.FirstName || ' ' || c.LastName AS Customer, SUM(i.Total) AS TotalSpent FROM Customer c JOIN Invoice i ON c.CustomerId = i.CustomerId GROUP BY c.CustomerId ORDER BY TotalSpent DESC LIMIT 10;",
             "SQLite", "TRUE"),
            ("Q006", "Employees and their sales",
             "SELECT e.FirstName || ' ' || e.LastName AS Employee, COUNT(i.InvoiceId) AS SalesCount, SUM(i.Total) AS TotalSales FROM Employee e JOIN Customer c ON e.EmployeeId = c.SupportRepId JOIN Invoice i ON c.CustomerId = i.CustomerId GROUP BY e.EmployeeId;",
             "SQLite", "TRUE"),
            ("Q007", "Tracks in a playlist",
             "SELECT p.Name AS Playlist, t.Name AS Track FROM Playlist p JOIN PlaylistTrack pt ON p.PlaylistId = pt.PlaylistId JOIN Track t ON pt.TrackId = t.TrackId WHERE p.Name = 'Music';",
             "SQLite", "TRUE"),
            ("Q008", "Albums with track count",
             "SELECT al.Title AS Album, ar.Name AS Artist, COUNT(t.TrackId) AS TrackCount FROM Album al JOIN Artist ar ON al.ArtistId = ar.ArtistId JOIN Track t ON al.AlbumId = t.AlbumId GROUP BY al.AlbumId ORDER BY TrackCount DESC LIMIT 10;",
             "SQLite", "TRUE"),
        ]
        for row in cookbook_queries:
            writer.writerow(row)
    print(f"✓ Generated {cookbook_path}")
    
    conn.close()
    return schema_path, glossary_path, cookbook_path


def initialize_chromadb(
    schema_path: str = "schema_cards.csv",
    glossary_path: str = "glossary.csv",
    cookbook_path: str = "query_cookbook.csv",
    persist_dir: str = "./chroma_db"
):
    """Initialize ChromaDB with the CSV data."""
    from sql_agent.utils.tools import initialize_vector_store
    
    vector_store = initialize_vector_store(
        schema_cards_path=schema_path,
        glossary_path=glossary_path,
        cookbook_path=cookbook_path,
        persist_directory=persist_dir
    )
    print(f"✓ ChromaDB initialized at {persist_dir}")
    return vector_store


def setup_all(db_path: str = "Chinook.db", chroma_dir: str = "./chroma_db"):
    """Run complete setup: download DB, generate CSVs, initialize ChromaDB."""
    print("=" * 50)
    print("SQL Agent Setup")
    print("=" * 50)
    
    # Step 1: Download database
    download_chinook_db(db_path)
    
    # Step 2: Generate CSV files
    schema_path, glossary_path, cookbook_path = generate_csv_files(db_path)
    
    # Step 3: Initialize ChromaDB
    initialize_chromadb(schema_path, glossary_path, cookbook_path, chroma_dir)
    
    print("=" * 50)
    print("✓ Setup complete!")
    print("=" * 50)


if __name__ == "__main__":
    setup_all()
