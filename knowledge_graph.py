import sqlite3
import os
from config import get_config

def get_db_file():
    """Get the database file path from configuration."""
    config = get_config()
    return config.database.knowledge_db_path

DB_FILE = get_db_file()

def create_database():
    """
    Creates the initial database schema if it doesn't exist.
    """
    config = get_config()
    # Ensure the directory for the database exists
    os.makedirs(config.database.knowledge_db_dir, exist_ok=True)

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS papers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        summary TEXT,
        source_pdf TEXT UNIQUE NOT NULL,
        publication_date TEXT
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS authors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS paper_authors (
        paper_id INTEGER,
        author_id INTEGER,
        FOREIGN KEY (paper_id) REFERENCES papers (id) ON DELETE CASCADE,
        FOREIGN KEY (author_id) REFERENCES authors (id) ON DELETE CASCADE,
        PRIMARY KEY (paper_id, author_id)
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS citations (
        citing_paper_id INTEGER,
        cited_paper_id INTEGER,
        FOREIGN KEY (citing_paper_id) REFERENCES papers (id) ON DELETE CASCADE,
        FOREIGN KEY (cited_paper_id) REFERENCES papers (id) ON DELETE CASCADE,
        PRIMARY KEY (citing_paper_id, cited_paper_id)
    );
    """)

    conn.commit()
    conn.close()

def get_db_connection():
    """Establishes a connection to the database."""
    return sqlite3.connect(DB_FILE)

def add_paper_with_authors(title, summary, source_pdf, author_names, publication_date=None):
    """
    Adds a paper and its authors to the database, creating relationships.
    Returns the ID of the paper.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO papers (title, summary, source_pdf, publication_date) VALUES (?, ?, ?, ?)",
            (title, summary, source_pdf, publication_date)
        )
        paper_id = cursor.lastrowid
        
        try:
            print(f"    - Added paper '{title}' to Knowledge Graph.")
        except UnicodeEncodeError:
            safe_title = title.encode('ascii', 'ignore').decode('ascii')
            print(f"    - Added paper '{safe_title}' to Knowledge Graph (title has unprintable characters).")

        for author_name in author_names:
            cursor.execute("SELECT id FROM authors WHERE name = ?", (author_name.strip(),))
            author_result = cursor.fetchone()
            if author_result:
                author_id = author_result[0]
            else:
                cursor.execute("INSERT INTO authors (name) VALUES (?)", (author_name.strip(),))
                author_id = cursor.lastrowid
            
            cursor.execute(
                "INSERT OR IGNORE INTO paper_authors (paper_id, author_id) VALUES (?, ?)",
                (paper_id, author_id)
            )
        
        conn.commit()
        return paper_id

    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()

if __name__ == '__main__':
    print("Initializing knowledge graph...")
    create_database()
    print("Knowledge graph ready.")