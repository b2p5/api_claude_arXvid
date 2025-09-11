import argparse
import os
import sqlite3

from config import get_config

def get_db_file():
    """Get database file path from configuration."""
    config = get_config()
    return config.database.knowledge_db_path

DB_FILE = get_db_file()

def get_db_connection():
    """Establishes a connection to the database."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def list_papers(args):
    """Lists all papers in the knowledge graph, ordered by ID."""
    print("--- All Papers in Knowledge Graph (by ID) ---")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, summary, publication_date FROM papers ORDER BY id")
    papers = cursor.fetchall()
    
    if not papers:
        print("No papers found in the knowledge graph.")
        return

    for paper in papers:
        date_str = f" ({paper['publication_date']})" if paper['publication_date'] else ""
        print(f"\n[ID: {paper['id']}] Title: {paper['title']}{date_str}")
        print(f"  Summary: {paper['summary']}")
    
    conn.close()

def list_by_date(args):
    """Lists all papers in the knowledge graph, ordered by date."""
    order = 'DESC' if args.newest_first else 'ASC'
    print(f"--- All Papers in Knowledge Graph (by Date, {'Newest' if order == 'DESC' else 'Oldest'} First) ---")
    conn = get_db_connection()
    cursor = conn.cursor()
    # Sorts NULLs last
    cursor.execute(f"SELECT id, title, summary, publication_date FROM papers ORDER BY publication_date IS NULL, publication_date {order}")
    papers = cursor.fetchall()
    
    if not papers:
        print("No papers found in the knowledge graph.")
        return

    for paper in papers:
        date_str = f" ({paper['publication_date']})" if paper['publication_date'] else " (No Date)"
        print(f"\n[ID: {paper['id']}] {date_str} - {paper['title']}")
        print(f"  Summary: {paper['summary']}")
    
    conn.close()


def find_by_author(args):
    """Finds papers written by a specific author."""
    author_name = args.name
    print(f"--- Papers by '{author_name}' ---")
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = """
    SELECT p.id, p.title, p.summary, p.publication_date
    FROM papers p
    JOIN paper_authors pa ON p.id = pa.paper_id
    JOIN authors a ON a.id = pa.author_id
    WHERE a.name LIKE ?
    ORDER BY p.publication_date DESC
    """
    
    cursor.execute(query, (f"%{author_name}%",))
    papers = cursor.fetchall()

    if not papers:
        print(f"No papers found for an author matching '{author_name}'.")
        return

    for paper in papers:
        date_str = f" ({paper['publication_date']})" if paper['publication_date'] else ""
        print(f"\n[ID: {paper['id']}] Title: {paper['title']}{date_str}")
        print(f"  Summary: {paper['summary']}")

    conn.close()

def list_authors(args):
    """Lists all unique authors in the knowledge graph."""
    print("--- All Authors in Knowledge Graph ---")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM authors ORDER BY name")
    authors = cursor.fetchall()

    if not authors:
        print("No authors found in the knowledge graph.")
        return

    for author in authors:
        print(f"- {author['name']}")
        
    conn.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze the research paper knowledge graph.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # 'list-papers' command
    parser_list = subparsers.add_parser("list-papers", help="List all papers in the database, ordered by ID.")
    parser_list.set_defaults(func=list_papers)

    # 'list-by-date' command
    parser_date = subparsers.add_parser("list-by-date", help="List all papers chronologically.")
    parser_date.add_argument("--oldest-first", dest='newest_first', action='store_false', help="Sort by oldest first (default is newest first).")
    parser_date.set_defaults(func=list_by_date)

    # 'find-by-author' command
    parser_find = subparsers.add_parser("find-by-author", help="Find papers by a specific author.")
    parser_find.add_argument("name", type=str, help="The name of the author to search for.")
    parser_find.set_defaults(func=find_by_author)

    # 'list-authors' command
    parser_authors = subparsers.add_parser("list-authors", help="List all unique authors in the database.")
    parser_authors.set_defaults(func=list_authors)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
