import sqlite3
import os
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
from config import get_config

config = get_config()
DB_FILE = config.database.knowledge_db_path
OUTPUT_DIR = config.visualization.output_dir

def visualize_graph():
    """
    Reads the knowledge graph from the SQLite database and generates a visual
    representation of the author-paper network.
    """
    if not os.path.exists(DB_FILE):
        print(f"Error: La base de datos no se encuentra en '{DB_FILE}'.")
        print("Por favor, ejecuta primero 'python rag_bbdd_vector.py --force' para crearla.")
        return

    print(f"Conectando a la base de datos en '{DB_FILE}'...")
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # --- Fetch data from database ---
    cursor.execute("SELECT id, title FROM papers")
    papers = cursor.fetchall()

    cursor.execute("SELECT id, name FROM authors")
    authors = cursor.fetchall()

    cursor.execute("SELECT paper_id, author_id FROM paper_authors")
    relations = cursor.fetchall()
    conn.close()

    if not papers or not authors:
        print("No se han encontrado papers o autores en la base de datos.")
        return

    print("Construyendo el grafo...")
    G = nx.Graph()

    # --- Add nodes to the graph ---
    paper_nodes = [f"p_{p['id']}" for p in papers]
    author_nodes = [f"a_{a['id']}" for a in authors]

    G.add_nodes_from(paper_nodes, type='paper')
    G.add_nodes_from(author_nodes, type='author')

    # --- Add edges (relations) ---
    for relation in relations:
        paper_node = f"p_{relation['paper_id']}"
        author_node = f"a_{relation['author_id']}"
        if G.has_node(paper_node) and G.has_node(author_node):
            G.add_edge(paper_node, author_node)

    # --- Prepare for drawing ---
    plt.figure(figsize=config.visualization.figure_size)
    pos = nx.spring_layout(
        G, 
        k=config.visualization.spring_layout_k, 
        iterations=config.visualization.spring_layout_iterations, 
        seed=config.visualization.spring_layout_seed
    )

    # Define node properties
    node_colors = [
        config.visualization.paper_node_color if G.nodes[n]['type'] == 'paper' 
        else config.visualization.author_node_color 
        for n in G.nodes()
    ]
    node_sizes = [
        config.visualization.paper_node_size if G.nodes[n]['type'] == 'paper' 
        else config.visualization.author_node_size 
        for n in G.nodes()
    ]
    
    # Create labels only for authors to avoid clutter
    author_labels = {f"a_{a['id']}": a['name'] for a in authors}

    print("Dibujando el grafo... (esto puede tardar un poco)")
    # Draw the graph
    nx.draw_networkx_edges(G, pos, alpha=config.visualization.edge_alpha)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=config.visualization.node_alpha)
    nx.draw_networkx_labels(G, pos, labels=author_labels, font_size=config.visualization.font_size, font_color='black')

    plt.title("Grafo de Conocimiento: Autores y Papers", size=20)
    plt.axis('off')
    
    # --- Save the image ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    date_str = datetime.now().strftime("%y-%m-%d")
    output_filename = f"grafo_conocimiento_{date_str}.png"
    full_path = os.path.join(OUTPUT_DIR, output_filename)

    print(f"Guardando el grafo en '{full_path}'...")
    plt.savefig(full_path, format="PNG", dpi=config.visualization.dpi)
    plt.close()

    print("\n¡Visualización del grafo creada con éxito!")
    print(f"Puedes abrir el fichero '{full_path}' para ver la red.")

if __name__ == "__main__":
    visualize_graph()