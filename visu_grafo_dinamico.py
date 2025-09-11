import sqlite3
import os
import re
from pyvis.network import Network
from datetime import datetime
from config import get_config

config = get_config()
DB_FILE = config.database.knowledge_db_path
OUTPUT_DIR = config.visualization.output_dir

def generate_dynamic_graph():
    """
    Reads the knowledge graph from the SQLite database and generates an
    interactive HTML file visualizing the author-paper network.
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
    cursor.execute("SELECT id, title, summary FROM papers")
    papers = cursor.fetchall()

    cursor.execute("SELECT id, name FROM authors")
    authors = cursor.fetchall()

    cursor.execute("SELECT paper_id, author_id FROM paper_authors")
    relations = cursor.fetchall()
    conn.close()

    if not papers or not authors:
        print("No se han encontrado papers o autores en la base de datos.")
        return

    print("Construyendo el grafo dinámico...")
    
    title = "Grafo de Conocimiento: Autores y Papers"
    # --- Create a Pyvis Network instance ---
    net = Network(
        height="900px", 
        width="100%", 
        notebook=False, 
        directed=False, 
        bgcolor="#222222", 
        font_color="white",
        heading=title
    )

    # --- Add nodes and edges ---
    for paper in papers:
        paper_id = f"p_{paper['id']}"
        label = paper['title']
        display_label = (label[:40] + '...') if len(label) > 40 else label
        net.add_node(paper_id, label=display_label, title=label, color=config.visualization.paper_node_color, size=config.visualization.paper_node_size)

    for author in authors:
        author_id = f"a_{author['id']}"
        net.add_node(author_id, label=author['name'], title=author['name'], color=config.visualization.author_node_color, size=config.visualization.author_node_size)

    for relation in relations:
        paper_node = f"p_{relation['paper_id']}"
        author_node = f"a_{relation['author_id']}"
        net.add_edge(paper_node, author_node)

    # --- Configure physics and save the graph ---
    print("Configurando la visualización...")
    net.show_buttons(filter_='physics')
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    date_str = datetime.now().strftime("%y-%m-%d")
    output_filename = f"grafo_dinamico_{date_str}.html"
    full_path = os.path.join(OUTPUT_DIR, output_filename)

    print(f"Guardando el grafo interactivo en '{full_path}'...")
    net.save_graph(full_path)

    # --- Post-process HTML to fix duplicate title bug ---
    try:
        print("Iniciando post-procesamiento para corregir título...")
        with open(full_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # --- DEBUGGING: Print the head of the HTML file ---
        print("--- DEBUG: Inicio del fichero HTML ---")
        print(html_content[:500])
        print("--- DEBUG: Fin del inicio del fichero HTML ---")

        # Regex to find the title block flexibly
        pattern = re.compile(r'<center>\s*<h1>.*?</h1>\s*</center>', re.DOTALL)
        
        # Replace only the first occurrence found by the regex
        html_content_fixed, num_replacements = pattern.subn("", html_content, count=1)

        if num_replacements > 0:
            print("Corrección de título aplicada. Guardando fichero modificado...")
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(html_content_fixed)
        else:
            print("No se ha encontrado un título duplicado que corregir con el patrón esperado.")

    except Exception as e:
        print(f"No se pudo post-procesar el fichero HTML: {e}")

    print("\n¡Grafo dinámico creado con éxito!")
    print(f"Puedes abrir el fichero '{full_path}' en tu navegador para ver la red interactiva.")

if __name__ == "__main__":
    generate_dynamic_graph()
