import os
import shutil
import argparse
import chromadb
import dotenv
import json
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_deepseek import ChatDeepSeek
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from config import get_config
from logger import get_logger, log_info, log_error, log_warning
from pdf_validator import validate_pdf
from llm_utils import extract_paper_entities_safe
from retry_utils import safe_execute

# Import and initialize the knowledge graph
import knowledge_graph

# --- LLM for Entity Extraction ---
config = get_config()
dotenv.load_dotenv()
llm = ChatDeepSeek(model=config.models.llm_model)

extraction_template = """
From the beginning of the following research paper, extract the title, authors, publication date, and a brief summary.
Provide the output in a clean JSON format with the keys "title", "authors" (as a list of strings), "publication_date" (as a string in YYYY-MM-DD format), and "summary".

PAPER_TEXT:
---
{text}
---

JSON_OUTPUT:
"""
prompt = PromptTemplate.from_template(extraction_template)
entity_extraction_chain = prompt | llm | StrOutputParser()

def extract_paper_details(text, pdf_path="unknown"):
    """Uses an LLM to extract structured details from the paper's text."""
    logger = get_logger()
    logger.log_operation_start("LLM entity extraction", pdf_path=pdf_path)
    
    try:
        # Use the new robust LLM extraction system
        data, errors, warnings = extract_paper_entities_safe(text)
        
        if errors:
            log_warning("Entity extraction had errors", pdf_path=pdf_path, error_count=len(errors))
        
        if warnings:
            log_info("Entity extraction had warnings", pdf_path=pdf_path, warning_count=len(warnings))
        
        logger.log_operation_success("LLM entity extraction", pdf_path=pdf_path)
        return data
        
    except Exception as e:
        logger.log_operation_failure("LLM entity extraction", e, pdf_path=pdf_path)
        # Return fallback data
        return {
            "title": "Unknown",
            "authors": ["Unknown"],
            "summary": "Could not extract summary",
            "publication_date": None
        }

def update_databases(root_docs_path, db_path, force):
    """
    Creates or updates both the ChromaDB vector database and the Knowledge Graph
    from all PDF documents found in subdirectories of the root_docs_path.
    """
    if os.path.exists(db_path) and force:
        print(f"--force flag detected. Deleting existing vector database at {db_path}...")
        shutil.rmtree(db_path)
        if os.path.exists(knowledge_graph.DB_FILE):
            os.remove(knowledge_graph.DB_FILE)
            print(f"Deleted existing knowledge graph at {knowledge_graph.DB_FILE}.")

    print(f"Initializing vector database at {db_path}...")
    vector_db = chromadb.PersistentClient(path=db_path)
    collection = vector_db.get_or_create_collection(name=config.database.vector_collection_name)
    
    knowledge_graph.create_database()
    kg_conn = knowledge_graph.get_db_connection()

    total_new_pdfs_kg = 0
    total_new_pdfs_vdb = 0

    if not os.path.exists(root_docs_path):
        print(f"Root documents directory not found: {root_docs_path}")
        return

    for concept_dir in os.listdir(root_docs_path):
        concept_path = os.path.join(root_docs_path, concept_dir)
        if not os.path.isdir(concept_path):
            continue

        print(f"\nProcessing directory: {concept_path}")
        for pdf_file in os.listdir(concept_path):
            if not pdf_file.endswith(".pdf"):
                continue

            pdf_path = os.path.join(concept_path, pdf_file)
            log_info(f"Processing file", filename=pdf_file)

            documents = None
            
            try:
                cursor = kg_conn.cursor()
                cursor.execute("SELECT id FROM papers WHERE source_pdf = ?", (pdf_path,))
                paper_exists_in_kg = cursor.fetchone()

                if not paper_exists_in_kg:
                    log_info("KG entry not found - processing for Knowledge Graph", filename=pdf_file)
                    
                    # First validate the PDF
                    validation_result = validate_pdf(pdf_path)
                    if not validation_result.is_valid:
                        log_warning("PDF validation failed - skipping KG processing", 
                                  filename=pdf_file, 
                                  error_count=len(validation_result.errors))
                        continue
                    
                    # Load PDF with error handling
                    def load_pdf():
                        loader = PyPDFLoader(pdf_path)
                        return loader.load()
                    
                    documents = safe_execute(
                        load_pdf,
                        "PDF loading",
                        default_return=None,
                        filename=pdf_file
                    )
                    
                    if documents:
                        paper_text = " ".join([doc.page_content for doc in documents])
                        details = extract_paper_details(paper_text, pdf_path)
                        
                        if details and details.get('title') != 'Unknown':
                            try:
                                knowledge_graph.add_paper_with_authors(
                                    title=details['title'],
                                    summary=details['summary'],
                                    source_pdf=pdf_path,
                                    author_names=details['authors'],
                                    publication_date=details.get('publication_date')
                                )
                                total_new_pdfs_kg += 1
                                log_info("Successfully added to Knowledge Graph", filename=pdf_file)
                            except Exception as e:
                                log_error("Failed to add to Knowledge Graph", error=e, filename=pdf_file)
                        else:
                            log_warning("Could not extract valid paper details", filename=pdf_file)
                    else:
                        log_error("Could not load text from PDF", filename=pdf_file)
                else:
                    log_info("KG entry found - skipping KG update", filename=pdf_file)

                existing_docs = collection.get(where={"source": pdf_path})
                if not existing_docs['ids']:
                    log_info("Vector DB entries not found - processing for Vector DB", filename=pdf_file)
                    
                    if documents is None:
                        def load_pdf():
                            loader = PyPDFLoader(pdf_path)
                            return loader.load()
                        
                        documents = safe_execute(
                            load_pdf,
                            "PDF loading for Vector DB",
                            default_return=None,
                            filename=pdf_file
                        )
                    
                    if documents:
                        try:
                            text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=config.models.chunk_size, 
                                chunk_overlap=config.models.chunk_overlap
                            )
                            texts = text_splitter.split_documents(documents)

                            if texts:
                                ids = [str(hash(f'{doc.metadata["source"]}_{doc.metadata["page"]}_{doc.page_content}')) for doc in texts]
                                collection.add(ids=ids, documents=[doc.page_content for doc in texts], metadatas=[doc.metadata for doc in texts])
                                total_new_pdfs_vdb += 1
                                log_info("Vector DB entries added successfully", filename=pdf_file, chunks=len(texts))
                            else:
                                log_warning("Could not extract text chunks", filename=pdf_file)
                        except Exception as e:
                            log_error("Failed to process for Vector DB", error=e, filename=pdf_file)
                    else:
                        log_error("Could not load text for Vector DB", filename=pdf_file)
                else:
                    log_info("Vector DB entries found - skipping vector update", filename=pdf_file)

            except Exception as e:
                log_error(f"Error processing file", error=e, filename=pdf_file)

    kg_conn.close()
    print("\n--- Database Update Summary ---")
    print(f"Total new PDFs added to Knowledge Graph: {total_new_pdfs_kg}")
    print(f"Total new PDFs added to Vector Database: {total_new_pdfs_vdb}")
    print("Databases are up to date.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create or update the vector DB and knowledge graph from PDFs in the 'documentos' directory.")
    parser.add_argument("--force", action="store_true", help="Force deletion of existing databases before updating.")
    args = parser.parse_args()

    config = get_config()
    update_databases(config.arxiv.documents_root, config.database.vector_db_path, args.force)
