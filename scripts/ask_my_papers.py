import argparse
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import chromadb
import dotenv
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_deepseek import ChatDeepSeek
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.config import get_config

def get_retriever(db_path=None, model_name=None):
    """Initializes a retriever for the ChromaDB vector store."""
    config = get_config()
    
    # Use config defaults if not provided
    if db_path is None:
        db_path = config.database.vector_db_path
    if model_name is None:
        model_name = config.models.embedding_model_name
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database path not found: {db_path}. Please create it first using 'python src/administration/indexing/rag_bbdd_vector_optimized.py'")

    embedding_function = HuggingFaceEmbeddings(model_name=model_name)
    db = chromadb.PersistentClient(path=db_path)
    collection = db.get_collection(name=config.database.vector_collection_name)
    
    # This is a simplified retriever. LangChain offers more advanced retrievers.
    # We manually retrieve and format the documents.
    def retrieve_docs(query):
        results = collection.query(
            query_texts=[query],
            n_results=config.rag.top_k_results
        )
        return "\n---\n".join(results['documents'][0])
    
    return retrieve_docs

def main():
    """
    Main function to ask a question to the RAG system.
    Loads DEEPSEEK_API_KEY from .env file.
    """
    dotenv.load_dotenv()

    # --- API Key Check ---
    if "DEEPSEEK_API_KEY" not in os.environ:
        print("Error: DEEPSEEK_API_KEY environment variable not set.")
        print("Please get your API key from DeepSeek (https://platform.deepseek.com/api_keys)")
        print("Then, add it to the .env file in the project root: DEEPSEEK_API_KEY=YOUR_API_KEY_HERE")
        return

    parser = argparse.ArgumentParser(description="Ask a question to your papers using a RAG model.")
    parser.add_argument("query", type=str, help="The question you want to ask.")
    args = parser.parse_args()

    try:
        print("Initializing components...")
        config = get_config()
        retriever = get_retriever()
        llm = ChatDeepSeek(model=config.models.llm_model)

        template = """
        You are a helpful assistant. Answer the question based only on the following context provided from research papers:
        
        CONTEXT:
        {context}
        
        QUESTION:
        {question}
        
        ANSWER:
        """
        prompt = PromptTemplate.from_template(template)

        # --- RAG Chain using LangChain Expression Language (LCEL) ---
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        print(f"\nAsking: {args.query}\n")
        print("--- Answer ---")
        response = rag_chain.invoke(args.query)
        print(response)

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()