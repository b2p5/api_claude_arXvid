#!/usr/bin/env python3
"""
Example script showing how to use the new configuration system.
This demonstrates how to access and modify configuration settings.
"""

from config import get_config, load_config
import os

def show_current_config():
    """Display current configuration settings."""
    print("Current Configuration Settings")
    print("=" * 40)
    
    config = get_config()
    
    print(f"Database Configuration:")
    print(f"  Vector DB Path: {config.database.vector_db_path}")
    print(f"  Knowledge DB Path: {config.database.knowledge_db_path}")
    print(f"  Collection Name: {config.database.vector_collection_name}")
    
    print(f"\nModel Configuration:")
    print(f"  Embedding Model: {config.models.embedding_model_name}")
    print(f"  LLM Model: {config.models.llm_model}")
    print(f"  Chunk Size: {config.models.chunk_size}")
    print(f"  Chunk Overlap: {config.models.chunk_overlap}")
    
    print(f"\narXiv Configuration:")
    print(f"  Max Results: {config.arxiv.max_results}")
    print(f"  Documents Root: {config.arxiv.documents_root}")
    print(f"  Download Delays: {config.arxiv.min_delay_seconds}-{config.arxiv.max_delay_seconds}s")
    
    print(f"\nRAG Configuration:")
    print(f"  Top-K Results: {config.rag.top_k_results}")
    print(f"  Temperature: {config.rag.temperature}")
    
    print(f"\nVisualization Configuration:")
    print(f"  Output Directory: {config.visualization.output_dir}")
    print(f"  Figure Size: {config.visualization.figure_size}")
    print(f"  DPI: {config.visualization.dpi}")

def show_api_keys_status():
    """Show API keys status without revealing actual keys."""
    print("\nAPI Keys Status")
    print("=" * 20)
    
    config = get_config()
    
    for key in config.required_api_keys:
        status = "[SET]" if os.getenv(key) else "[NOT SET]"
        print(f"  {key}: {status}")
    
    for key in config.optional_api_keys:
        status = "[SET]" if os.getenv(key) else "[NOT SET]"
        print(f"  {key} (optional): {status}")

def modify_config_example():
    """Example of how to modify configuration at runtime."""
    print("\nModifying Configuration Example")
    print("=" * 35)
    
    config = get_config()
    
    print(f"Original chunk size: {config.models.chunk_size}")
    
    # Modify configuration
    config.models.chunk_size = 1500
    config.models.chunk_overlap = 300
    config.arxiv.max_results = 15
    
    print(f"New chunk size: {config.models.chunk_size}")
    print(f"New chunk overlap: {config.models.chunk_overlap}")
    print(f"New max results: {config.arxiv.max_results}")
    
    print("\nNote: These changes only affect the current session.")
    print("To make permanent changes, modify config.py or use environment variables.")

def show_paths_example():
    """Example of how to use configuration paths."""
    print("\nUsing Configuration Paths")
    print("=" * 28)
    
    config = get_config()
    
    # Example: Get path for a specific concept
    concept = "machine_learning"
    concept_path = config.arxiv.get_concept_path(concept)
    print(f"Path for concept '{concept}': {concept_path}")
    
    # Example: Create directories if they don't exist
    directories = [
        config.database.vector_db_path,
        config.database.knowledge_db_dir,
        config.arxiv.documents_root,
        config.visualization.output_dir
    ]
    
    print(f"\nEnsuring directories exist:")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        exists = "EXISTS" if os.path.exists(directory) else "CREATED"
        print(f"  {directory}: {exists}")

def main():
    """Main function demonstrating configuration usage."""
    print("arXiv Papers Analysis System - Configuration Example")
    print("=" * 55)
    
    # Load configuration (happens automatically when importing config)
    print("Loading configuration...")
    config = load_config()
    
    # Validate configuration
    errors = config.validate()
    if errors:
        print(f"Configuration errors found:")
        for error in errors:
            print(f"  - {error}")
        return
    else:
        print("Configuration is valid!")
    
    # Show various configuration aspects
    show_current_config()
    show_api_keys_status()
    show_paths_example()
    modify_config_example()
    
    print(f"\nConfiguration system is ready to use!")
    print(f"Run 'python validate_config.py' to perform a full system check.")

if __name__ == "__main__":
    main()