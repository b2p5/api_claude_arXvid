#!/usr/bin/env python3
"""
Configuration validation script for arXiv papers analysis system.
Run this before using the system to ensure all dependencies and settings are correct.
"""

import sys
import os
from pathlib import Path

def validate_system():
    """Validate the entire system configuration and dependencies."""
    
    print("Validating arXiv Papers Analysis System Configuration...")
    print("=" * 60)
    
    errors = []
    warnings = []
    
    # 1. Import and validate configuration
    try:
        from config import get_config
        config = get_config()
        print("[OK] Configuration module loaded successfully")
        
        # Validate configuration
        config_errors = config.validate()
        if config_errors:
            errors.extend([f"Config: {err}" for err in config_errors])
        else:
            print("[OK] Configuration validation passed")
            
    except ImportError as e:
        errors.append(f"Cannot import configuration: {e}")
        return False
    except Exception as e:
        errors.append(f"Configuration error: {e}")
        return False
    
    # 2. Check Python dependencies
    print("\nChecking Python dependencies...")
    required_packages = [
        'arxiv',
        'langchain',
        'chromadb',
        'sentence_transformers',
        'pypdf',
        'langchain_huggingface',
        'langchain_deepseek',
        'python_dotenv',
        'networkx',
        'matplotlib',
        'pyvis'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            # Handle packages with different import names
            import_name = package
            if package == 'python_dotenv':
                import_name = 'dotenv'
            elif package == 'sentence_transformers':
                import_name = 'sentence_transformers'
                
            __import__(import_name)
            print(f"  [OK] {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  [FAIL] {package}")
    
    if missing_packages:
        errors.append(f"Missing required packages: {', '.join(missing_packages)}")
        print(f"\nInstall missing packages with: pip install {' '.join(missing_packages)}")
    
    # 3. Check API keys
    print(f"\nChecking API keys...")
    for key in config.required_api_keys:
        if os.getenv(key):
            print(f"  [OK] {key} is set")
        else:
            print(f"  [FAIL] {key} is missing")
    
    for key in config.optional_api_keys:
        if os.getenv(key):
            print(f"  [OK] {key} is set (optional)")
        else:
            print(f"  [WARN] {key} is not set (optional)")
            warnings.append(f"Optional API key {key} not set")
    
    # 4. Check and create directories
    print(f"\nChecking directories...")
    directories = [
        config.database.vector_db_path,
        config.database.knowledge_db_dir,
        config.arxiv.documents_root,
        config.visualization.output_dir
    ]
    
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"  [OK] {directory}")
        except Exception as e:
            errors.append(f"Cannot create directory {directory}: {e}")
            print(f"  [FAIL] {directory}: {e}")
    
    # 5. Check .env file
    print(f"\nChecking environment file...")
    if os.path.exists(config.env_file):
        print(f"  [OK] {config.env_file} exists")
        
        # Check if .env file contains sensitive information
        with open(config.env_file, 'r') as f:
            env_content = f.read()
            if 'API_KEY' in env_content:
                print(f"  [OK] API keys found in {config.env_file}")
            else:
                warnings.append(f"No API keys found in {config.env_file}")
                print(f"  [WARN] No API keys found in {config.env_file}")
    else:
        warnings.append(f"Environment file {config.env_file} not found")
        print(f"  [WARN] {config.env_file} not found")
    
    # 6. Test model loading (quick test)
    print(f"\nTesting model initialization...")
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name=config.models.embedding_model_name)
        print(f"  [OK] Embedding model '{config.models.embedding_model_name}' loaded successfully")
    except Exception as e:
        warnings.append(f"Could not load embedding model: {e}")
        print(f"  [WARN] Embedding model load failed: {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    if not errors and not warnings:
        print("All checks passed! System is ready to use.")
        return True
    
    if warnings and not errors:
        print(f"System is functional but has {len(warnings)} warnings:")
        for warning in warnings:
            print(f"   - {warning}")
        print("\nYou can proceed, but consider addressing these warnings.")
        return True
    
    if errors:
        print(f"System has {len(errors)} critical errors:")
        for error in errors:
            print(f"   - {error}")
        
        if warnings:
            print(f"\nAdditionally, {len(warnings)} warnings:")
            for warning in warnings:
                print(f"   - {warning}")
        
        print(f"\nPlease fix these errors before using the system.")
        return False


if __name__ == "__main__":
    success = validate_system()
    sys.exit(0 if success else 1)