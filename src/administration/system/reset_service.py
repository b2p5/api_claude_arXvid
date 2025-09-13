#!/usr/bin/env python3
"""
System Reset Service for arXiv Papers RAG System.
Provides comprehensive reset functionality with safety confirmations.
"""

import os
import shutil
import sqlite3
import argparse
import sys
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
import json

import chromadb
from src.config import get_config
from src.logger import get_logger, log_info, log_warning, log_error


class SystemResetService:
    """Service for resetting various components of the RAG system."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger()
        
    def create_backup(self, backup_name: Optional[str] = None) -> str:
        """
        Create a backup of the current system state.
        
        Args:
            backup_name: Custom backup name, defaults to timestamp
            
        Returns:
            Path to the backup directory
        """
        if backup_name is None:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        backup_dir = Path(f"backups/{backup_name}")
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        log_info("Creating system backup", backup_dir=str(backup_dir))
        
        try:
            # Backup documents
            docs_path = Path(self.config.arxiv.documents_root)
            if docs_path.exists():
                shutil.copytree(docs_path, backup_dir / "documentos", dirs_exist_ok=True)
                log_info("Backed up documents", path=str(docs_path))
            
            # Backup databases
            db_backup_dir = backup_dir / "databases"
            db_backup_dir.mkdir(exist_ok=True)
            
            # Vector DB
            vector_db_path = Path(self.config.database.vector_db_path)
            if vector_db_path.exists():
                shutil.copytree(vector_db_path, db_backup_dir / "chroma", dirs_exist_ok=True)
                log_info("Backed up vector database", path=str(vector_db_path))
            
            # Knowledge DB
            knowledge_db_path = Path(self.config.database.knowledge_db_path)
            if knowledge_db_path.exists():
                shutil.copy2(knowledge_db_path, db_backup_dir / "knowledge_graph.sqlite")
                log_info("Backed up knowledge database", path=str(knowledge_db_path))
            
            # Users DB
            users_db_path = Path("db/users.db")
            if users_db_path.exists():
                shutil.copy2(users_db_path, db_backup_dir / "users.db")
                log_info("Backed up users database")
            
            # Embeddings cache
            cache_path = Path("cache")
            if cache_path.exists():
                shutil.copytree(cache_path, backup_dir / "cache", dirs_exist_ok=True)
                log_info("Backed up embeddings cache")
                
            # Create backup info file
            backup_info = {
                "created_at": datetime.now().isoformat(),
                "backup_name": backup_name,
                "config": {
                    "documents_root": self.config.arxiv.documents_root,
                    "vector_db_path": self.config.database.vector_db_path,
                    "knowledge_db_path": self.config.database.knowledge_db_path
                }
            }
            
            with open(backup_dir / "backup_info.json", "w") as f:
                json.dump(backup_info, f, indent=2)
                
            log_info("Backup created successfully", backup_dir=str(backup_dir))
            return str(backup_dir)
            
        except Exception as e:
            log_error("Failed to create backup", error=str(e))
            raise
    
    def reset_documents(self, username: Optional[str] = None) -> Dict[str, Any]:
        """
        Reset documents (all or for specific user).
        
        Args:
            username: If provided, reset only this user's documents
            
        Returns:
            Reset statistics
        """
        docs_path = Path(self.config.arxiv.documents_root)
        stats = {"deleted_files": 0, "deleted_dirs": 0, "errors": []}
        
        try:
            if username:
                user_path = docs_path / username
                if user_path.exists():
                    stats["deleted_files"] = sum(1 for f in user_path.rglob("*") if f.is_file())
                    shutil.rmtree(user_path)
                    stats["deleted_dirs"] = 1
                    log_info("Reset user documents", username=username, stats=stats)
                else:
                    log_warning("User documents not found", username=username)
            else:
                if docs_path.exists():
                    stats["deleted_files"] = sum(1 for f in docs_path.rglob("*") if f.is_file())
                    stats["deleted_dirs"] = sum(1 for d in docs_path.rglob("*") if d.is_dir())
                    shutil.rmtree(docs_path)
                    log_info("Reset all documents", stats=stats)
                
                # Recreate root documents directory
                docs_path.mkdir(parents=True, exist_ok=True)
                log_info("Recreated documents root directory", path=str(docs_path))
                
        except Exception as e:
            error_msg = f"Error resetting documents: {str(e)}"
            stats["errors"].append(error_msg)
            log_error("Documents reset failed", error=str(e))
            
        return stats
    
    def reset_vector_database(self) -> Dict[str, Any]:
        """
        Reset ChromaDB vector database.
        
        Returns:
            Reset statistics
        """
        stats = {"collections_deleted": 0, "errors": []}
        
        try:
            # Delete ChromaDB directory
            vector_db_path = Path(self.config.database.vector_db_path)
            if vector_db_path.exists():
                shutil.rmtree(vector_db_path)
                stats["collections_deleted"] = 1
                log_info("Reset vector database", path=str(vector_db_path))
            
            # Recreate directory structure
            vector_db_path.parent.mkdir(parents=True, exist_ok=True)
            log_info("Recreated vector database directory")
            
        except Exception as e:
            error_msg = f"Error resetting vector database: {str(e)}"
            stats["errors"].append(error_msg)
            log_error("Vector database reset failed", error=str(e))
            
        return stats
    
    def reset_knowledge_database(self) -> Dict[str, Any]:
        """
        Reset knowledge graph database.
        
        Returns:
            Reset statistics
        """
        stats = {"tables_cleared": 0, "errors": []}
        
        try:
            knowledge_db_path = Path(self.config.database.knowledge_db_path)
            
            if knowledge_db_path.exists():
                # Connect and clear all tables
                conn = sqlite3.connect(knowledge_db_path)
                cursor = conn.cursor()
                
                # Get all table names
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                
                # Clear all tables
                for table in tables:
                    table_name = table[0]
                    cursor.execute(f"DELETE FROM {table_name};")
                    stats["tables_cleared"] += 1
                    log_info("Cleared knowledge table", table=table_name)
                
                conn.commit()
                conn.close()
                
                log_info("Reset knowledge database", stats=stats)
            else:
                log_warning("Knowledge database not found", path=str(knowledge_db_path))
                
        except Exception as e:
            error_msg = f"Error resetting knowledge database: {str(e)}"
            stats["errors"].append(error_msg)
            log_error("Knowledge database reset failed", error=str(e))
            
        return stats
    
    def reset_users_database(self) -> Dict[str, Any]:
        """
        Reset users database.
        
        Returns:
            Reset statistics
        """
        stats = {"users_deleted": 0, "errors": []}
        
        try:
            users_db_path = Path("db/users.db")
            
            if users_db_path.exists():
                # Count users before deletion
                conn = sqlite3.connect(users_db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM users;")
                user_count = cursor.fetchone()[0]
                conn.close()
                
                # Delete database file
                users_db_path.unlink()
                stats["users_deleted"] = user_count
                log_info("Reset users database", users_deleted=user_count)
            else:
                log_warning("Users database not found")
                
        except Exception as e:
            error_msg = f"Error resetting users database: {str(e)}"
            stats["errors"].append(error_msg)
            log_error("Users database reset failed", error=str(e))
            
        return stats
    
    def reset_embeddings_cache(self) -> Dict[str, Any]:
        """
        Reset embeddings cache.
        
        Returns:
            Reset statistics
        """
        stats = {"cache_files_deleted": 0, "errors": []}
        
        try:
            cache_path = Path("cache")
            
            if cache_path.exists():
                cache_files = list(cache_path.rglob("*"))
                stats["cache_files_deleted"] = len([f for f in cache_files if f.is_file()])
                shutil.rmtree(cache_path)
                log_info("Reset embeddings cache", stats=stats)
                
                # Recreate cache directory
                cache_path.mkdir(parents=True, exist_ok=True)
                log_info("Recreated cache directory")
            else:
                log_warning("Cache directory not found")
                
        except Exception as e:
            error_msg = f"Error resetting embeddings cache: {str(e)}"
            stats["errors"].append(error_msg)
            log_error("Embeddings cache reset failed", error=str(e))
            
        return stats
    
    def full_system_reset(self, create_backup_first: bool = True) -> Dict[str, Any]:
        """
        Perform a complete system reset.
        
        Args:
            create_backup_first: Whether to create backup before reset
            
        Returns:
            Combined reset statistics
        """
        stats = {
            "backup_path": None,
            "documents": {},
            "vector_db": {},
            "knowledge_db": {},
            "users_db": {},
            "embeddings_cache": {},
            "start_time": datetime.now().isoformat()
        }
        
        try:
            # Create backup first
            if create_backup_first:
                stats["backup_path"] = self.create_backup()
            
            # Reset all components
            log_info("Starting full system reset")
            
            stats["documents"] = self.reset_documents()
            stats["vector_db"] = self.reset_vector_database()
            stats["knowledge_db"] = self.reset_knowledge_database()
            stats["users_db"] = self.reset_users_database()
            stats["embeddings_cache"] = self.reset_embeddings_cache()
            
            stats["end_time"] = datetime.now().isoformat()
            log_info("Full system reset completed", stats=stats)
            
        except Exception as e:
            log_error("Full system reset failed", error=str(e))
            stats["error"] = str(e)
            
        return stats


def confirm_reset(reset_type: str, details: str = "") -> bool:
    """
    Get user confirmation for reset operation.
    
    Args:
        reset_type: Type of reset being performed
        details: Additional details about the reset
        
    Returns:
        True if user confirms, False otherwise
    """
    print(f"\n*** WARNING: You are about to perform a {reset_type}")
    if details:
        print(f"   {details}")
    print("\n   This action is IRREVERSIBLE and will:")
    
    if "full" in reset_type.lower():
        print("   [X] Delete ALL user documents")
        print("   [X] Clear ALL vector embeddings")
        print("   [X] Clear ALL knowledge graph data")
        print("   [X] Delete ALL user accounts")
        print("   [X] Clear ALL embeddings cache")
    elif "documents" in reset_type.lower():
        print("   [X] Delete user documents")
    elif "databases" in reset_type.lower():
        print("   [X] Clear vector embeddings")
        print("   [X] Clear knowledge graph data")
        print("   [X] Clear embeddings cache")
    
    print(f"\n   Type '{reset_type.upper()}' to confirm:")
    
    confirmation = input("   > ").strip()
    
    if confirmation != reset_type.upper():
        print("   [X] Reset cancelled - confirmation text did not match.")
        return False
    
    print(f"\n   Are you absolutely sure? Type 'YES I AM SURE' to proceed:")
    final_confirmation = input("   > ").strip()
    
    if final_confirmation != "YES I AM SURE":
        print("   [X] Reset cancelled - final confirmation failed.")
        return False
    
    print("   [OK] Reset confirmed. Proceeding...")
    return True


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Reset various components of the arXiv Papers RAG System"
    )
    
    parser.add_argument(
        "--type",
        choices=["full", "documents", "databases", "user"],
        default="full",
        help="Type of reset to perform (default: full)"
    )
    
    parser.add_argument(
        "--username",
        type=str,
        help="Username for user-specific reset (only with --type user)"
    )
    
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating backup before reset"
    )
    
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompts (DANGEROUS)"
    )
    
    args = parser.parse_args()
    
    # Initialize service
    service = SystemResetService()
    
    # Determine reset type and get confirmation
    if args.type == "full":
        if not args.yes and not confirm_reset("full system reset"):
            sys.exit(0)
        stats = service.full_system_reset(create_backup_first=not args.no_backup)
        
    elif args.type == "documents":
        username = args.username if args.type == "user" else None
        details = f"for user: {username}" if username else "for ALL users"
        if not args.yes and not confirm_reset("documents reset", details):
            sys.exit(0)
        stats = service.reset_documents(username)
        
    elif args.type == "databases":
        if not args.yes and not confirm_reset("databases reset"):
            sys.exit(0)
        stats = {
            "vector_db": service.reset_vector_database(),
            "knowledge_db": service.reset_knowledge_database(),
            "embeddings_cache": service.reset_embeddings_cache()
        }
        
    elif args.type == "user":
        if not args.username:
            print("❌ Error: --username is required for user reset")
            sys.exit(1)
        if not args.yes and not confirm_reset("user reset", f"for user: {args.username}"):
            sys.exit(0)
        stats = service.reset_documents(args.username)
    
    # Print results
    print("\n" + "="*60)
    print("RESET COMPLETED")
    print("="*60)
    print(json.dumps(stats, indent=2, default=str))
    print("="*60)
    
    # Check for errors
    total_errors = 0
    if isinstance(stats, dict):
        for key, value in stats.items():
            if isinstance(value, dict) and "errors" in value:
                total_errors += len(value["errors"])
    
    if total_errors > 0:
        print(f"WARNING: Reset completed with {total_errors} errors. Check logs for details.")
        sys.exit(1)
    else:
        print("SUCCESS: Reset completed successfully!")


if __name__ == "__main__":
    main()
