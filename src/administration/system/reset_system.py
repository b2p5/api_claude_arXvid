#!/usr/bin/env python3
"""
Simple script to reset the RAG system.
Provides easy-to-use commands for common reset operations.
"""

import sys
from reset_service import SystemResetService, confirm_reset


def main():
    """Main interactive function."""
    service = SystemResetService()
    
    print("=" * 60)
    print("RAG SYSTEM RESET UTILITY")
    print("=" * 60)
    
    print("\nAvailable reset options:")
    print("1. Full system reset (documents + databases + users)")
    print("2. Reset only documents (all users)")  
    print("3. Reset only databases (vectors + knowledge + cache)")
    print("4. Reset specific user")
    print("5. Create backup only")
    print("6. Exit")
    
    while True:
        try:
            choice = input("\nSelect option (1-6): ").strip()
            
            if choice == "1":
                # Full reset
                if confirm_reset("full system reset"):
                    print("\nCreating backup and resetting system...")
                    stats = service.full_system_reset()
                    print_results(stats)
                break
                
            elif choice == "2":
                # Documents only
                if confirm_reset("documents reset", "for ALL users"):
                    print("\nResetting all documents...")
                    stats = service.reset_documents()
                    print_results({"documents": stats})
                break
                
            elif choice == "3":
                # Databases only
                if confirm_reset("databases reset"):
                    print("\nResetting databases...")
                    stats = {
                        "vector_db": service.reset_vector_database(),
                        "knowledge_db": service.reset_knowledge_database(),
                        "embeddings_cache": service.reset_embeddings_cache()
                    }
                    print_results(stats)
                break
                
            elif choice == "4":
                # User specific
                username = input("Enter username to reset: ").strip()
                if not username:
                    print("[X] Username cannot be empty")
                    continue
                    
                if confirm_reset("user reset", f"for user: {username}"):
                    print(f"\nResetting user: {username}...")
                    stats = service.reset_documents(username)
                    print_results({"user_documents": stats})
                break
                
            elif choice == "5":
                # Backup only
                print("\nCreating backup...")
                backup_path = service.create_backup()
                print(f"[OK] Backup created at: {backup_path}")
                break
                
            elif choice == "6":
                print("Goodbye!")
                sys.exit(0)
                
            else:
                print("[X] Invalid option. Please select 1-6.")
                
        except KeyboardInterrupt:
            print("\n\nReset cancelled by user.")
            sys.exit(0)
        except Exception as e:
            print(f"\n[X] Error: {e}")
            sys.exit(1)


def print_results(stats):
    """Print reset results in a readable format."""
    print("\n" + "=" * 50)
    print("RESET RESULTS")
    print("=" * 50)
    
    total_errors = 0
    
    for component, data in stats.items():
        if component == "backup_path" and data:
            print(f"Backup: {data}")
            continue
            
        if not isinstance(data, dict):
            continue
            
        print(f"\n{component.replace('_', ' ').title()}:")
        
        for key, value in data.items():
            if key == "errors":
                if value:
                    total_errors += len(value)
                    for error in value:
                        print(f"  [X] {error}")
            else:
                print(f"  [OK] {key.replace('_', ' ').title()}: {value}")
    
    print("=" * 50)
    
    if total_errors > 0:
        print(f"WARNING: Completed with {total_errors} errors.")
    else:
        print("SUCCESS: Reset completed successfully!")
        
    print("=" * 50)


if __name__ == "__main__":
    main()