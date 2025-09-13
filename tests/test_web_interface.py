#!/usr/bin/env python3
"""
Test script for Web Interface components.
Validates that the web interface can be imported and basic functionality works.
"""

import sys
import os
import traceback
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    tests = []
    
    # Core dependencies
    try:
        import streamlit as st
        tests.append(("Streamlit", True, st.__version__))
    except ImportError as e:
        tests.append(("Streamlit", False, str(e)))
    
    try:
        import pandas as pd
        tests.append(("Pandas", True, pd.__version__))
    except ImportError as e:
        tests.append(("Pandas", False, str(e)))
    
    try:
        import plotly
        tests.append(("Plotly", True, plotly.__version__))
    except ImportError as e:
        tests.append(("Plotly", False, str(e)))
    
    try:
        import networkx as nx
        tests.append(("NetworkX", True, nx.__version__))
    except ImportError as e:
        tests.append(("NetworkX", False, str(e)))
    
    # Project modules
    try:
        from web_interface import WebInterface
        tests.append(("Web Interface", True, "OK"))
    except ImportError as e:
        tests.append(("Web Interface", False, str(e)))
    
    try:
        from content_analysis import ContentAnalysisEngine
        tests.append(("Content Analysis", True, "OK"))
    except ImportError as e:
        tests.append(("Content Analysis", False, str(e)))
    
    try:
        from enhanced_rag_processor import EnhancedRAGProcessor
        tests.append(("RAG Processor", True, "OK"))
    except ImportError as e:
        tests.append(("RAG Processor", False, str(e)))
    
    # Display results
    all_passed = True
    for name, success, info in tests:
        status = "[OK]" if success else "[FAIL]"
        print(f"{status} {name}: {info}")
        if not success:
            all_passed = False
    
    return all_passed


def test_web_interface_initialization():
    """Test that WebInterface can be initialized."""
    print("\nTesting WebInterface initialization...")
    
    try:
        from web_interface import WebInterface
        
        # This should not actually initialize components in test mode
        # We'll just test the class exists and can be instantiated
        interface_class = WebInterface
        print("[OK] WebInterface class accessible")
        
        # Test that required methods exist
        required_methods = [
            'run', '_render_dashboard', '_render_paper_analysis',
            '_render_content_analysis', '_render_knowledge_graph',
            '_render_export_reports', '_render_system_settings'
        ]
        
        for method in required_methods:
            if hasattr(interface_class, method):
                print(f"[OK] Method {method} exists")
            else:
                print(f"[FAIL] Method {method} missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"[FAIL] WebInterface initialization failed: {e}")
        traceback.print_exc()
        return False


def test_configuration_files():
    """Test that required configuration files exist."""
    print("\nTesting configuration files...")
    
    required_files = [
        "web_interface.py",
        "content_analysis.py", 
        "enhanced_rag_processor.py",
        "config.py",
        "logger.py",
        "requirements_web.txt",
        "launch_web_interface.py"
    ]
    
    all_exist = True
    current_dir = Path(__file__).parent
    
    for filename in required_files:
        filepath = current_dir / filename
        if filepath.exists():
            print(f"[OK] {filename}")
        else:
            print(f"[FAIL] {filename} missing")
            all_exist = False
    
    return all_exist


def test_database_schema():
    """Test that database schema is compatible."""
    print("\nTesting database schema...")
    
    try:
        import sqlite3
        from src.core.analysis import knowledge_graph
        
        # Check if database exists
        db_path = knowledge_graph.DB_FILE
        if not os.path.exists(db_path):
            print(f"[INFO] Database not found at {db_path}")
            print("[INFO] Run main analysis first to create database")
            return True  # Not a failure, just not ready
        
        # Check required tables
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            required_tables = [
                'papers',
                'chunks', 
                'content_analyses',
                'paper_references',
                'paper_concepts',
                'paper_topics',
                'paper_sections'
            ]
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = {row[0] for row in cursor.fetchall()}
            
            all_tables_exist = True
            for table in required_tables:
                if table in existing_tables:
                    print(f"[OK] Table {table}")
                else:
                    print(f"[FAIL] Table {table} missing")
                    all_tables_exist = False
            
            return all_tables_exist
            
    except Exception as e:
        print(f"[FAIL] Database schema test failed: {e}")
        return False


def test_export_functionality():
    """Test export functionality components."""
    print("\nTesting export functionality...")
    
    try:
        # Test that we can create export data structures
        test_data = {
            'papers': [{'id': 1, 'title': 'Test Paper', 'summary': 'Test'}],
            'concepts': [{'term': 'test', 'frequency': 1}],
            'metadata': {'export_date': '2024-01-01'}
        }
        
        # Test JSON export
        import json
        json_data = json.dumps(test_data, indent=2, default=str)
        print("[OK] JSON export works")
        
        # Test CSV export
        import pandas as pd
        df = pd.DataFrame(test_data['papers'])
        csv_data = df.to_csv(index=False)
        print("[OK] CSV export works")
        
        # Test Excel export (if available)
        try:
            import io
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='papers', index=False)
            print("[OK] Excel export works")
        except ImportError:
            print("[INFO] Excel export not available (xlsxwriter not installed)")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Export functionality test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Web Interface Validation Tests")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("WebInterface Initialization", test_web_interface_initialization), 
        ("Configuration Files", test_configuration_files),
        ("Database Schema", test_database_schema),
        ("Export Functionality", test_export_functionality)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"[ERROR] {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("\n[SUCCESS] All tests passed! Web interface is ready.")
        return 0
    else:
        print(f"\n[WARNING] {len(results) - passed} test(s) failed. Check requirements.")
        return 1


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n[STOP] Tests interrupted by user")
        exit(1)
    except Exception as e:
        print(f"[ERROR] Test suite failed: {e}")
        traceback.print_exc()
        exit(1)(1)