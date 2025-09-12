#!/usr/bin/env python3
"""
Launcher script for the arXiv Papers Analysis Web Interface.
Checks dependencies and launches the Streamlit application.

"""

import sys
import subprocess
import importlib
import os
from pathlib import Path

# Add the 'src' directory to the Python path to resolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))


def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'streamlit',
        'pandas',
        'plotly',
        'networkx',
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"[OK] {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"[MISSING] {package}")
    
    return missing_packages


def install_missing_packages(packages):
    """Install missing packages using pip."""
    if not packages:
        return True
    
    print(f"\n[INSTALL] Installing missing packages: {', '.join(packages)}")
    
    try:
        # Install from requirements file if it exists
        requirements_file = Path(__file__).parent / "requirements_web.txt"
        if requirements_file.exists():
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "-r", str(requirements_file)
            ])
        else:
            # Install individual packages
            for package in packages:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package
                ])
        
        print("[SUCCESS] All packages installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to install packages: {e}")
        return False


def check_system_requirements():
    """Check system requirements and setup."""
    checks = []
    project_root = Path(__file__).parent
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 8):
        checks.append(("Python Version", True, f"Python {python_version.major}.{python_version.minor}"))
    else:
        checks.append(("Python Version", False, f"Python {python_version.major}.{python_version.minor} (3.8+ required)"))
    
    # Dynamically import config to get DB path
    try:
        from config import get_config
        config = get_config()
        db_path = project_root / config.database.knowledge_db_dir / config.database.knowledge_db_file
        checks.append(("Knowledge Graph DB", db_path.exists(), str(db_path)))
    except (ImportError, AttributeError):
        db_path_fallback = project_root / "db" / "knowledge" / "knowledge_graph.sqlite"
        checks.append(("Knowledge Graph DB", False, f"Could not load from config, checked fallback: {db_path_fallback}"))

    # Check core modules in their new locations
    core_modules = {
        "src/config.py": "Configuration",
        "src/core/analysis/content_analysis.py": "Module: content_analysis",
        "src/core/rag/enhanced_rag_processor.py": "Module: enhanced_rag_processor",
        "src/logger.py": "Module: logger"
    }
    for path, name in core_modules.items():
        module_path = project_root / path
        checks.append((name, module_path.exists(), str(module_path)))
    
    return checks


def launch_streamlit():
    """Launch the Streamlit web interface."""
    web_interface_path = Path(__file__).parent / "src" / "web" / "web_interface.py"
    
    if not web_interface_path.exists():
        print(f"[ERROR] Web interface file not found: {web_interface_path}")
        return False
    
    print("[LAUNCH] Launching arXiv Papers Analysis Web Interface...")
    print("[INFO] Interface will be available at: http://localhost:8501")
    print("[INFO] Press Ctrl+C to stop the server")
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(web_interface_path),
            "--server.address", "0.0.0.0",
            "--server.port", "8501",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ])
        
    except KeyboardInterrupt:
        print("\n[STOP] Web interface stopped by user")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to launch web interface: {e}")
        return False
    except FileNotFoundError:
        print("[ERROR] Streamlit not found. Please install it with: pip install streamlit")
        return False


def main():
    """Main launcher function."""
    print("arXiv Papers Analysis System - Web Interface Launcher")
    print("=" * 60)
    
    # Check Python version first
    if sys.version_info < (3, 8):
        print("[ERROR] Python 3.8 or higher is required")
        return 1
    
    # Change to the script directory to ensure relative paths work
    os.chdir(Path(__file__).parent)
    
    # Check system requirements
    print("[CHECK] Checking system requirements...")
    system_checks = check_system_requirements()
    
    all_system_ok = True
    for name, status, details in system_checks:
        status_icon = "[OK]" if status else "[MISSING]"
        print(f"{status_icon} {name}: {details}")
        if not status:
            all_system_ok = False
    
    if not all_system_ok:
        print("\n[WARNING] Some system requirements are missing.")
        print("Please ensure you have:")
        print("- Processed some papers (run the main analysis first)")
        print("- All required configuration files in the 'src' directory")
        print("- Core analysis modules in their correct 'src/core' paths")
        
        response = input("\nContinue anyway? (y/N): ")
        if response.lower() != 'y':
            return 1
    
    # Check dependencies
    print("\n[CHECK] Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"\n[WARNING] Missing dependencies: {', '.join(missing)}")
        response = input("Install missing dependencies? (Y/n): ")
        
        if response.lower() != 'n':
            if not install_missing_packages(missing):
                print("[ERROR] Failed to install dependencies")
                return 1
            
            # Re-check dependencies
            print("\n[RECHECK] Re-checking dependencies...")
            missing = check_dependencies()
            
            if missing:
                print(f"[ERROR] Still missing: {', '.join(missing)}")
                return 1
        else:
            print("[WARNING] Continuing with missing dependencies - some features may not work")
    
    # All checks passed, launch the interface
    print("\n[SUCCESS] All systems ready!")
    
    # Launch Streamlit interface
    success = launch_streamlit()
    
    return 0 if success else 1


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n[STOP] Launcher interrupted by user")
        exit(0)
    except Exception as e:
        print(f"[ERROR] Launcher failed: {e}")
        exit(1)
