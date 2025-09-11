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


def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'streamlit',
        'pandas', 
        'plotly',
        'networkx',
        'sqlite3',  # Built-in with Python
    ]
    
    missing_packages = []
    
    for package in required_packages:
        if package == 'sqlite3':
            continue  # Built-in module
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
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 8):
        checks.append(("Python Version", True, f"Python {python_version.major}.{python_version.minor}"))
    else:
        checks.append(("Python Version", False, f"Python {python_version.major}.{python_version.minor} (3.8+ required)"))
    
    # Check if knowledge graph database exists
    db_file = Path(__file__).parent / "knowledge_graph.db"
    checks.append(("Knowledge Graph DB", db_file.exists(), str(db_file)))
    
    # Check config file
    config_file = Path(__file__).parent / "config.py"
    checks.append(("Configuration", config_file.exists(), str(config_file)))
    
    # Check core analysis modules
    core_modules = ["content_analysis.py", "enhanced_rag_processor.py", "logger.py"]
    for module in core_modules:
        module_path = Path(__file__).parent / module
        checks.append((f"Module: {module}", module_path.exists(), str(module_path)))
    
    return checks


def launch_streamlit():
    """Launch the Streamlit web interface."""
    web_interface_path = Path(__file__).parent / "web_interface.py"
    
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
        print("- All required configuration files")
        print("- Core analysis modules")
        
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
    
    # Change to the script directory
    os.chdir(Path(__file__).parent)
    
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