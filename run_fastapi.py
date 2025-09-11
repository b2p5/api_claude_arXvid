#!/usr/bin/env python3
"""
FastAPI server launcher for RAG System API.
Provides convenient startup with configuration options.
"""

import argparse
import sys
import os

# Add the 'src' directory to the Python path to resolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import uvicorn
from config import get_config, load_config
from logger import get_logger, log_info, log_error

def main():
    """Main function to launch FastAPI server."""
    parser = argparse.ArgumentParser(
        description="Launch FastAPI RAG System API server"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to bind to (overrides config)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind to (overrides config)"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Log level"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        if args.config:
            load_config(args.config)
        
        config = get_config()
        logger = get_logger()
        
        # Validate configuration
        config.validate_and_raise()
        
        # Override config with command line arguments
        fastapi_config = config.fastapi
        
        host = args.host or fastapi_config.host
        port = args.port or fastapi_config.port
        reload = args.reload or fastapi_config.reload
        debug = args.debug or fastapi_config.debug
        
        # Set environment variables for uvicorn
        if debug:
            os.environ["FASTAPI_DEBUG"] = "true"
        
        log_info("Starting FastAPI server", 
                host=host, 
                port=port, 
                reload=reload, 
                debug=debug,
                workers=args.workers)
        
        # Server configuration
        server_config = {
            "app": "api.main:app",
            "host": host,
            "port": port,
            "reload": reload,
            "log_level": args.log_level,
            "access_log": True
        }
        
        # Add workers only if not in reload mode (they're incompatible)
        if args.workers and not reload:
            server_config["workers"] = args.workers
        
        # Launch server
        uvicorn.run(**server_config)
        
    except KeyboardInterrupt:
        log_info("Server shutdown requested by user")
        sys.exit(0)
        
    except Exception as e:
        log_error("Failed to start server", error=str(e))
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()