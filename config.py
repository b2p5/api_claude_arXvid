"""
Configuration management for arXiv papers analysis system.
Centralizes all configuration parameters and provides validation.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    
    # Vector database
    vector_db_path: str = "db/chroma"
    vector_collection_name: str = "arxiv_papers"
    
    # Knowledge graph database
    knowledge_db_dir: str = "db/knowledge"
    knowledge_db_file: str = "knowledge_graph.sqlite"
    
    @property
    def knowledge_db_path(self) -> str:
        return os.path.join(self.knowledge_db_dir, self.knowledge_db_file)


@dataclass
class ModelConfig:
    """LLM and embedding model configuration."""
    
    # Embedding model
    embedding_model_name: str = "all-MiniLM-L6-v2"
    
    # LLM configuration
    llm_model: str = "deepseek-chat"
    llm_provider: str = "deepseek"
    
    # Text processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Entity extraction
    extraction_text_limit: int = 4000


@dataclass
class ArxivConfig:
    """arXiv search and download configuration."""
    
    # Search parameters
    max_results: int = 10
    sort_criterion: str = "Relevance"  # Relevance, LastUpdatedDate, SubmittedDate
    
    # Download parameters
    min_delay_seconds: int = 5
    max_delay_seconds: int = 30
    download_timeout: int = 300
    
    # File organization
    documents_root: str = "documentos"
    
    def get_concept_path(self, concept: str) -> str:
        return os.path.join(self.documents_root, concept)
    
    def get_user_concept_path(self, username: str, concept: str) -> str:
        """Get path for a user's concept directory."""
        return os.path.join(self.documents_root, username, concept)
    
    def get_user_root_path(self, username: str) -> str:
        """Get root path for a user's documents."""
        return os.path.join(self.documents_root, username)


@dataclass
class RAGConfig:
    """RAG system configuration."""
    
    # Retrieval parameters
    top_k_results: int = 5
    similarity_threshold: float = 0.7
    
    # Generation parameters
    max_context_length: int = 4000
    temperature: float = 0.1


@dataclass
class VisualizationConfig:
    """Visualization configuration."""
    
    # Output settings
    output_dir: str = "grafos"
    dpi: int = 300
    
    # Graph layout
    figure_size: tuple = (20, 20)
    spring_layout_k: float = 0.2
    spring_layout_iterations: int = 50
    spring_layout_seed: int = 42
    
    # Node styling
    paper_node_color: str = "#d62728"
    author_node_color: str = "#1f77b4"
    paper_node_size: int = 1000
    author_node_size: int = 200
    edge_alpha: float = 0.4
    node_alpha: float = 0.9
    font_size: int = 8


@dataclass
class FastAPIConfig:
    """FastAPI specific configuration."""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = False
    
    # API settings
    title: str = "RAG System API"
    description: str = "Multi-user RAG system for academic papers with Gemini and ArXiv integration"
    version: str = "1.0.0"
    
    # Upload limits
    max_upload_size: int = 50 * 1024 * 1024  # 50MB
    allowed_file_types: List[str] = field(default_factory=lambda: [".pdf"])
    
    # Rate limiting
    rate_limit_per_minute: int = 60
    
    # CORS settings
    allow_origins: List[str] = field(default_factory=lambda: ["*"])
    allow_credentials: bool = True
    allow_methods: List[str] = field(default_factory=lambda: ["*"])
    allow_headers: List[str] = field(default_factory=lambda: ["*"])
    
    # Background task settings
    max_background_workers: int = 4


@dataclass
class AppConfig:
    """Main application configuration."""
    
    # Environment
    env_file: str = ".env"
    
    # Required API keys
    required_api_keys: List[str] = field(default_factory=lambda: [
        "DEEPSEEK_API_KEY"
    ])
    
    # Optional API keys
    optional_api_keys: List[str] = field(default_factory=lambda: [
        "GOOGLE_API_KEY"
    ])
    
    # Sub-configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    arxiv: ArxivConfig = field(default_factory=ArxivConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    fastapi: FastAPIConfig = field(default_factory=FastAPIConfig)
    
    def validate(self) -> List[str]:
        """
        Validate configuration and return list of errors.
        
        Returns:
            List of validation error messages. Empty if all valid.
        """
        errors = []
        
        # Check API keys
        missing_keys = []
        for key in self.required_api_keys:
            if not os.getenv(key):
                missing_keys.append(key)
        
        if missing_keys:
            errors.append(f"Missing required API keys: {', '.join(missing_keys)}")
        
        # Validate paths exist or can be created
        paths_to_check = [
            self.database.vector_db_path,
            self.database.knowledge_db_dir,
            self.arxiv.documents_root,
            self.visualization.output_dir
        ]
        
        for path in paths_to_check:
            try:
                os.makedirs(path, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create directory '{path}': {e}")
        
        # Validate model parameters
        if self.models.chunk_size <= 0:
            errors.append("chunk_size must be positive")
        
        if self.models.chunk_overlap >= self.models.chunk_size:
            errors.append("chunk_overlap must be less than chunk_size")
        
        if self.arxiv.max_results <= 0:
            errors.append("max_results must be positive")
        
        if self.rag.top_k_results <= 0:
            errors.append("top_k_results must be positive")
        
        return errors
    
    def validate_and_raise(self) -> None:
        """Validate configuration and raise exception if invalid."""
        errors = self.validate()
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors))


# Global configuration instance
config = AppConfig()


def load_config(env_file: Optional[str] = None) -> AppConfig:
    """
    Load configuration from environment file.
    
    Args:
        env_file: Path to environment file. If None, uses config.env_file
    
    Returns:
        Configured AppConfig instance
    """
    if env_file:
        config.env_file = env_file
    
    # Load environment variables if file exists
    if os.path.exists(config.env_file):
        import dotenv
        dotenv.load_dotenv(config.env_file)
    
    return config


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    return config


# Auto-load configuration when module is imported
load_config()