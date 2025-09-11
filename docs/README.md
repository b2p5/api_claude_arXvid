# Documentation Index

Complete documentation for the arXiv Papers Analysis System.

## 📚 Main Documentation

- **[Project Overview](../README.md)** - Main project documentation, features, and quick start guide

## 🔧 Setup & Configuration

- **[Configuration Guide](README_CONFIG.md)** - Detailed configuration options and environment setup
- **[Error Handling](README_ERROR_HANDLING.md)** - Error handling strategies and troubleshooting

## 💻 User Interfaces

- **[Web Interface](README_WEB_INTERFACE.md)** - Streamlit web interface usage and features

## 🔍 Core Features

- **[Advanced Search](ADVANCED_SEARCH_README.md)** - Advanced search capabilities and hybrid search
- **[Content Analysis](CONTENT_ANALYSIS_README.md)** - Deep content analysis using LLMs

## 📁 Project Organization

### Core Modules Structure
```
claude_arXiv/
├── config.py              # Centralized configuration system
├── logger.py               # Structured logging system
├── retry_utils.py          # Robust retry mechanisms
├── llm_utils.py           # LLM utilities and entity extraction
├── pdf_validator.py        # PDF validation and processing
├── embedding_cache.py      # Intelligent embedding caching
├── database_optimizer.py   # Database performance optimization
├── parallel_processing.py  # Parallel processing capabilities
└── intelligent_chunking.py # Smart text segmentation
```

### Search & Retrieval
```
├── advanced_search.py      # Hybrid search implementation
├── search_filters.py       # Advanced filtering capabilities
├── search_ranking.py       # Intelligent result ranking
└── rag_bbdd_vector*.py    # RAG system implementations
```

### Content Analysis
```
├── content_analysis.py     # Deep content analysis using LLMs
├── content_analysis_db.py  # Analysis result storage
├── enhanced_rag_processor.py # Advanced RAG processing
└── rag_with_content_analysis.py # RAG with analysis integration
```

### Web Interface & Tools
```
├── web_interface.py        # Main Streamlit application
├── launch_web_interface.py # Interface launcher
├── chat_with_advanced_search.py # Chat interface
├── get_arxiv.py           # Paper download utility
├── ask_my_papers.py       # Command-line QA interface
└── knowledge_graph.py     # Knowledge graph generation
```

### Testing & Examples
```
├── test_*.py              # Comprehensive test suites
├── example_*.py           # Usage examples
└── validate_config.py     # Configuration validation
```

## 🚀 Quick Navigation

### Getting Started
1. [Installation](../README.md#installation)
2. [Configuration](README_CONFIG.md)
3. [First Steps](../README.md#basic-usage)

### Core Workflows
1. **Paper Collection**: `get_arxiv.py` → `rag_bbdd_vector.py`
2. **Question Answering**: `ask_my_papers.py` or Web Interface
3. **Advanced Search**: `advanced_search.py` or Web Interface
4. **Content Analysis**: `content_analysis.py` or Web Interface

### Development
1. [Testing](../README.md#testing)
2. [Error Handling](README_ERROR_HANDLING.md)
3. [Performance Optimization](../README.md#performance)

## 📊 Features Matrix

| Feature | CLI | Web Interface | API |
|---------|-----|---------------|-----|
| Paper Search | ✅ | ✅ | ❌ |
| Question Answering | ✅ | ✅ | ❌ |
| Content Analysis | ✅ | ✅ | ❌ |
| Knowledge Graphs | ✅ | ✅ | ❌ |
| Batch Processing | ✅ | ❌ | ❌ |
| Performance Testing | ✅ | ❌ | ❌ |

## 🔗 External References

- [arXiv API Documentation](https://arxiv.org/help/api)
- [LangChain Documentation](https://langchain.readthedocs.io/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

For specific feature documentation, navigate to the respective markdown files listed above.