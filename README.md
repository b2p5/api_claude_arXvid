# arXiv Papers Analysis System

A comprehensive RAG (Retrieval-Augmented Generation) system for analyzing and interacting with arXiv research papers. The system provides advanced search capabilities, content analysis, knowledge graph generation, and a web interface for intuitive interaction.

## ✨ Features

### Core Functionality
- **Advanced Search**: Hybrid semantic and keyword search with intelligent ranking
- **Content Analysis**: Deep analysis of paper content using LLMs
- **Knowledge Graph**: Automatic generation of entity relationships and research networks
- **RAG System**: Question-answering over your paper collection
- **Web Interface**: User-friendly Streamlit-based interface

### Advanced Capabilities
- **Intelligent Chunking**: Smart text segmentation for better retrieval
- **Parallel Processing**: Efficient batch processing of multiple papers
- **Caching System**: Intelligent embedding and result caching
- **Error Handling**: Robust retry mechanisms and graceful degradation
- **Performance Optimization**: Database optimization and query acceleration

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Required API keys:
  - DeepSeek API key for LLM functionality

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd claude_arXiv
   ```

2. **Install dependencies**
   ```bash
   # For basic functionality
   pip install -r requirements/base.txt
   
   # For web interface
   pip install -r requirements/web.txt
   
   # For development
   pip install -r requirements/dev.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Validate configuration**
   ```bash
   python validate_config.py
   ```

### Basic Usage

1. **Download and process papers**
   ```bash
   python get_arxiv.py "machine learning" --max-results 10
   ```

2. **Build vector database**
   ```bash
   python rag_bbdd_vector.py
   ```

3. **Ask questions about your papers**
   ```bash
   python ask_my_papers.py "What are the main approaches to attention mechanisms?"
   ```

4. **Launch web interface**
   ```bash
   python launch_web_interface.py
   ```

## 📁 Project Structure

```
claude_arXiv/
├── requirements/           # Dependency management
│   ├── base.txt           # Core dependencies
│   ├── web.txt            # Web interface dependencies
│   └── dev.txt            # Development dependencies
├── docs/                  # Documentation
├── config.py              # Centralized configuration
├── logger.py              # Logging system
├── core modules/          # Core functionality
├── web_interface.py       # Streamlit web interface
├── tests/                 # Test files
└── scripts/               # Utility scripts
```

## 🔧 Configuration

The system uses a centralized configuration system in `config.py`. Key settings include:

- **Database**: Vector DB and knowledge graph storage paths
- **Models**: LLM and embedding model configuration
- **Processing**: Chunking, extraction, and retrieval parameters
- **Web Interface**: UI and visualization settings

See [Configuration Guide](README_CONFIG.md) for detailed configuration options.

## 📚 Core Modules

### Search & Retrieval
- `advanced_search.py` - Hybrid search implementation
- `search_filters.py` - Advanced filtering capabilities
- `search_ranking.py` - Intelligent result ranking

### Content Analysis
- `content_analysis.py` - Deep content analysis using LLMs
- `content_analysis_db.py` - Analysis result storage
- `enhanced_rag_processor.py` - Advanced RAG processing

### Knowledge Management
- `knowledge_graph.py` - Entity extraction and graph building
- `intelligent_chunking.py` - Smart text segmentation
- `embedding_cache.py` - Efficient embedding management

### Web Interface
- `web_interface.py` - Main Streamlit application
- `launch_web_interface.py` - Interface launcher with dependency checking

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test suites
python test_advanced_search.py
python test_content_analysis.py
python test_performance.py
```

## 📖 Documentation

- [Configuration Guide](README_CONFIG.md) - Detailed configuration options
- [Error Handling](README_ERROR_HANDLING.md) - Error handling strategies
- [Web Interface](README_WEB_INTERFACE.md) - Web interface usage
- [Advanced Search](ADVANCED_SEARCH_README.md) - Advanced search features
- [Content Analysis](CONTENT_ANALYSIS_README.md) - Content analysis capabilities

## 🚀 Performance

The system is optimized for performance with:
- Parallel processing capabilities
- Intelligent caching systems
- Database optimizations
- Memory-efficient operations

Performance benchmarks and optimization guides available in the test results.

## 🤝 Contributing

1. Follow the established code conventions
2. Use the centralized logging system
3. Add tests for new functionality
4. Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Related Projects

- [arXiv API](https://arxiv.org/help/api) - Paper download and metadata
- [LangChain](https://langchain.com/) - LLM integration framework
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Streamlit](https://streamlit.io/) - Web interface framework

---

For questions, issues, or contributions, please refer to the documentation or open an issue in the repository.