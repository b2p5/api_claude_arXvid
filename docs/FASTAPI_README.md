# FastAPI RAG System Documentation

## Overview

FastAPI RAG System API provides a multi-user RAG (Retrieval Augmented Generation) system for academic papers with Gemini and ArXiv integration. The API supports username-based document organization with individual user spaces.

## Features

- ✅ **Multi-user support** with username-based folder structure
- ✅ **PDF document upload** and processing
- ✅ **Real-time chat** with document collections
- ✅ **Advanced search** capabilities
- ✅ **Background processing** for document indexing
- ✅ **Streaming responses** for real-time interaction
- ✅ **User document management**
- ✅ **Knowledge graph integration**
- ✅ **Vector database storage**

## Installation

### 1. Install Dependencies

```bash
# Install FastAPI-specific requirements
pip install -r requirements/fastapi.txt

# Or install all requirements
pip install -r requirements/base.txt
pip install -r requirements/fastapi.txt
```

### 2. Configuration

Create or update your `.env` file:

```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
GOOGLE_API_KEY=your_google_api_key_here  # Optional
```

### 3. Initialize Databases

The databases will be automatically initialized on first run, but you can also initialize them manually:

```python
python -c "from knowledge_graph import create_database; create_database()"
```

## Quick Start

### 1. Start the Server

```bash
# Basic startup
python run_fastapi.py

# Development mode with auto-reload
python run_fastapi.py --reload --debug

# Custom host and port
python run_fastapi.py --host 127.0.0.1 --port 8080

# Production mode with multiple workers
python run_fastapi.py --workers 4 --log-level info
```

### 2. Access the API

- **API Server**: http://localhost:8000
- **Interactive Documentation**: http://localhost:8000/docs
- **Alternative Documentation**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## API Endpoints

### Core Endpoints

#### Health Check
```http
GET /health
```
Returns system health status and database connectivity.

#### Root Information
```http
GET /
```
Returns API information and available endpoints.

### User Management

#### List All Users
```http
GET /users
```
Returns list of all users with document statistics.

#### List User Documents
```http
GET /users/{username}/documents
```
Returns all documents for a specific user organized by category.

### Document Management

#### Upload PDF Document
```http
POST /users/{username}/upload-pdf
```
Upload a PDF document for a specific user and category.

**Parameters:**
- `username`: User identifier (will be normalized for filesystem)
- `category`: Document category/topic
- `file`: PDF file to upload

**Example:**
```bash
curl -X POST \
  "http://localhost:8000/users/juan_perez/upload-pdf" \
  -H "Content-Type: multipart/form-data" \
  -F "category=machine_learning" \
  -F "file=@paper.pdf"
```

#### Delete Document
```http
DELETE /users/{username}/documents/{category}/{filename}
```
Delete a specific document from user's collection.

### Search and Chat

#### Chat with Documents
```http
POST /users/{username}/chat
```
Chat with user's document collection using RAG.

**Request Body:**
```json
{
  "query": "What are the main contributions of transformer models?",
  "category": "transformers",  // Optional filter
  "stream": false,             // Optional streaming
  "max_tokens": 2000          // Optional token limit
}
```

**Example:**
```bash
curl -X POST \
  "http://localhost:8000/users/juan_perez/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain attention mechanisms in deep learning",
    "category": "machine_learning"
  }'
```

#### Search Documents
```http
POST /users/{username}/search
```
Advanced search in user's document collection.

**Request Body:**
```json
{
  "query": "attention mechanism",
  "category": "transformers",  // Optional
  "limit": 10,                 // Optional, max results
  "filters": {}                // Optional additional filters
}
```

### Background Processing

#### Rebuild User Index
```http
POST /users/{username}/rebuild-index
```
Rebuild the search and knowledge graph indexes for a user's documents.

## Username Normalization

Usernames are automatically normalized for safe filesystem usage:

- **Original**: `"María García"`
- **Normalized**: `"maria_garcia"`

**Rules:**
1. Convert to lowercase
2. Remove accents and special characters
3. Replace spaces and special chars with underscores
4. Remove consecutive underscores
5. Trim leading/trailing underscores

## Directory Structure

```
documentos/
├── juan_perez/
│   ├── machine_learning/
│   │   ├── paper1.pdf
│   │   └── paper2.pdf
│   ├── transformers/
│   │   └── attention_paper.pdf
│   └── computer_vision/
│       └── cnn_paper.pdf
├── maria_garcia/
│   ├── nlp/
│   │   └── bert_paper.pdf
│   └── robotics/
│       └── control_paper.pdf
└── admin/
    └── general/
        └── overview.pdf
```

## Streaming Responses

For real-time interaction, enable streaming in chat requests:

```json
{
  "query": "Explain neural networks",
  "stream": true
}
```

The response will be streamed as Server-Sent Events (SSE).

## Background Processing

Document processing (PDF validation, text extraction, embedding generation, knowledge graph updates) happens in the background to provide immediate upload responses.

### Processing Pipeline

1. **Upload**: PDF saved to user's directory
2. **Validation**: PDF integrity check (background)
3. **Text Extraction**: Extract text content
4. **Chunking**: Intelligent text chunking
5. **Embeddings**: Generate vector embeddings
6. **Knowledge Graph**: Extract entities and relationships
7. **Indexing**: Update search indexes

## Configuration

### Environment Variables

```env
# Required
DEEPSEEK_API_KEY=your_api_key

# Optional
GOOGLE_API_KEY=your_google_api_key

# FastAPI Settings (optional)
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
FASTAPI_DEBUG=false
FASTAPI_MAX_UPLOAD_SIZE=52428800  # 50MB
```

### Configuration File

The system uses `config.py` with the following key settings:

```python
fastapi = FastAPIConfig(
    host="0.0.0.0",
    port=8000,
    max_upload_size=50 * 1024 * 1024,  # 50MB
    allowed_file_types=[".pdf"],
    rate_limit_per_minute=60
)
```

## Error Handling

The API provides comprehensive error handling with structured error responses:

```json
{
  "detail": "Error description",
  "error_code": "UPLOAD_FAILED",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Common Error Codes

- `UPLOAD_FAILED`: File upload error
- `PDF_INVALID`: PDF validation failed
- `USER_NOT_FOUND`: User directory doesn't exist
- `PROCESSING_ERROR`: Document processing failed
- `SEARCH_ERROR`: Search operation failed

## Performance Considerations

### Upload Limits
- Maximum file size: 50MB per PDF
- Supported formats: PDF only
- Rate limit: 60 requests per minute per IP

### Background Processing
- Parallel document processing
- Embedding caching for performance
- Database optimization
- Intelligent chunking

### Scaling
- Multiple worker processes supported
- Redis integration ready for task queues
- Database connection pooling
- Memory-efficient processing

## Development

### Running in Development Mode

```bash
python run_fastapi.py --reload --debug --log-level debug
```

### Testing

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/
```

### API Testing

Use the built-in documentation at `/docs` for interactive testing, or use curl/Postman with the examples above.

## Security Notes

- Configure CORS appropriately for production
- Consider implementing authentication/authorization
- Validate file uploads thoroughly
- Rate limiting is recommended for production
- Use HTTPS in production environments

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Database Errors**: Check database permissions and paths
3. **Upload Failures**: Verify file size limits and disk space
4. **API Key Errors**: Ensure DEEPSEEK_API_KEY is set correctly

### Debug Mode

Enable debug mode for detailed logging:

```bash
python run_fastapi.py --debug --log-level debug
```

### Logs

Check logs in the `logs/` directory for detailed error information.

## Future Enhancements

- [ ] User authentication and authorization
- [ ] WebSocket support for real-time updates
- [ ] Advanced search filters
- [ ] Document version control
- [ ] API key management per user
- [ ] Batch document processing
- [ ] Export functionality (PDF reports, etc.)
- [ ] Integration with more LLM providers

## Support

For issues and questions:
1. Check the logs in `logs/` directory
2. Review the configuration in `config.py`
3. Test with the interactive documentation at `/docs`
4. Check database connectivity with `/health` endpoint