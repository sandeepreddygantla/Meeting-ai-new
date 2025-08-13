# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Meetings AI is a Flask-based document analysis and chat application that processes meeting documents using OpenAI/Azure OpenAI LLM technologies. Features modular architecture with AI-powered document processing, semantic search, and conversational interfaces using PostgreSQL+pgvector for production-grade vector storage.

## Development Commands

### Running the Application
```bash
python flask_app.py
# Visit: http://127.0.0.1:5000/meetingsai/
```

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Create .env file:
OPENAI_API_KEY=your_api_key                 # For OpenAI
SECRET_KEY=your-secure-random-key           # Required for Flask sessions
BASE_PATH=/meetingsai                       # Optional, defaults to /meetingsai

# OR for Azure:
# AZURE_CLIENT_ID=your_azure_client_id
# AZURE_CLIENT_SECRET=your_azure_client_secret  
# AZURE_PROJECT_ID=your_azure_project_id

# PostgreSQL Configuration (required):
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=meetingsai
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password

# Create tiktoken cache directory
mkdir tiktoken_cache
```

### Database Operations
Uses PostgreSQL + pgvector for unified storage:
- PostgreSQL tables for metadata, users, projects, sessions
- pgvector extension for high-performance vector similarity search
- Single database system replacing previous SQLite+FAISS dual architecture

```bash
# PostgreSQL setup (ensure pgvector extension is installed)
createdb meetingsai
psql meetingsai -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

## Architecture Overview

### Clean PostgreSQL+pgvector Architecture
Single unified implementation with clean separation of concerns:

### Core Components
- `flask_app.py` - Main Flask application entry point with PostgreSQL initialization
- `meeting_processor.py` - Global AI client variables: `access_token`, `embedding_model`, `llm`
- `src/database/postgres_manager.py` - PostgreSQL+pgvector database operations
- `src/database/manager.py` - Compatibility layer delegating to PostgresManager
- `src/services/` - Business logic: AuthService, ChatService, DocumentService, UploadService
- `src/api/` - Flask blueprints for routes
- `src/ai/` - LLM operations and query processing

### Key Design Patterns
- **Global Variable Pattern:** All AI operations use globals from `meeting_processor.py`
- **Compatibility Layer:** DatabaseManager delegates to PostgresManager for API compatibility
- **Service Composition:** Shared DatabaseManager instance across services
- **PostgreSQL-Only:** Single database system with pgvector for semantic search
- **Environment Switching:** Modify initialization functions for OpenAI/Azure switching

## Critical Development Rules

### Default Project Upload Confirmation
Confirmation dialog for "Default Project" uploads:
- Shows modal for Default Project or empty selection
- "Continue Upload" button uses color `#FF612B` (orange theme)
- Modal in `templates/chat.html`, closable via ESC/outside click

### LLM Integration Requirements
**MANDATORY:** Always use global variables for AI operations:
```python
from meeting_processor import access_token, embedding_model, llm

# Always check for None before using
if llm is not None:
    response = llm.invoke(prompt)
else:
    logger.error("LLM not available - check API key configuration")
```

**NEVER instantiate directly:** `ChatOpenAI()`, `OpenAIEmbeddings()`, `AzureChatOpenAI()`

### Environment Switching Protocol
Modify only these functions in `meeting_processor.py`:
- `get_access_token()` - Return None for OpenAI, Azure token for Azure
- `get_llm()` - Return ChatOpenAI or AzureChatOpenAI  
- `get_embedding_model()` - Return OpenAIEmbeddings or AzureOpenAIEmbeddings

### Database Access Pattern
Always use DatabaseManager (automatically delegates to PostgreSQL):
```python
db_manager = DatabaseManager()  # Initializes PostgresManager internally
documents = db_manager.get_all_documents(user_id)
```

### PostgreSQL Connection Management
Direct PostgreSQL access when needed:
```python
from src.database.postgres_manager import PostgresManager

postgres_manager = PostgresManager(
    host="localhost",
    database="meetingsai", 
    user="postgres",
    password="your_password",
    port=5432,
    vector_dimension=1536
)
```

## IIS Deployment & Performance

### IIS Configuration
**web.config requirements:**
- **WSGI Handler:** `flask_app.app`
- **Python Path:** Application root directory
- **Base Path:** Routes support `/meetingsai` prefix via `BASE_PATH`

### Performance Patterns
- **Vector Operations:** pgvector extension with optimized similarity search
- **Tiktoken Cache:** Directory at `tiktoken_cache/` for token caching
- **Connection Pooling:** PostgreSQL connection management with context managers
- **Session Management:** PostgreSQL session backend for production scalability
- **In-Memory Processing:** Documents processed in memory, no physical file storage needed

## Frontend Architecture

### Mention System
Located in `static/js/modules/mentions.js`:
- `@project:name` - Filter by project
- `@meeting:name` - Filter by meeting  
- `@date:today|yesterday|YYYY-MM-DD` - Date filtering
- `#folder` - Folder navigation
- `#folder>` - Show folder contents

### Upload Modal
- Event listeners set when modal opens (not page load)
- Supports click, drag/drop, file selection
- Prevents duplicate event listeners

## File Processing Pipeline
1. **Upload & Validation:** File type validation, SHA-256 deduplication
2. **Content Extraction:** .docx, .pdf, .txt support with fallbacks
3. **AI Analysis:** LLM metadata extraction (topics, participants, decisions)
4. **Chunking & Embedding:** RecursiveCharacterTextSplitter + text-embedding-3-large
5. **Storage:** PostgreSQL metadata + pgvector embeddings in single database
6. **Background Processing:** ThreadPoolExecutor with job tracking

## Environment Variables
```bash
# AI Configuration
OPENAI_API_KEY=sk-...                    # For OpenAI
# OR for Azure:
AZURE_CLIENT_ID=...
AZURE_CLIENT_SECRET=...
AZURE_PROJECT_ID=...

# Database Configuration (PostgreSQL + pgvector)
POSTGRES_HOST=localhost                  # PostgreSQL host
POSTGRES_PORT=5432                       # PostgreSQL port
POSTGRES_DB=meetingsai                   # Database name
POSTGRES_USER=postgres                   # Database user
POSTGRES_PASSWORD=your_password          # Database password

# Application Configuration  
BASE_PATH=/meetingsai                   # Route prefix
SECRET_KEY=your-flask-secret-key        # Flask sessions
TIKTOKEN_CACHE_DIR=tiktoken_cache       # Token caching
```

## Troubleshooting Common Issues

### PostgreSQL Connection Issues
Check PostgreSQL service and pgvector extension:
```bash
# Check PostgreSQL status
systemctl status postgresql
# OR
pg_isready -h localhost -p 5432

# Verify pgvector extension
psql meetingsai -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

### Enhanced Search Issues
- Check user_id filtering in `src/database/postgres_manager.py`
- Monitor logs for "Enhanced search returned 0 results"
- Ensure enhanced processing for document-specific queries
- Verify vector embeddings are being stored correctly

### LLM Initialization Failures  
- Verify environment variables are set correctly
- Check `logs/flask_app.log` for initialization errors
- Ensure tiktoken cache directory exists and is writable

### Logging
- **Main app logs**: `logs/flask_app.log`
- **Processor logs**: `logs/meeting_processor.log`
- **Log levels**: INFO (default)

## Development Patterns

### Notification System
Manual-close notifications in `static/script.js`:
- No auto-dismiss, stacking with proper positioning  
- ESC key support, close button with hover effects

### Service Composition
Services share DatabaseManager instance:
```python
db_manager = DatabaseManager()
services = {
    'auth': AuthService(db_manager),
    'chat': ChatService(db_manager, processor),
    'document': DocumentService(db_manager),
    'upload': UploadService(db_manager, processor)
}
```

## Development Commands

### Database Inspection
```bash
# PostgreSQL database checks
psql meetingsai -c "SELECT COUNT(*) FROM documents;"
psql meetingsai -c "SELECT COUNT(*) FROM document_chunks;"
psql meetingsai -c "SELECT COUNT(*) FROM users;"

# Check vector embeddings
psql meetingsai -c "SELECT COUNT(*) FROM document_chunks WHERE embedding IS NOT NULL;"

# Validate environment setup
python -c "
from meeting_processor import access_token, embedding_model, llm
print(f'LLM: {\"✓\" if llm else \"✗\"}')
print(f'Embedding: {\"✓\" if embedding_model else \"✗\"}')"

# Test PostgreSQL connection
python -c "
from src.database.postgres_manager import PostgresManager
try:
    pm = PostgresManager()
    stats = pm.get_database_stats()
    print(f'PostgreSQL: ✓ ({stats.get(\"total_documents\", 0)} docs)')
except Exception as e:
    print(f'PostgreSQL: ✗ ({e})')
"
```

## Processing Strategy

### PostgreSQL+pgvector Processing
`ChatService` leverages unified PostgreSQL processing for:
- Vector similarity search using pgvector extension
- Complex multi-meeting summaries with metadata filtering
- Date range queries spanning multiple meetings
- Project-wide analysis with optimized SQL joins
- Enhanced search with user isolation and permission filtering

### Vector Search Architecture
- **Embedding Model**: text-embedding-3-large (1536 dimensions)
- **Vector Storage**: pgvector extension with optimized indexing
- **Search Strategy**: Cosine similarity with metadata filters
- **Performance**: Single database query combining text and vector search

