# Meeting AI Data Processing & Retrieval Architecture Documentation

**Complete Technical Documentation with Real Function Names and Examples**

## Table of Contents
1. [System Architecture Overview](#system-architecture-overview)
2. [Data Upload and Processing Pipeline](#data-upload-and-processing-pipeline)
3. [Data Chunking and Vector Storage](#data-chunking-and-vector-storage)
4. [Metadata Extraction and SQL Storage](#metadata-extraction-and-sql-storage)
5. [Query Processing and Similarity Search](#query-processing-and-similarity-search)
6. [Hybrid Search and Response Generation](#hybrid-search-and-response-generation)
7. [Database Examples and Storage Details](#database-examples-and-storage-details)

---

## 1. System Architecture Overview

The Meeting AI system uses a **dual-database architecture** combining:

- **FAISS Vector Database**: Stores document embeddings for semantic similarity search
- **SQLite Database**: Stores metadata, user information, and structured data

### Key Components

| Component | File Location | Primary Responsibility |
|-----------|---------------|------------------------|
| **Flask App** | `flask_app.py` | Main application entry point and routing |
| **Meeting Processor** | `meeting_processor.py` | Core document processing and AI operations |
| **Database Manager** | `src/database/manager.py` | Unified interface for SQLite + FAISS |
| **Chat Service** | `src/services/chat_service.py` | Query processing and response generation |
| **Document Service** | `src/services/document_service.py` | Document management operations |
| **Upload Service** | `src/services/upload_service.py` | File upload coordination |

### AI Components

- **LLM**: GPT-4 (`get_llm()` in `meeting_processor.py`)
- **Embeddings**: OpenAI text-embedding-3-large (`get_embedding_model()`)
- **Dimensions**: 3072-dimensional vectors
- **Text Splitter**: `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap)

---

## 2. Data Upload and Processing Pipeline

### Entry Point: File Upload

**Function**: `UploadService.handle_file_upload()`
**Location**: `src/services/upload_service.py`

```python
def handle_file_upload(
    self,
    files: List[Any],
    user_id: str,
    username: str,
    project_id: Optional[str] = None,
    meeting_id: Optional[str] = None
) -> Tuple[bool, Dict[str, Any], str]:
```

### Processing Steps

#### Step 1: File Validation
**Function**: `DocumentService.validate_file_upload()`
```python
def validate_file_upload(self, files: List[Any], project_id: Optional[str], 
                        meeting_id: Optional[str], user_id: str) -> Tuple[bool, str]:
```

- Validates file formats (`.docx`, `.pdf`, `.txt`)
- Checks user access to projects/meetings
- Verifies file upload parameters

#### Step 2: Directory Preparation
**Function**: `DocumentService.prepare_upload_directory()`
```python
def prepare_upload_directory(self, user_id: str, username: str, 
                           project_id: Optional[str]) -> Tuple[bool, str, str]:
```

**Example Directory Structure:**
```
meeting_documents/
└── user_john_doe/
    └── project_quarterly_review/
        ├── meeting_notes_2025-01-15.docx
        └── action_items_2025-01-15.pdf
```

#### Step 3: File Processing and Deduplication
**Function**: `DocumentService.process_file_validation()`
```python
def process_file_validation(self, files: List[Any], upload_folder: str, 
                          user_id: str) -> Tuple[List[Dict[str, str]], 
                                                 List[Dict[str, str]], 
                                                 List[Dict[str, str]]]:
```

**Deduplication Process:**
1. **SHA-256 Hash Calculation**: `DatabaseManager.calculate_file_hash()`
2. **Duplicate Check**: `DatabaseManager.is_file_duplicate()`
3. **Smart Restore**: If duplicate is soft-deleted, automatically restore it

#### Step 4: Background Processing Initiation
**Function**: `DocumentService.start_background_processing()`
```python
def start_background_processing(self, file_list: List[Dict[str, str]], 
                              user_id: str, project_id: Optional[str], 
                              meeting_id: Optional[str]) -> Tuple[bool, str, Optional[str]]:
```

**Creates Job Tracking:**
- **Job ID**: UUID for tracking progress
- **Database Record**: `upload_jobs` table entry
- **Background Thread**: Processes files asynchronously

---

## 3. Data Chunking and Vector Storage

### Document Content Extraction

**Function**: `EnhancedMeetingDocumentProcessor.read_document_content()`
**Location**: `meeting_processor.py`

```python
def read_document_content(self, file_path: str) -> str:
```

**Supported Formats:**
- **TXT**: Direct UTF-8 reading
- **DOCX**: Uses `python-docx` library
- **PDF**: Uses `PyPDF2` for text extraction

### Document Parsing and Intelligence Extraction

**Function**: `EnhancedMeetingDocumentProcessor.parse_document_content()`
```python
def parse_document_content(self, content: str, filename: str, user_id: str, 
                         project_id: str = None, meeting_id: str = None) -> MeetingDocument:
```

**AI Extraction Process:**
1. **Date Extraction**: `extract_date_from_filename()` with multiple fallback strategies
2. **LLM Parsing**: Extracts structured metadata using GPT-4
3. **Content Summary**: `create_content_summary()` generates condensed summaries

### Text Chunking Process

**Function**: `EnhancedMeetingDocumentProcessor.chunk_document()`
```python
def chunk_document(self, document: MeetingDocument, intelligence_data: Dict = None) -> List[DocumentChunk]:
```

**Chunking Configuration:**
- **Chunk Size**: 1000 characters
- **Overlap**: 200 characters
- **Separators**: `["\n\n", "\n", ". ", "? ", "! ", " ", ""]`

**Intelligence Enhancement Process:**
```python
def _create_intelligent_chunks(self, document: MeetingDocument, intelligence_data: Dict) -> List[DocumentChunk]:
```

**Per-Chunk Processing:**
1. **Base Chunking**: `RecursiveCharacterTextSplitter.split_text()`
2. **Embedding Generation**: `embedding_model.embed_query(chunk_content)`
3. **Intelligence Extraction**: `_extract_chunk_intelligence()`
4. **Metadata Enhancement**: Adds speakers, decisions, actions, topics

### Vector Storage

**Function**: `VectorOperations.add_vectors()`
**Location**: `src/database/vector_operations.py`

```python
def add_vectors(self, vectors: List[np.ndarray], chunk_ids: List[str]):
```

**FAISS Implementation:**
- **Index Type**: `IndexFlatIP` (Inner Product for cosine similarity)
- **Vector Normalization**: `faiss.normalize_L2(vectors_array)`
- **Automatic Persistence**: `save_index()` after additions

---

## 4. Metadata Extraction and SQL Storage

### Database Schema

#### Documents Table
```sql
CREATE TABLE documents (
    document_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    date TIMESTAMP NOT NULL,
    title TEXT,
    content_summary TEXT,
    main_topics TEXT,         -- JSON array
    past_events TEXT,         -- JSON array
    future_actions TEXT,      -- JSON array
    participants TEXT,        -- JSON array
    chunk_count INTEGER,
    file_size INTEGER,
    user_id TEXT,
    meeting_id TEXT,
    project_id TEXT,
    folder_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_deleted BOOLEAN DEFAULT 0,
    deleted_at TIMESTAMP NULL,
    deleted_by TEXT NULL
);
```

#### Chunks Table
```sql
CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    start_char INTEGER,
    end_char INTEGER,
    user_id TEXT,
    meeting_id TEXT,
    project_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_deleted BOOLEAN DEFAULT 0,
    deleted_at TIMESTAMP NULL
);
```

### Metadata Storage Examples

#### Example Document Record
```json
{
    "document_id": "meeting_notes_2025-01-15.docx_20250115_143000",
    "filename": "meeting_notes_2025-01-15.docx",
    "date": "2025-01-15T14:30:00",
    "title": "Q1 Planning Meeting",
    "content_summary": "Quarterly planning session covering goals, resource allocation, and timeline review with action items for team leads.",
    "main_topics": "[\"Q1 Goals\", \"Resource Planning\", \"Timeline Review\", \"Budget Allocation\"]",
    "past_events": "[\"Q4 Performance Review\", \"Budget Approval\", \"Team Restructuring\"]",
    "future_actions": "[\"Submit Q1 proposals by Jan 20\", \"Schedule team meetings\", \"Review resource requests\"]",
    "participants": "[\"John Smith\", \"Sarah Johnson\", \"Mike Chen\", \"Lisa Rodriguez\"]",
    "chunk_count": 15,
    "file_size": 8456,
    "user_id": "user_12345",
    "project_id": "proj_quarterly_2025",
    "meeting_id": "meet_q1_planning",
    "folder_path": "user_john/project_quarterly_planning"
}
```

#### Example Chunk Record
```json
{
    "chunk_id": "meeting_notes_2025-01-15.docx_20250115_143000_chunk_3",
    "document_id": "meeting_notes_2025-01-15.docx_20250115_143000",
    "filename": "meeting_notes_2025-01-15.docx",
    "chunk_index": 3,
    "content": "Sarah Johnson presented the Q1 budget allocation proposal, highlighting a 15% increase in the development budget. The team agreed to prioritize the mobile app enhancement project over the web redesign initiative. Action item: Mike to prepare detailed cost breakdown by January 20th.",
    "start_char": 1823,
    "end_char": 2187,
    "user_id": "user_12345",
    "meeting_id": "meet_q1_planning",
    "project_id": "proj_quarterly_2025"
}
```

### Enhanced Intelligence Storage

**Function**: `DatabaseManager.add_document()`
```python
def add_document(self, document, chunks: List):
```

**Intelligence Metadata Per Chunk:**
- **speakers**: JSON array of participant names
- **speaker_contributions**: JSON object of contribution summaries
- **topics**: JSON array of topics discussed
- **decisions**: JSON array of decisions made
- **actions**: JSON array of action items
- **questions**: JSON array of questions asked
- **importance_score**: Float (0.0-1.0) indicating chunk importance

---

## 5. Query Processing and Similarity Search

### Query Entry Point

**Function**: `ChatService.process_chat_query()`
**Location**: `src/services/chat_service.py`

```python
def process_chat_query(
    self,
    message: str,
    user_id: str,
    document_ids: Optional[List[str]] = None,
    project_id: Optional[str] = None,
    project_ids: Optional[List[str]] = None,
    meeting_ids: Optional[List[str]] = None,
    date_filters: Optional[Dict[str, Any]] = None,
    folder_path: Optional[str] = None
) -> Tuple[str, List[str], str]:
```

### Query Intelligence Analysis

**Function**: `ChatService._should_use_enhanced_processing()`
```python
def _should_use_enhanced_processing(self, message: str, user_id: str, ...) -> bool:
```

**Analysis Functions:**
1. **Summary Detection**: `detect_enhanced_summary_query()`
2. **Timeframe Detection**: `detect_timeframe_from_query()`
3. **Project Analysis**: `detect_project_summary_query()`
4. **Filter Analysis**: `analyze_query_for_filters()`

### Enhanced Query Processing

**Function**: `EnhancedMeetingDocumentProcessor.answer_query_with_intelligence()`
**Location**: `meeting_processor.py`

```python
def answer_query_with_intelligence(
    self, 
    query: str, 
    user_id: str = None,
    document_ids: List[str] = None,
    project_id: str = None,
    meeting_ids: List[str] = None,
    date_filters: Dict[str, Any] = None,
    folder_path: str = None,
    context_limit: int = 100,
    include_context: bool = False
) -> Union[str, Tuple[str, str]]:
```

### Query Vector Generation

**Process:**
1. **Query Preprocessing**: Clean and prepare query text
2. **Embedding Generation**: `embedding_model.embed_query(query)`
3. **Vector Normalization**: Prepare for cosine similarity search

### Similarity Search Execution

**Function**: `DatabaseManager.enhanced_search_with_metadata()`
**Location**: `src/database/manager.py`

```python
def enhanced_search_with_metadata(self, query_embedding: np.ndarray, user_id: str, 
                                filters: Dict = None, top_k: int = 20) -> List[Dict]:
```

**Search Pipeline:**
1. **Vector Search**: `VectorOperations.search_similar_chunks()`
2. **Chunk Retrieval**: `get_chunks_by_ids()`
3. **User Filtering**: Strict user_id validation
4. **Metadata Filtering**: Apply project, meeting, date filters
5. **Score Ranking**: Sort by similarity scores

---

## 6. Hybrid Search and Response Generation

### Hybrid Search Components

#### Semantic Search
**Function**: `VectorOperations.search_similar_chunks()`
```python
def search_similar_chunks(self, query_embedding: np.ndarray, top_k: int = 20) -> List[Tuple[str, float]]:
```

**FAISS Search Process:**
1. **Query Vector Normalization**: `faiss.normalize_L2(query_vector)`
2. **Index Search**: `self.index.search(query_vector, top_k)`
3. **Position Mapping**: `_map_positions_to_chunk_ids()`

#### Keyword Search
**Function**: `SQLiteOperations.keyword_search_chunks()`
```python
def keyword_search_chunks(self, keywords: List[str], limit: int = 50) -> List[str]:
```

**SQL Implementation:**
```sql
SELECT DISTINCT chunk_id FROM chunks 
WHERE content LIKE '%keyword1%' 
   OR content LIKE '%keyword2%' 
ORDER BY chunk_index 
LIMIT ?
```

### Response Generation Process

**Function**: `EnhancedMeetingDocumentProcessor._generate_intelligence_response()`
```python
def _generate_intelligence_response(self, query: str, enhanced_results: List[Dict], 
                                  user_id: str) -> Tuple[str, str]:
```

### Context Analysis and Response Scaling

**Context Analysis Function**: `_analyze_context_richness()`
```python
def _analyze_context_richness(self, enhanced_results: List[Dict]) -> Dict[str, Any]:
```

**Analyzes:**
- **Speaker Diversity**: Unique speakers across chunks
- **Meeting Coverage**: Number of different meetings
- **Content Volume**: Total character count
- **Decision Count**: Number of decisions found
- **Action Items**: Number of action items
- **Time Span**: Date range of content

**Response Requirements Function**: `_determine_response_requirements()`
```python
def _determine_response_requirements(self, query: str, context_analysis: Dict[str, Any]) -> Dict[str, Any]:
```

**Dynamic Scaling:**
- **Base Response**: 200-800 words
- **Rich Context**: 400-1000 words  
- **Technical Queries**: +150 words
- **Complex Queries**: +100 words

### LLM Response Generation

**Comprehensive Prompt Structure:**
```python
response_prompt = f"""
User Query: "{query}"

Based on the following meeting context and intelligence data, provide a comprehensive, well-structured response.

Context Analysis:
- {context_analysis['speakers_count']} unique speakers
- {context_analysis['meetings_count']} meetings  
- {context_analysis['decisions_count']} decisions
- {context_analysis['actions_count']} action items
- Content spans {context_analysis['time_span_days']} days

Meeting Intelligence Context:
{comprehensive_context}

Response Requirements:
- Minimum {response_requirements['min_words']} words
- Maximum {response_requirements['max_words']} words  
- Detail level: {response_requirements['detail_level']}
- Include specific quotes and attributions
- Organize chronologically when relevant
- Always cite document filenames

Provide a detailed, accurate response that directly addresses the user's question.
"""
```

### Response Post-Processing

**Function**: `_post_process_response()`
1. **Citation Enhancement**: Ensure proper document filename citations
2. **Structure Validation**: Check response organization
3. **Length Validation**: Verify word count requirements
4. **Context Preservation**: Maintain important details

---

## 7. Database Examples and Storage Details

### Storage Architecture Comparison

| Data Type | Storage Location | Access Method | Purpose |
|-----------|------------------|---------------|---------|
| **Vector Embeddings** | FAISS Index File | Similarity search | Semantic search |
| **Document Metadata** | SQLite `documents` table | SQL queries | Filtering, organization |
| **Chunk Content** | SQLite `chunks` table | ID-based retrieval | Text content |
| **User Data** | SQLite `users` table | Authentication | Access control |
| **Intelligence Data** | Enhanced chunk fields | JSON deserialization | AI insights |

### Example Data Flow

#### 1. Document Upload
```
File: "Q1_Planning_Meeting_2025-01-15.docx" (12 KB)
↓
Content Extraction: "Meeting started at 2:30 PM with John Smith presenting..."
↓  
Intelligence Extraction: {
    "participants": ["John Smith", "Sarah Johnson", "Mike Chen"],
    "main_topics": ["Q1 Goals", "Budget Review", "Resource Planning"],
    "decisions": ["Approved 15% budget increase", "Prioritize mobile app"],
    "actions": ["Mike: Cost breakdown by Jan 20", "Sarah: Schedule follow-up"]
}
↓
Document Record Created in SQLite
```

#### 2. Text Chunking
```
Chunk 0: "Meeting started at 2:30 PM with John Smith presenting the Q1 goals..."
→ Vector: [0.123, -0.456, 0.789, ...] (3072 dimensions)
→ Intelligence: {"speakers": ["John Smith"], "topics": ["Q1 Goals"]}

Chunk 1: "Sarah Johnson discussed the budget allocation, noting a 15% increase..."
→ Vector: [-0.234, 0.567, -0.123, ...] (3072 dimensions)  
→ Intelligence: {"speakers": ["Sarah Johnson"], "decisions": ["15% budget increase"]}
```

#### 3. Query Processing
```
User Query: "What did Sarah discuss about the budget?"
↓
Query Vector Generation: [0.345, -0.678, 0.234, ...] (3072 dimensions)
↓
FAISS Similarity Search: Returns chunk IDs with similarity scores
↓
Chunk Retrieval from SQLite: Get content for matching chunk IDs
↓
User Filtering: Only return chunks where user_id matches
↓
Metadata Filtering: Apply speaker filter for "Sarah"
↓
Response Generation: "Sarah Johnson discussed the budget allocation..."
```

### Performance Characteristics

| Operation | Time Complexity | Storage Space | Notes |
|-----------|----------------|---------------|--------|
| **Vector Search** | O(log n) with FAISS | 3072 × 4 bytes per vector | Approximate nearest neighbors |
| **Metadata Filtering** | O(n) linear scan | Minimal overhead | Post-search filtering |
| **Chunk Retrieval** | O(1) by ID | Text storage size | Direct SQLite lookup |
| **User Isolation** | O(1) index lookup | Index space | Enforced at query time |

### Data Persistence and Reliability

#### Backup Strategy
- **SQLite**: File-based, atomic transactions
- **FAISS Index**: Periodic saves after vector additions
- **File Storage**: Original documents preserved in filesystem

#### Consistency Model
- **Vector-SQL Sync**: Chunks added to both systems transactionally
- **Soft Deletion**: Vectors remain in FAISS, flagged in SQLite
- **Recovery**: FAISS index can be rebuilt from SQLite data

---

## Summary

This Meeting AI system implements a sophisticated document processing and retrieval pipeline that:

1. **Ingests** documents with comprehensive validation and deduplication
2. **Processes** content using advanced AI techniques for intelligence extraction  
3. **Chunks** documents with intelligent context preservation
4. **Stores** data in a hybrid SQLite + FAISS architecture
5. **Searches** using semantic similarity with metadata filtering
6. **Generates** responses with dynamic scaling and comprehensive context analysis

The architecture ensures scalable, accurate, and user-isolated document retrieval with rich AI-powered insights while maintaining data consistency and performance.