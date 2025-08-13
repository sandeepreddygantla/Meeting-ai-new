"""
PostgreSQL + pgvector Database Manager
Production-ready replacement for the current SQLite + FAISS setup
"""

import logging
import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import os
import json
import hashlib
import uuid
from contextlib import contextmanager

# Get the AI clients and legacy models from meeting_processor
from meeting_processor import (
    access_token, embedding_model, llm,
    User, Project, Meeting, MeetingDocument
)

# Import the correct DocumentChunk model from src.models.document
from src.models.document import DocumentChunk

logger = logging.getLogger(__name__)

class PostgresManager:
    """
    Production PostgreSQL + pgvector database manager for Meeting AI.
    Replaces the current SQLite + FAISS dual-database approach.
    """
    
    def __init__(self, 
                 host: str = "localhost",
                 database: str = "meetingsai", 
                 user: str = "postgres",
                 password: str = "Sandeep@0904",
                 port: int = 5432,
                 vector_dimension: int = 1536):
        """
        Initialize PostgreSQL connection and setup.
        
        Args:
            host: PostgreSQL host
            database: Database name
            user: Database user
            password: Database password
            port: Database port
            vector_dimension: Vector embedding dimension
        """
        self.db_config = {
            'host': host,
            'database': database,
            'user': user,
            'password': password,
            'port': port
        }
        self.vector_dimension = vector_dimension
        self._connection_pool = None
        
        # Add db_path property for compatibility with existing code
        self.db_path = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        
        # Initialize database schema
        self._ensure_database_exists()
        self._initialize_schema()
        self._run_migrations()
        
        logger.info(f"PostgresManager initialized for database '{database}' with {vector_dimension}d vectors")
    
    @contextmanager
    def get_connection(self):
        """Get a database connection with automatic cleanup"""
        conn = None
        try:
            conn = psycopg2.connect(**self.db_config)
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    @contextmanager
    def get_cursor(self, dict_cursor=False):
        """Get a database cursor with automatic cleanup"""
        with self.get_connection() as conn:
            cursor_factory = RealDictCursor if dict_cursor else None
            cursor = conn.cursor(cursor_factory=cursor_factory)
            try:
                yield conn, cursor
            finally:
                cursor.close()
    
    def _ensure_database_exists(self):
        """Ensure the target database exists, create if not"""
        try:
            # Connect to default postgres database first
            temp_config = self.db_config.copy()
            temp_config['database'] = 'postgres'
            
            with psycopg2.connect(**temp_config) as conn:
                conn.autocommit = True
                with conn.cursor() as cursor:
                    # Check if database exists
                    cursor.execute(
                        "SELECT 1 FROM pg_database WHERE datname = %s;",
                        (self.db_config['database'],)
                    )
                    
                    if not cursor.fetchone():
                        # Create database
                        cursor.execute(
                            f"CREATE DATABASE {self.db_config['database']};"
                        )
                        logger.info(f"Created database '{self.db_config['database']}'")
                    else:
                        logger.info(f"Database '{self.db_config['database']}' already exists")
                        
        except Exception as e:
            logger.error(f"Error ensuring database exists: {e}")
            # Continue anyway - database might exist but connection failed
    
    def _initialize_schema(self):
        """Initialize database schema with all required tables and indexes"""
        try:
            with self.get_cursor() as (conn, cursor):
                # Enable pgvector extension
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                # Users table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        user_id VARCHAR(255) PRIMARY KEY,
                        username VARCHAR(100) UNIQUE NOT NULL,
                        email VARCHAR(255) UNIQUE NOT NULL,
                        full_name VARCHAR(255),
                        password_hash TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_login TIMESTAMP,
                        is_active BOOLEAN DEFAULT TRUE
                    );
                """)
                
                # Projects table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS projects (
                        project_id VARCHAR(255) PRIMARY KEY,
                        user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                        project_name VARCHAR(255) NOT NULL,
                        description TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(user_id, project_name)
                    );
                """)
                
                # Meetings table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS meetings (
                        meeting_id VARCHAR(255) PRIMARY KEY,
                        user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                        project_id VARCHAR(255) REFERENCES projects(project_id) ON DELETE SET NULL,
                        meeting_name VARCHAR(255) NOT NULL,
                        meeting_date TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Documents table
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS documents (
                        document_id VARCHAR(255) PRIMARY KEY,
                        user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                        project_id VARCHAR(255) REFERENCES projects(project_id) ON DELETE SET NULL,
                        meeting_id VARCHAR(255) REFERENCES meetings(meeting_id) ON DELETE SET NULL,
                        filename VARCHAR(255) NOT NULL,
                        original_filename VARCHAR(255),
                        file_path TEXT,
                        file_size BIGINT,
                        file_hash VARCHAR(64),
                        content_type VARCHAR(100),
                        upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        processed_date TIMESTAMP,
                        
                        -- AI-extracted metadata
                        content_summary TEXT,
                        main_topics TEXT,
                        participants TEXT,
                        key_decisions TEXT,
                        action_items TEXT,
                        
                        -- Processing status
                        processing_status VARCHAR(50) DEFAULT 'pending',
                        chunk_count INTEGER DEFAULT 0,
                        
                        -- Folder organization (for # functionality)
                        folder_path TEXT,
                        
                        -- Soft deletion support
                        is_deleted BOOLEAN DEFAULT FALSE,
                        deleted_at TIMESTAMP,
                        deleted_by VARCHAR(255) REFERENCES users(user_id)
                    );
                """)
                
                # Document chunks table with pgvector support
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        chunk_id VARCHAR(255) PRIMARY KEY,
                        document_id VARCHAR(255) NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
                        user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                        project_id VARCHAR(255) REFERENCES projects(project_id) ON DELETE SET NULL,
                        meeting_id VARCHAR(255) REFERENCES meetings(meeting_id) ON DELETE SET NULL,
                        
                        -- Chunk content and position
                        content TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        start_char INTEGER,
                        end_char INTEGER,
                        
                        -- Vector embedding (pgvector)
                        embedding VECTOR({self.vector_dimension}),
                        
                        -- Timestamps
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        
                        -- Indexing for performance
                        UNIQUE(document_id, chunk_index)
                    );
                """)
                
                # Sessions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id VARCHAR(255) PRIMARY KEY,
                        user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP NOT NULL,
                        is_active BOOLEAN DEFAULT TRUE
                    );
                """)
                
                # File hashes table for deduplication
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS file_hashes (
                        hash_id VARCHAR(255) PRIMARY KEY,
                        file_hash VARCHAR(64) NOT NULL,
                        filename VARCHAR(255) NOT NULL,
                        original_filename VARCHAR(255),
                        file_size BIGINT,
                        user_id VARCHAR(255) NOT NULL REFERENCES users(user_id),
                        project_id VARCHAR(255) REFERENCES projects(project_id),
                        meeting_id VARCHAR(255) REFERENCES meetings(meeting_id),
                        document_id VARCHAR(255) REFERENCES documents(document_id),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(file_hash, filename, user_id)
                    );
                """)
                
                # Upload jobs table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS upload_jobs (
                        job_id VARCHAR(255) PRIMARY KEY,
                        user_id VARCHAR(255) NOT NULL REFERENCES users(user_id),
                        project_id VARCHAR(255) REFERENCES projects(project_id),
                        meeting_id VARCHAR(255) REFERENCES meetings(meeting_id),
                        total_files INTEGER NOT NULL,
                        processed_files INTEGER DEFAULT 0,
                        failed_files INTEGER DEFAULT 0,
                        status VARCHAR(50) DEFAULT 'pending',
                        error_message TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Create indexes for performance
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_docs_user_id ON documents(user_id);",
                    "CREATE INDEX IF NOT EXISTS idx_docs_project_id ON documents(project_id);",
                    "CREATE INDEX IF NOT EXISTS idx_docs_upload_date ON documents(upload_date);",
                    "CREATE INDEX IF NOT EXISTS idx_docs_file_hash ON documents(file_hash);",
                    "CREATE INDEX IF NOT EXISTS idx_docs_is_deleted ON documents(is_deleted);",
                    "CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON document_chunks(document_id);",
                    "CREATE INDEX IF NOT EXISTS idx_chunks_user_id ON document_chunks(user_id);",
                    "CREATE INDEX IF NOT EXISTS idx_chunks_project_id ON document_chunks(project_id);",
                    "CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);",
                    "CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at);",
                    # Vector index for similarity search
                    f"CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON document_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);"
                ]
                
                for index_sql in indexes:
                    try:
                        cursor.execute(index_sql)
                    except Exception as e:
                        logger.warning(f"Index creation warning: {e}")
                        # Continue with other indexes
                
                conn.commit()
                logger.info("Database schema initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing schema: {e}")
            raise
    
    def _run_migrations(self):
        """Run database migrations for schema updates"""
        try:
            with self.get_cursor() as (conn, cursor):
                # Check if folder_path column exists in documents table
                cursor.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'documents' AND column_name = 'folder_path';
                """)
                
                if not cursor.fetchone():
                    logger.info("Adding folder_path column to documents table...")
                    cursor.execute("ALTER TABLE documents ADD COLUMN folder_path TEXT;")
                    
                    # Update existing documents with a default folder path based on project
                    cursor.execute("""
                        UPDATE documents 
                        SET folder_path = CASE 
                            WHEN project_id IS NOT NULL THEN 'user_folder/project_' || project_id
                            ELSE 'user_folder/default_project'
                        END
                        WHERE folder_path IS NULL;
                    """)
                    
                    logger.info("folder_path column added and existing documents updated")
                
                # Check for existing unique constraints on documents table
                cursor.execute("""
                    SELECT constraint_name 
                    FROM information_schema.table_constraints 
                    WHERE table_name = 'documents' 
                    AND constraint_type = 'UNIQUE'
                    AND constraint_name LIKE '%user%filename%';
                """)
                
                existing_constraints = [row[0] for row in cursor.fetchall()]
                logger.info(f"Found existing constraints: {existing_constraints}")
                
                # Drop any old unique constraints that conflict with our new approach
                for constraint in existing_constraints:
                    if constraint in ['documents_user_filename_unique', 'documents_user_filename_status_unique']:
                        try:
                            cursor.execute(f"ALTER TABLE documents DROP CONSTRAINT IF EXISTS {constraint};")
                            logger.info(f"Dropped constraint: {constraint}")
                        except Exception as e:
                            logger.warning(f"Could not drop constraint {constraint}: {e}")
                
                # Check if we need to add the correct deduplication constraint
                cursor.execute("""
                    SELECT constraint_name 
                    FROM information_schema.table_constraints 
                    WHERE table_name = 'documents' AND constraint_name = 'documents_user_filename_unique_v2';
                """)
                
                if not cursor.fetchone():
                    logger.info("Adding improved deduplication constraint...")
                    try:
                        # Clean up any existing duplicates first (including empty file_hash)
                        cursor.execute("""
                            DELETE FROM documents 
                            WHERE document_id NOT IN (
                                SELECT MIN(document_id) 
                                FROM documents 
                                GROUP BY user_id, filename
                            );
                        """)
                        logger.info("Cleaned up duplicate documents")
                        
                        # Use filename-based constraint that handles empty hashes properly
                        # This is more reliable than hash-based since hashes can be empty
                        cursor.execute("""
                            ALTER TABLE documents 
                            ADD CONSTRAINT documents_user_filename_unique_v2 
                            UNIQUE (user_id, filename);
                        """)
                        logger.info("Added unique constraint on user_id + filename")
                    except Exception as e:
                        logger.warning(f"Could not add filename-based constraint: {e}")
                        # Continue without unique constraint - handle duplicates in application logic
                
                conn.commit()
                logger.info("Database migrations completed successfully")
                
        except Exception as e:
            logger.error(f"Error running migrations: {e}")
            # Don't raise - migrations are optional
    
    # Document Operations
    def add_document(self, document, chunks: List):
        """Add a document and all its chunks to the database"""
        try:
            with self.get_cursor() as (conn, cursor):
                # Get file hash, ensure it's not empty
                file_hash = getattr(document, 'file_hash', '')
                if not file_hash:
                    # Generate a simple hash from filename + user_id if file_hash is empty
                    file_hash = hashlib.sha256(f"{document.user_id}_{document.filename}".encode()).hexdigest()[:16]
                    logger.warning(f"Using fallback hash for {document.filename}: {file_hash} - This should not happen with proper document processing!")
                
                # Check if document already exists (any status)
                cursor.execute("""
                    SELECT document_id, processing_status FROM documents 
                    WHERE user_id = %s AND filename = %s
                    ORDER BY upload_date DESC LIMIT 1;
                """, (document.user_id, document.filename))
                
                existing_doc = cursor.fetchone()
                
                if existing_doc:
                    document_id, status = existing_doc
                    
                    if status == 'pending':
                        # Update existing pending record to completed
                        logger.info(f"Updating existing pending document: {document_id}")
                        
                        cursor.execute("""
                            UPDATE documents SET
                                original_filename = %s, file_path = %s, file_size = %s, file_hash = %s,
                                content_type = %s, content_summary = %s, main_topics = %s, 
                                participants = %s, key_decisions = %s, action_items = %s,
                                processing_status = 'completed', chunk_count = %s,
                                processed_date = CURRENT_TIMESTAMP
                            WHERE document_id = %s;
                        """, (
                            getattr(document, 'original_filename', document.filename),
                            getattr(document, 'file_path', ''), getattr(document, 'file_size', 0),
                            file_hash, getattr(document, 'content_type', ''),
                            getattr(document, 'content_summary', ''), getattr(document, 'main_topics', ''),
                            getattr(document, 'participants', ''), getattr(document, 'key_decisions', ''),
                            getattr(document, 'action_items', ''), len(chunks), document_id
                        ))
                    else:
                        # Document already completed - skip processing to avoid constraint violation
                        logger.info(f"Document {document.filename} already exists with status {status} - skipping")
                        return document_id
                        
                else:
                    # No existing record found, insert new completed record
                    document_id = str(uuid.uuid4())
                    logger.info(f"Creating new document record: {document_id}")
                    
                    cursor.execute("""
                        INSERT INTO documents (
                            document_id, user_id, project_id, meeting_id, filename, original_filename,
                            file_path, file_size, file_hash, content_type,
                            content_summary, main_topics, participants, key_decisions, action_items,
                            processing_status, chunk_count, folder_path
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                    """, (
                        document_id, document.user_id, document.project_id, document.meeting_id,
                        document.filename, getattr(document, 'original_filename', document.filename),
                        getattr(document, 'file_path', ''), getattr(document, 'file_size', 0),
                        file_hash, getattr(document, 'content_type', ''),
                        getattr(document, 'content_summary', ''), getattr(document, 'main_topics', ''),
                        getattr(document, 'participants', ''), getattr(document, 'key_decisions', ''),
                        getattr(document, 'action_items', ''), 'completed', len(chunks),
                        getattr(document, 'folder_path', None)
                    ))
                
                # Prepare chunk data for batch insert
                chunk_data = []
                for chunk in chunks:
                    if chunk.embedding is not None:
                        # Generate chunk_id since we're using VARCHAR instead of UUID with auto-generation
                        chunk_id = str(uuid.uuid4())
                        chunk_data.append((
                            chunk_id, document_id, document.user_id, document.project_id, document.meeting_id,
                            chunk.content, chunk.chunk_index, chunk.start_char, chunk.end_char,
                            chunk.embedding.tolist() if isinstance(chunk.embedding, np.ndarray) else chunk.embedding
                        ))
                
                # Batch insert chunks
                if chunk_data:
                    execute_values(
                        cursor,
                        """INSERT INTO document_chunks (
                            chunk_id, document_id, user_id, project_id, meeting_id, content, 
                            chunk_index, start_char, end_char, embedding
                        ) VALUES %s""",
                        chunk_data
                    )
                    logger.info(f"Inserted {len(chunk_data)} chunks into document_chunks table")
                else:
                    logger.warning(f"No chunks with embeddings to insert for document {document.filename}")
                
                conn.commit()
                logger.info(f"Successfully added document {document.filename} with {len(chunks)} chunks")
                
                return document_id
                
        except Exception as e:
            logger.error(f"Error adding document {document.filename}: {e}")
            raise
    
    def search_similar_chunks(self, query_embedding: np.ndarray, user_id: str = None, top_k: int = 20) -> List[Tuple[str, float]]:
        """Search for similar chunks using pgvector"""
        try:
            with self.get_cursor() as (conn, cursor):
                # Convert embedding to list for PostgreSQL
                embedding_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
                
                # Build query with optional user filtering
                base_query = """
                    SELECT 
                        chunk_id,
                        1 - (embedding <=> %s::vector) as similarity_score
                    FROM document_chunks c
                    JOIN documents d ON c.document_id = d.document_id
                    WHERE d.is_deleted = FALSE
                """
                
                params = [embedding_list]
                
                if user_id:
                    base_query += " AND c.user_id = %s"
                    params.append(user_id)
                
                base_query += """
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                """
                
                params.extend([embedding_list, top_k])
                
                cursor.execute(base_query, params)
                results = cursor.fetchall()
                
                return [(row[0], row[1]) for row in results]
                
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def get_chunks_by_ids(self, chunk_ids: List[str]):
        """Retrieve chunks by their IDs"""
        try:
            with self.get_cursor(dict_cursor=True) as (conn, cursor):
                cursor.execute("""
                    SELECT 
                        c.*,
                        d.filename,
                        d.content_summary,
                        d.upload_date,
                        p.project_name,
                        m.meeting_name
                    FROM document_chunks c
                    JOIN documents d ON c.document_id = d.document_id
                    LEFT JOIN projects p ON c.project_id = p.project_id
                    LEFT JOIN meetings m ON c.meeting_id = m.meeting_id
                    WHERE c.chunk_id = ANY(%s)
                    ORDER BY d.upload_date DESC, c.chunk_index;
                """, (chunk_ids,))
                
                rows = cursor.fetchall()
                
                # Convert to DocumentChunk objects
                chunks = []
                for row in rows:
                    chunk = DocumentChunk(
                        chunk_id=row['chunk_id'],
                        document_id=row['document_id'],
                        user_id=row['user_id'],
                        content=row['content'],
                        chunk_index=row['chunk_index'],
                        start_char=row['start_char'] or 0,
                        end_char=row['end_char'] or len(row['content'])
                    )
                    
                    # Add metadata
                    chunk.document_title = row['filename']
                    chunk.content_summary = row['content_summary']
                    chunk.project_id = row['project_id']
                    chunk.meeting_id = row['meeting_id']
                    
                    chunks.append(chunk)
                
                return chunks
                
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []
    
    def enhanced_search_with_metadata(self, query_embedding: np.ndarray, user_id: str, 
                                    filters: Dict = None, top_k: int = 20) -> List[Dict]:
        """Enhanced search combining vector similarity with metadata filtering"""
        try:
            with self.get_cursor(dict_cursor=True) as (conn, cursor):
                # Build dynamic query based on filters
                base_query = """
                    SELECT 
                        c.chunk_id,
                        c.content,
                        c.chunk_index,
                        d.filename,
                        d.content_summary,
                        d.upload_date,
                        p.project_name,
                        m.meeting_name,
                        1 - (c.embedding <=> %s::vector) as similarity_score
                    FROM document_chunks c
                    JOIN documents d ON c.document_id = d.document_id
                    LEFT JOIN projects p ON c.project_id = p.project_id
                    LEFT JOIN meetings m ON c.meeting_id = m.meeting_id
                    WHERE d.is_deleted = FALSE AND c.user_id = %s
                """
                
                embedding_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
                params = [embedding_list, user_id]
                
                # Apply filters
                if filters:
                    if filters.get('document_ids'):
                        document_ids = filters['document_ids']
                        if document_ids:
                            placeholders = ', '.join(['%s'] * len(document_ids))
                            base_query += f" AND c.document_id IN ({placeholders})"
                            params.extend(document_ids)
                    
                    if filters.get('project_id'):
                        base_query += " AND c.project_id = %s"
                        params.append(filters['project_id'])
                    
                    if filters.get('meeting_id'):
                        base_query += " AND c.meeting_id = %s"
                        params.append(filters['meeting_id'])
                    
                    if filters.get('meeting_ids'):
                        meeting_ids = filters['meeting_ids']
                        if meeting_ids:
                            placeholders = ', '.join(['%s'] * len(meeting_ids))
                            base_query += f" AND c.meeting_id IN ({placeholders})"
                            params.extend(meeting_ids)
                    
                    if filters.get('date_range'):
                        start_date, end_date = filters['date_range']
                        if start_date:
                            base_query += " AND d.upload_date >= %s"
                            params.append(start_date)
                        if end_date:
                            base_query += " AND d.upload_date <= %s"
                            params.append(end_date)
                    
                    if filters.get('keywords'):
                        keyword_conditions = []
                        for keyword in filters['keywords']:
                            keyword_conditions.append("c.content ILIKE %s")
                            params.append(f"%{keyword}%")
                        if keyword_conditions:
                            base_query += f" AND ({' OR '.join(keyword_conditions)})"
                
                # Add ordering and limit
                base_query += """
                    ORDER BY c.embedding <=> %s::vector
                    LIMIT %s;
                """
                params.extend([embedding_list, top_k])
                
                cursor.execute(base_query, params)
                rows = cursor.fetchall()
                
                # Convert to enhanced results format
                enhanced_results = []
                for row in rows:
                    # Create chunk object
                    chunk = DocumentChunk(
                        chunk_id=row['chunk_id'],
                        document_id=None,  # Not needed for display
                        user_id=user_id,
                        content=row['content'],
                        chunk_index=row['chunk_index'],
                        start_char=0,  # Default value for search results
                        end_char=len(row['content'])  # End of chunk content
                    )
                    
                    # Add metadata
                    chunk.document_title = row['filename']
                    chunk.content_summary = row['content_summary']
                    
                    # Create result with context
                    result = {
                        'chunk': chunk,
                        'similarity_score': float(row['similarity_score']),
                        'context': {
                            'document_title': row['filename'],
                            'document_date': row['upload_date'].isoformat() if row['upload_date'] else '',
                            'chunk_position': str(row['chunk_index'] + 1),
                            'document_summary': row['content_summary'] or '',
                            'project_name': row['project_name'] or '',
                            'meeting_name': row['meeting_name'] or ''
                        }
                    }
                    
                    enhanced_results.append(result)
                
                logger.info(f"Enhanced search returned {len(enhanced_results)} results")
                return enhanced_results
                
        except Exception as e:
            logger.error(f"Error in enhanced search: {e}")
            return []
    
    # User Management
    def create_user(self, username: str, email: str, full_name: str, password_hash: str) -> str:
        """Create a new user"""
        try:
            with self.get_cursor() as (conn, cursor):
                # Generate user_id since we're using VARCHAR instead of UUID with auto-generation
                user_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO users (user_id, username, email, full_name, password_hash)
                    VALUES (%s, %s, %s, %s, %s) RETURNING user_id;
                """, (user_id, username, email, full_name, password_hash))
                
                user_id = cursor.fetchone()[0]
                conn.commit()
                
                logger.info(f"Created user: {username}")
                return str(user_id)
                
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            raise
    
    def get_user_by_username(self, username: str):
        """Get user by username"""
        try:
            with self.get_cursor(dict_cursor=True) as (conn, cursor):
                cursor.execute("""
                    SELECT user_id, username, email, full_name, password_hash, created_at, last_login, is_active
                    FROM users WHERE username = %s AND is_active = TRUE;
                """, (username,))
                
                row = cursor.fetchone()
                if row:
                    return User(
                        user_id=str(row['user_id']),
                        username=row['username'],
                        email=row['email'],
                        full_name=row['full_name'],
                        password_hash=row['password_hash'],
                        created_at=row['created_at'],
                        last_login=row['last_login'],
                        is_active=row['is_active']
                    )
                return None
                
        except Exception as e:
            logger.error(f"Error getting user by username: {e}")
            return None
    
    def get_user_by_id(self, user_id: str):
        """Get user by user_id"""
        try:
            with self.get_cursor(dict_cursor=True) as (conn, cursor):
                cursor.execute("""
                    SELECT user_id, username, email, full_name, password_hash, created_at, last_login, is_active
                    FROM users WHERE user_id = %s AND is_active = TRUE;
                """, (user_id,))
                
                row = cursor.fetchone()
                if row:
                    return User(
                        user_id=str(row['user_id']),
                        username=row['username'],
                        email=row['email'],
                        full_name=row['full_name'],
                        password_hash=row['password_hash'],
                        created_at=row['created_at'],
                        last_login=row['last_login'],
                        is_active=row['is_active']
                    )
                return None
                
        except Exception as e:
            logger.error(f"Error getting user by ID: {e}")
            return None
    
    def update_user_last_login(self, user_id: str):
        """Update user's last login timestamp"""
        try:
            with self.get_cursor() as (conn, cursor):
                cursor.execute("""
                    UPDATE users SET last_login = CURRENT_TIMESTAMP
                    WHERE user_id = %s;
                """, (user_id,))
                
                conn.commit()
                logger.info(f"Updated last login for user: {user_id}")
                
        except Exception as e:
            logger.error(f"Error updating user last login: {e}")
    
    # Project Management
    def create_project(self, user_id: str, project_name: str, description: str = "") -> str:
        """Create a new project for a user"""
        try:
            with self.get_cursor() as (conn, cursor):
                # Generate project_id since we're using VARCHAR instead of UUID with auto-generation
                project_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO projects (project_id, user_id, project_name, description)
                    VALUES (%s, %s, %s, %s);
                """, (project_id, user_id, project_name, description))
                
                conn.commit()
                
                logger.info(f"Created project: {project_name}")
                return str(project_id)
                
        except Exception as e:
            logger.error(f"Error creating project: {e}")
            raise
    
    def get_user_projects(self, user_id: str):
        """Get all projects for a user"""
        try:
            with self.get_cursor(dict_cursor=True) as (conn, cursor):
                cursor.execute("""
                    SELECT project_id, project_name, description, created_at
                    FROM projects 
                    WHERE user_id = %s 
                    ORDER BY created_at DESC;
                """, (user_id,))
                
                rows = cursor.fetchall()
                projects = []
                for row in rows:
                    project = Project(
                        project_id=str(row['project_id']),
                        user_id=user_id,
                        project_name=row['project_name'],
                        description=row['description'],
                        created_at=row['created_at']
                    )
                    projects.append(project)
                
                return projects
                
        except Exception as e:
            logger.error(f"Error getting user projects: {e}")
            return []
    
    # Meeting Management
    def create_meeting(self, user_id: str, project_id: str, meeting_name: str, meeting_date: datetime = None) -> str:
        """Create a new meeting"""
        try:
            with self.get_cursor() as (conn, cursor):
                meeting_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO meetings (meeting_id, user_id, project_id, meeting_name, meeting_date)
                    VALUES (%s, %s, %s, %s, %s) RETURNING meeting_id;
                """, (meeting_id, user_id, project_id, meeting_name, meeting_date))
                
                meeting_id = cursor.fetchone()[0]
                conn.commit()
                
                logger.info(f"Created meeting: {meeting_name}")
                return str(meeting_id)
                
        except Exception as e:
            logger.error(f"Error creating meeting: {e}")
            raise
    
    def get_user_meetings(self, user_id: str, project_id: str = None):
        """Get meetings for a user, optionally filtered by project"""
        try:
            with self.get_cursor(dict_cursor=True) as (conn, cursor):
                base_query = """
                    SELECT m.meeting_id, m.meeting_name, m.meeting_date, m.created_at,
                           p.project_name
                    FROM meetings m
                    LEFT JOIN projects p ON m.project_id = p.project_id
                    WHERE m.user_id = %s
                """
                
                params = [user_id]
                
                if project_id:
                    base_query += " AND m.project_id = %s"
                    params.append(project_id)
                
                base_query += " ORDER BY m.meeting_date DESC, m.created_at DESC;"
                
                cursor.execute(base_query, params)
                rows = cursor.fetchall()
                
                meetings = []
                for row in rows:
                    meeting = Meeting(
                        meeting_id=str(row['meeting_id']),
                        user_id=user_id,
                        project_id=project_id,
                        meeting_name=row['meeting_name'],
                        meeting_date=row['meeting_date'],
                        created_at=row['created_at']
                    )
                    meetings.append(meeting)
                
                return meetings
                
        except Exception as e:
            logger.error(f"Error getting user meetings: {e}")
            return []
    
    # Session Management
    def create_session(self, user_id: str, session_id: str, expires_at: datetime) -> bool:
        """Create a new user session"""
        try:
            with self.get_cursor() as (conn, cursor):
                cursor.execute("""
                    INSERT INTO sessions (session_id, user_id, expires_at)
                    VALUES (%s, %s, %s);
                """, (session_id, user_id, expires_at))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return False
    
    def validate_session(self, session_id: str) -> Optional[str]:
        """Validate a session and return user_id if valid"""
        try:
            with self.get_cursor() as (conn, cursor):
                cursor.execute("""
                    SELECT user_id FROM sessions 
                    WHERE session_id = %s 
                    AND is_active = TRUE 
                    AND expires_at > CURRENT_TIMESTAMP;
                """, (session_id,))
                
                row = cursor.fetchone()
                return str(row[0]) if row else None
                
        except Exception as e:
            logger.error(f"Error validating session: {e}")
            return None
    
    # Statistics and Maintenance
    def get_statistics(self, user_id: str = None) -> Dict[str, Any]:
        """Get comprehensive database statistics with optional user filtering"""
        try:
            from datetime import datetime
            # Auto-detect user from Flask context if not provided
            if user_id is None:
                try:
                    from flask_login import current_user
                    if hasattr(current_user, 'user_id') and current_user.is_authenticated:
                        user_id = current_user.user_id
                        logger.info(f"Auto-detected user_id from Flask context: {user_id}")
                except:
                    pass  # No Flask context or user not authenticated
                    
            with self.get_cursor() as (conn, cursor):
                stats = {}
                
                if user_id:
                    # User-specific statistics
                    
                    # User document statistics
                    cursor.execute("SELECT COUNT(*) FROM documents WHERE user_id = %s AND is_deleted = FALSE;", (user_id,))
                    user_documents = cursor.fetchone()[0]
                    
                    # User chunk statistics
                    cursor.execute("""
                        SELECT COUNT(*) FROM document_chunks dc
                        JOIN documents d ON dc.document_id = d.document_id
                        WHERE d.user_id = %s AND d.is_deleted = FALSE;
                    """, (user_id,))
                    user_chunks = cursor.fetchone()[0]
                    
                    # User project statistics
                    cursor.execute("SELECT COUNT(*) FROM projects WHERE user_id = %s;", (user_id,))
                    user_projects = cursor.fetchone()[0]
                    
                    # User meeting statistics (if any)
                    cursor.execute("SELECT COUNT(*) FROM meetings WHERE user_id = %s;", (user_id,))
                    user_meetings = cursor.fetchone()[0]
                    
                    # User date range - extract meeting dates from filenames
                    cursor.execute("""
                        SELECT filename
                        FROM documents 
                        WHERE user_id = %s AND is_deleted = FALSE;
                    """, (user_id,))
                    documents = cursor.fetchall()
                    
                    # Extract meeting dates from filenames
                    meeting_dates = []
                    import re
                    for doc in documents:
                        filename = doc[0]
                        # Extract date pattern YYYYMMDD from filename
                        date_match = re.search(r'(\d{8})', filename)
                        if date_match:
                            date_str = date_match.group(1)
                            try:
                                meeting_date = datetime.strptime(date_str, '%Y%m%d').date()
                                meeting_dates.append(meeting_date)
                            except:
                                pass  # Skip invalid dates
                    
                    # Get earliest and latest meeting dates
                    if meeting_dates:
                        earliest_date = min(meeting_dates)
                        latest_date = max(meeting_dates)
                    else:
                        earliest_date = None
                        latest_date = None
                    
                    # Format dates for display
                    def format_date(date_obj):
                        if not date_obj:
                            return None
                        try:
                            if hasattr(date_obj, 'strftime'):
                                return date_obj.strftime('%Y-%m-%d')
                            else:
                                # Handle string timestamps
                                if 'T' in str(date_obj):
                                    # ISO format timestamp
                                    dt = datetime.fromisoformat(str(date_obj).replace('Z', '+00:00'))
                                    return dt.strftime('%Y-%m-%d')
                                else:
                                    return str(date_obj)
                        except Exception as e:
                            logger.warning(f"Error formatting date {date_obj}: {e}")
                            return str(date_obj) if date_obj else None
                    
                    formatted_earliest = format_date(earliest_date)
                    formatted_latest = format_date(latest_date)
                    
                    # Build user-specific response
                    stats = {
                        # User specific data
                        'user_documents': user_documents,
                        'user_chunks': user_chunks, 
                        'user_projects': user_projects,
                        'user_meetings': user_meetings,
                        
                        # Date range
                        'date_range': {
                            'earliest': formatted_earliest,
                            'latest': formatted_latest
                        },
                        'earliest_meeting': formatted_earliest,
                        'latest_meeting': formatted_latest,
                        
                        # System data for comparison
                        'active_users': 0,  # Will be filled below
                        'documents': 0,     # Will be filled below  
                        'chunks': 0,        # Will be filled below
                        'projects': 0,      # Will be filled below
                        'database_size': '', # Will be filled below
                        'table_sizes': {},  # Will be filled below
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    stats = {}
                
                # Always include system-wide statistics
                cursor.execute("SELECT COUNT(*) FROM users WHERE is_active = TRUE;")
                stats['active_users'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM documents WHERE is_deleted = FALSE;")
                stats['documents'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM document_chunks;")
                stats['chunks'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM projects;")
                stats['projects'] = cursor.fetchone()[0]
                
                # Database size
                cursor.execute("""
                    SELECT pg_size_pretty(pg_database_size(current_database()));
                """)
                stats['database_size'] = cursor.fetchone()[0]
                
                # Table sizes
                cursor.execute("""
                    SELECT 
                        schemaname,
                        tablename,
                        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename))
                    FROM pg_tables 
                    WHERE schemaname = 'public'
                    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
                """)
                
                table_sizes = cursor.fetchall()
                stats['table_sizes'] = {row[1]: row[2] for row in table_sizes}
                
                if not user_id:
                    stats['timestamp'] = datetime.now().isoformat()
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Alias for get_statistics for compatibility"""
        return self.get_statistics()
    
    def get_deletion_statistics(self) -> Dict[str, Any]:
        """Get deletion and cleanup statistics"""
        try:
            with self.get_cursor() as (conn, cursor):
                stats = {}
                
                # Total documents vs deleted documents
                cursor.execute("SELECT COUNT(*) FROM documents;")
                stats['total_documents'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM documents WHERE is_deleted = TRUE;")
                stats['deleted_documents'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM documents WHERE is_deleted = FALSE;")
                stats['active_documents'] = cursor.fetchone()[0]
                
                # Document chunks
                cursor.execute("SELECT COUNT(*) FROM document_chunks;")
                stats['total_chunks'] = cursor.fetchone()[0]
                
                # Orphaned chunks (chunks without documents)
                cursor.execute("""
                    SELECT COUNT(*) FROM document_chunks dc
                    LEFT JOIN documents d ON dc.document_id = d.document_id
                    WHERE d.document_id IS NULL OR d.is_deleted = TRUE;
                """)
                stats['orphaned_chunks'] = cursor.fetchone()[0]
                
                # Session cleanup stats
                cursor.execute("SELECT COUNT(*) FROM sessions;")
                stats['total_sessions'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM sessions WHERE expires_at < CURRENT_TIMESTAMP;")
                stats['expired_sessions'] = cursor.fetchone()[0]
                
                # Storage efficiency
                stats['storage_efficiency'] = {
                    'document_retention_rate': (stats['active_documents'] / max(stats['total_documents'], 1)) * 100,
                    'chunk_cleanup_needed': stats['orphaned_chunks'] > 0,
                    'session_cleanup_needed': stats['expired_sessions'] > 0
                }
                
                stats['timestamp'] = datetime.now().isoformat()
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting deletion statistics: {e}")
            return {}
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions and return count of cleaned sessions"""
        try:
            with self.get_cursor() as (conn, cursor):
                cursor.execute("""
                    DELETE FROM sessions 
                    WHERE expires_at < CURRENT_TIMESTAMP OR is_active = FALSE;
                """)
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                logger.info(f"Cleaned up {deleted_count} expired sessions")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Error cleaning up sessions: {e}")
            return 0
    
    # File Management
    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file"""
        hasher = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating file hash: {e}")
            return ""
    
    def is_file_duplicate(self, file_hash: str, filename: str, user_id: str) -> Optional[Dict]:
        """Check if file is a duplicate based on hash and return original file info"""
        try:
            with self.get_cursor(dict_cursor=True) as (conn, cursor):
                cursor.execute("""
                    SELECT d.document_id, d.filename, d.upload_date, d.file_size
                    FROM documents d
                    WHERE d.file_hash = %s 
                    AND d.user_id = %s 
                    AND d.is_deleted = FALSE
                    ORDER BY d.upload_date DESC
                    LIMIT 1;
                """, (file_hash, user_id))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'document_id': str(row['document_id']),
                        'filename': row['filename'],
                        'upload_date': row['upload_date'].isoformat() if row['upload_date'] else '',
                        'file_size': row['file_size']
                    }
                return None
                
        except Exception as e:
            logger.error(f"Error checking file duplicate: {e}")
            return None
    
    def store_file_hash(self, file_hash: str, filename: str, original_filename: str, 
                       file_size: int, user_id: str, project_id: str = None, 
                       meeting_id: str = None, document_id: str = None) -> str:
        """Store file hash information for deduplication"""
        try:
            with self.get_cursor() as (conn, cursor):
                # Generate hash_id since we're using VARCHAR instead of UUID with auto-generation
                hash_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO file_hashes 
                    (hash_id, file_hash, filename, original_filename, file_size, user_id, project_id, meeting_id, document_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
                """, (hash_id, file_hash, filename, original_filename, file_size, user_id, project_id, meeting_id, document_id))
                
                conn.commit()
                
                return str(hash_id)
                
        except Exception as e:
            logger.error(f"Error storing file hash: {e}")
            return ""

    # Job Management
    def create_upload_job(self, user_id: str, total_files: int, project_id: str = None, 
                         meeting_id: str = None) -> str:
        """Create a new upload job for tracking batch processing"""
        try:
            with self.get_cursor() as (conn, cursor):
                # Generate job_id since we're using VARCHAR instead of UUID with auto-generation
                job_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO upload_jobs (job_id, user_id, total_files, project_id, meeting_id)
                    VALUES (%s, %s, %s, %s, %s);
                """, (job_id, user_id, total_files, project_id, meeting_id))
                
                conn.commit()
                
                logger.info(f"Created upload job: {job_id}")
                return str(job_id)
                
        except Exception as e:
            logger.error(f"Error creating upload job: {e}")
            return ""
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get current job status and progress"""
        try:
            with self.get_cursor(dict_cursor=True) as (conn, cursor):
                cursor.execute("""
                    SELECT job_id, user_id, total_files, processed_files, failed_files, 
                           status, error_message, created_at, updated_at
                    FROM upload_jobs WHERE job_id = %s;
                """, (job_id,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'job_id': str(row['job_id']),
                        'user_id': str(row['user_id']),
                        'total_files': row['total_files'],
                        'processed_files': row['processed_files'],
                        'failed_files': row['failed_files'],
                        'status': row['status'],
                        'error_message': row['error_message'],
                        'created_at': row['created_at'].isoformat() if row['created_at'] else '',
                        'updated_at': row['updated_at'].isoformat() if row['updated_at'] else ''
                    }
                return None
                
        except Exception as e:
            logger.error(f"Error getting job status: {e}")
            return None
    
    def update_job_status(self, job_id: str, status: str, processed_files: int = None, 
                         failed_files: int = None, error_message: str = None):
        """Update job status and progress"""
        try:
            with self.get_cursor() as (conn, cursor):
                update_parts = ["status = %s", "updated_at = CURRENT_TIMESTAMP"]
                params = [status]
                
                if processed_files is not None:
                    update_parts.append("processed_files = %s")
                    params.append(processed_files)
                
                if failed_files is not None:
                    update_parts.append("failed_files = %s") 
                    params.append(failed_files)
                
                if error_message is not None:
                    update_parts.append("error_message = %s")
                    params.append(error_message)
                
                params.append(job_id)
                
                cursor.execute(f"""
                    UPDATE upload_jobs SET {', '.join(update_parts)}
                    WHERE job_id = %s;
                """, params)
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error updating job status: {e}")
    
    def create_file_processing_status(self, job_id: str, filename: str, file_size: int, 
                                    file_hash: str) -> str:
        """Create file processing status entry"""
        # For PostgreSQL, this could be tracked in a separate table or within upload_jobs
        # For now, return a UUID as placeholder similar to SQLite implementation
        import uuid
        return str(uuid.uuid4())
    
    def update_file_processing_status(self, status_id: str, status: str, 
                                    error_message: str = None, document_id: str = None, 
                                    chunks_created: int = None):
        """Update file processing status"""
        # For PostgreSQL, this could update a status tracking table
        # For now, pass silently similar to SQLite implementation
        pass
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get vector index statistics for PostgreSQL + pgvector"""
        try:
            with self.get_cursor() as (conn, cursor):
                # Get total number of vectors
                cursor.execute("SELECT COUNT(*) FROM document_chunks WHERE embedding IS NOT NULL;")
                total_vectors = cursor.fetchone()[0]
                
                # Get vector dimension (sample from first vector)
                cursor.execute("SELECT vector_dims(embedding) FROM document_chunks WHERE embedding IS NOT NULL LIMIT 1;")
                dimension_result = cursor.fetchone()
                dimension = dimension_result[0] if dimension_result else self.vector_dimension
                
                # Get metadata entries (total chunks with content)
                cursor.execute("SELECT COUNT(*) FROM document_chunks WHERE content IS NOT NULL;")
                metadata_entries = cursor.fetchone()[0]
                
                stats = {
                    'total_vectors': total_vectors,
                    'dimension': dimension,
                    'metadata_entries': metadata_entries,
                    'index_type': 'pgvector',
                    'database_type': 'postgresql'
                }
                
                logger.info(f"Index stats: {stats}")
                return stats
                
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {
                'total_vectors': 0,
                'dimension': self.vector_dimension,
                'metadata_entries': 0,
                'index_type': 'pgvector',
                'database_type': 'postgresql'
            }
    
    def save_index(self):
        """Save index - for PostgreSQL this is a no-op since data is persisted automatically"""
        # PostgreSQL + pgvector doesn't need explicit index saving like FAISS
        # Data is automatically persisted to disk
        logger.info("PostgreSQL index save requested - no action needed (auto-persisted)")
        pass
    
    def store_document_metadata(self, filename: str, content: str, user_id: str, 
                              project_id: str = None, meeting_id: str = None, 
                              folder_path: str = None) -> str:
        """Store document metadata and return document_id"""
        try:
            # Generate folder_path if not provided
            if folder_path is None:
                if project_id:
                    folder_path = f"user_folder/project_{project_id}"
                else:
                    folder_path = "user_folder/default_project"
            
            with self.get_cursor() as (conn, cursor):
                # Generate document_id since we're using VARCHAR instead of UUID with auto-generation
                document_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO documents (document_id, user_id, project_id, meeting_id, filename, processing_status, folder_path)
                    VALUES (%s, %s, %s, %s, %s, %s, %s);
                """, (document_id, user_id, project_id, meeting_id, filename, 'pending', folder_path))
                
                conn.commit()
                
                return str(document_id)
                
        except Exception as e:
            logger.error(f"Error storing document metadata: {e}")
            return ""
    
    def get_document_metadata(self, document_id: str) -> Dict[str, Any]:
        """Get metadata for a specific document"""
        try:
            with self.get_cursor(dict_cursor=True) as (conn, cursor):
                cursor.execute("""
                    SELECT d.*, p.project_name, m.meeting_name
                    FROM documents d
                    LEFT JOIN projects p ON d.project_id = p.project_id
                    LEFT JOIN meetings m ON d.meeting_id = m.meeting_id
                    WHERE d.document_id = %s;
                """, (document_id,))
                
                row = cursor.fetchone()
                if row:
                    return dict(row)
                return {}
                
        except Exception as e:
            logger.error(f"Error getting document metadata: {e}")
            return {}
    
    def get_project_documents(self, project_id: str, user_id: str) -> List[Dict[str, Any]]:
        """Get all documents for a specific project"""
        try:
            with self.get_cursor(dict_cursor=True) as (conn, cursor):
                cursor.execute("""
                    SELECT d.document_id, d.filename, d.content_summary, d.upload_date,
                           d.file_size, d.chunk_count, d.processing_status
                    FROM documents d
                    WHERE d.project_id = %s AND d.user_id = %s AND d.is_deleted = FALSE
                    ORDER BY d.upload_date DESC;
                """, (project_id, user_id))
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error getting project documents: {e}")
            return []

    def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a single document by ID"""
        try:
            with self.get_cursor(dict_cursor=True) as (conn, cursor):
                cursor.execute("""
                    SELECT 
                        d.document_id, d.user_id, d.project_id, d.meeting_id, d.filename, 
                        d.original_filename, d.content_summary, d.main_topics, d.participants,
                        d.key_decisions, d.action_items, d.upload_date, d.file_size, d.chunk_count,
                        p.project_name, m.meeting_name
                    FROM documents d
                    LEFT JOIN projects p ON d.project_id = p.project_id
                    LEFT JOIN meetings m ON d.meeting_id = m.meeting_id
                    WHERE d.document_id = %s AND d.is_deleted = FALSE;
                """, (document_id,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'document_id': str(row['document_id']),
                        'user_id': str(row['user_id']),
                        'project_id': str(row['project_id']) if row['project_id'] else None,
                        'meeting_id': str(row['meeting_id']) if row['meeting_id'] else None,
                        'filename': row['filename'],
                        'original_filename': row['original_filename'],
                        'content_summary': row['content_summary'],
                        'main_topics': row['main_topics'],
                        'participants': row['participants'],
                        'key_decisions': row['key_decisions'],
                        'action_items': row['action_items'],
                        'upload_date': row['upload_date'].isoformat() if row['upload_date'] else '',
                        'file_size': row['file_size'],
                        'chunk_count': row['chunk_count'],
                        'project_name': row['project_name'],
                        'meeting_name': row['meeting_name']
                    }
                return None
                
        except Exception as e:
            logger.error(f"Error getting document by ID {document_id}: {e}")
            return None

    def get_document_chunks(self, document_id: str, user_id: str = None):
        """
        Get all chunks for a specific document
        
        Args:
            document_id: Document ID
            user_id: User ID (optional, will be fetched from document if not provided)
            
        Returns:
            List of DocumentChunk objects for the document
        """
        try:
            # Get user_id from document if not provided
            if user_id is None:
                document = self.get_document_by_id(document_id)
                if not document:
                    logger.error(f"Document {document_id} not found")
                    return []
                user_id = document.get('user_id')
                if not user_id:
                    logger.error(f"User ID not found for document {document_id}")
                    return []
            
            with self.get_cursor() as (conn, cursor):
                cursor.execute("""
                    SELECT chunk_id, content, chunk_index, start_char, end_char, embedding
                    FROM document_chunks 
                    WHERE document_id = %s 
                    ORDER BY chunk_index;
                """, (document_id,))
                
                rows = cursor.fetchall()
                chunks = []
                
                for row in rows:
                    # Create DocumentChunk object using the class structure
                    chunk = DocumentChunk(
                        chunk_id=str(row[0]),
                        document_id=document_id,
                        user_id=user_id,
                        content=row[1],
                        chunk_index=row[2],
                        start_char=row[3],
                        end_char=row[4]
                    )
                    chunks.append(chunk)
                
                return chunks
                
        except Exception as e:
            logger.error(f"Error getting chunks for document {document_id}: {e}")
            return []

    def get_all_documents(self, user_id: str = None) -> List[Dict[str, Any]]:
        """Get all documents with metadata"""
        try:
            with self.get_cursor(dict_cursor=True) as (conn, cursor):
                base_query = """
                    SELECT 
                        d.document_id, d.user_id, d.project_id, d.meeting_id, d.filename, 
                        d.original_filename, d.content_summary, d.upload_date, d.file_size, 
                        d.chunk_count, d.folder_path, p.project_name, m.meeting_name
                    FROM documents d
                    LEFT JOIN projects p ON d.project_id = p.project_id
                    LEFT JOIN meetings m ON d.meeting_id = m.meeting_id
                    WHERE d.is_deleted = FALSE
                """
                
                params = []
                if user_id:
                    base_query += " AND d.user_id = %s"
                    params.append(user_id)
                
                base_query += " ORDER BY d.upload_date DESC;"
                
                cursor.execute(base_query, params)
                rows = cursor.fetchall()
                
                documents = []
                for row in rows:
                    doc = {
                        'document_id': str(row['document_id']),
                        'user_id': str(row['user_id']),
                        'project_id': str(row['project_id']) if row['project_id'] else None,
                        'meeting_id': str(row['meeting_id']) if row['meeting_id'] else None,
                        'filename': row['filename'],
                        'original_filename': row['original_filename'],
                        'content_summary': row['content_summary'],
                        'upload_date': row['upload_date'].isoformat() if row['upload_date'] else '',
                        'file_size': row['file_size'],
                        'chunk_count': row['chunk_count'],
                        'folder_path': row['folder_path'],
                        'project_name': row['project_name'],
                        'meeting_name': row['meeting_name']
                    }
                    documents.append(doc)
                
                return documents
                
        except Exception as e:
            logger.error(f"Error getting documents: {e}")
            return []

    # ===============================================
    # FOLDER-BASED SEARCH METHODS (for # functionality)
    # ===============================================
    
    def search_similar_chunks_by_folder(self, query_embedding: np.ndarray, user_id: str, 
                                      folder_path: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Search for similar chunks using pgvector, filtered by folder path
        
        Args:
            query_embedding: Query vector
            user_id: User ID for filtering
            folder_path: Folder path for filtering
            top_k: Number of top results to return
            
        Returns:
            List of (chunk_id, similarity_score) tuples
        """
        try:
            # Convert query embedding to proper format for pgvector
            query_vector = query_embedding.tolist()
            
            with self.get_cursor() as (conn, cursor):
                # Search for similar chunks in documents with matching folder_path
                cursor.execute("""
                    SELECT 
                        dc.chunk_id,
                        1 - (dc.embedding <=> %s::vector) as similarity
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.document_id
                    WHERE d.user_id = %s 
                      AND d.folder_path = %s 
                      AND d.is_deleted = FALSE
                    ORDER BY dc.embedding <=> %s::vector
                    LIMIT %s;
                """, (query_vector, user_id, folder_path, query_vector, top_k))
                
                results = cursor.fetchall()
                return [(str(row[0]), float(row[1])) for row in results]
                
        except Exception as e:
            logger.error(f"Error searching similar chunks by folder: {e}")
            return []
    
    def keyword_search_chunks_by_folder(self, keywords: List[str], user_id: str, 
                                      folder_path: str, top_k: int = 20) -> List[str]:
        """
        Search for chunks containing keywords, filtered by folder path
        
        Args:
            keywords: List of keywords to search for
            user_id: User ID for filtering
            folder_path: Folder path for filtering
            top_k: Number of top results to return
            
        Returns:
            List of chunk_ids
        """
        try:
            if not keywords:
                return []
            
            # Create search query for full-text search
            search_terms = ' & '.join(keywords)
            
            with self.get_cursor() as (conn, cursor):
                # Search for chunks containing keywords in documents with matching folder_path
                cursor.execute("""
                    SELECT 
                        dc.chunk_id,
                        ts_rank(to_tsvector('english', dc.content), plainto_tsquery('english', %s)) as rank
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.document_id
                    WHERE d.user_id = %s 
                      AND d.folder_path = %s 
                      AND d.is_deleted = FALSE
                      AND to_tsvector('english', dc.content) @@ plainto_tsquery('english', %s)
                    ORDER BY rank DESC
                    LIMIT %s;
                """, (search_terms, user_id, folder_path, search_terms, top_k))
                
                results = cursor.fetchall()
                return [str(row[0]) for row in results]
                
        except Exception as e:
            logger.error(f"Error keyword searching chunks by folder: {e}")
            return []
    
    def get_user_documents_by_folder(self, user_id: str, folder_path: str) -> List[str]:
        """
        Get all document IDs for a user in a specific folder path
        
        Args:
            user_id: User ID for filtering
            folder_path: Folder path for filtering
            
        Returns:
            List of document_ids
        """
        try:
            with self.get_cursor() as (conn, cursor):
                cursor.execute("""
                    SELECT document_id
                    FROM documents
                    WHERE user_id = %s 
                      AND folder_path = %s 
                      AND is_deleted = FALSE
                    ORDER BY upload_date DESC;
                """, (user_id, folder_path))
                
                results = cursor.fetchall()
                return [str(row[0]) for row in results]
                
        except Exception as e:
            logger.error(f"Error getting user documents by folder: {e}")
            return []

    def get_documents_by_timeframe(self, timeframe: str, user_id: str = None) -> List[Dict[str, Any]]:
        """Get documents filtered by intelligent timeframe calculation"""
        try:
            start_date, end_date = self._calculate_date_range(timeframe)
            
            with self.get_cursor(dict_cursor=True) as (conn, cursor):
                query_params = []
                base_query = """
                    SELECT document_id, filename, upload_date, content_summary, 
                           main_topics, participants, user_id, meeting_id, project_id, folder_path
                    FROM documents
                    WHERE is_deleted = FALSE
                """
                
                if user_id:
                    base_query += " AND user_id = %s"
                    query_params.append(user_id)
                
                if start_date:
                    base_query += " AND upload_date >= %s"
                    query_params.append(start_date)
                
                if end_date:
                    base_query += " AND upload_date <= %s"
                    query_params.append(end_date)
                
                base_query += " ORDER BY upload_date DESC"
                
                cursor.execute(base_query, tuple(query_params))
                rows = cursor.fetchall()
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error getting documents by timeframe: {e}")
            return []
    
    def _calculate_date_range(self, timeframe: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Calculate start and end dates for intelligent timeframe filtering"""
        import calendar
        
        now = datetime.now()
        today = now.date()
        
        # Get current week boundaries (Monday to Sunday)
        current_week_start = today - timedelta(days=today.weekday())
        current_week_end = current_week_start + timedelta(days=6)
        
        # Get last week boundaries
        last_week_start = current_week_start - timedelta(days=7)
        last_week_end = current_week_start - timedelta(days=1)
        
        # Get current month boundaries
        current_month_start = today.replace(day=1)
        _, last_day = calendar.monthrange(today.year, today.month)
        current_month_end = today.replace(day=last_day)
        
        # Get last month boundaries
        if today.month == 1:
            last_month_year = today.year - 1
            last_month = 12
        else:
            last_month_year = today.year
            last_month = today.month - 1
        
        last_month_start = today.replace(year=last_month_year, month=last_month, day=1)
        _, last_month_last_day = calendar.monthrange(last_month_year, last_month)
        last_month_end = today.replace(year=last_month_year, month=last_month, day=last_month_last_day)
        
        # Convert dates to datetime objects for comparison
        def to_datetime(date_obj):
            return datetime.combine(date_obj, datetime.min.time())
        
        # Map timeframes to date ranges
        timeframe_map = {
            'current_week': (to_datetime(current_week_start), to_datetime(current_week_end)),
            'last_week': (to_datetime(last_week_start), to_datetime(last_week_end)),
            'current_month': (to_datetime(current_month_start), to_datetime(current_month_end)),
            'last_month': (to_datetime(last_month_start), to_datetime(last_month_end)),
            'current_year': (to_datetime(today.replace(month=1, day=1)), to_datetime(today.replace(month=12, day=31))),
            'last_year': (to_datetime(today.replace(year=today.year-1, month=1, day=1)), 
                         to_datetime(today.replace(year=today.year-1, month=12, day=31))),
            'last_7_days': (to_datetime(today - timedelta(days=7)), to_datetime(today)),
            'last_14_days': (to_datetime(today - timedelta(days=14)), to_datetime(today)),
            'last_30_days': (to_datetime(today - timedelta(days=30)), to_datetime(today)),
            'last_60_days': (to_datetime(today - timedelta(days=60)), to_datetime(today)),
            'last_90_days': (to_datetime(today - timedelta(days=90)), to_datetime(today)),
            'last_3_months': (to_datetime(today - timedelta(days=90)), to_datetime(today)),
            'last_6_months': (to_datetime(today - timedelta(days=180)), to_datetime(today)),
            'last_12_months': (to_datetime(today - timedelta(days=365)), to_datetime(today)),
            'recent': (to_datetime(today - timedelta(days=30)), to_datetime(today))
        }
        
        return timeframe_map.get(timeframe, (None, None))