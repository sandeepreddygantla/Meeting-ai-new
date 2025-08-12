#!/usr/bin/env python3
"""
Fix PostgreSQL schema to use VARCHAR(255) instead of UUID for all ID fields.
This resolves compatibility issues with the existing string-based ID system.
"""

import psycopg2

def fix_schema():
    """Drop and recreate the database schema with VARCHAR IDs"""
    
    # Connection config
    db_config = {
        'host': 'localhost',
        'database': 'meetingsai', 
        'user': 'postgres',
        'password': 'Sandeep@0904',
        'port': 5432
    }
    
    try:
        # Connect and reset schema
        conn = psycopg2.connect(**db_config)
        conn.autocommit = True
        cursor = conn.cursor()
        
        print("Resetting database schema...")
        
        # Drop and recreate schema
        cursor.execute('DROP SCHEMA public CASCADE;')
        cursor.execute('CREATE SCHEMA public;')
        cursor.execute('GRANT ALL ON SCHEMA public TO postgres;')
        cursor.execute('GRANT ALL ON SCHEMA public TO public;')
        
        # Enable pgvector extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        print("Creating tables with corrected schema...")
        
        # Users table
        cursor.execute("""
            CREATE TABLE users (
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
            CREATE TABLE projects (
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
            CREATE TABLE meetings (
                meeting_id VARCHAR(255) PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                project_id VARCHAR(255) REFERENCES projects(project_id) ON DELETE SET NULL,
                meeting_name VARCHAR(255) NOT NULL,
                meeting_date TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Documents table
        cursor.execute("""
            CREATE TABLE documents (
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
                
                -- Soft deletion support
                is_deleted BOOLEAN DEFAULT FALSE,
                deleted_at TIMESTAMP,
                deleted_by VARCHAR(255) REFERENCES users(user_id)
            );
        """)
        
        # Document chunks table with pgvector support
        cursor.execute("""
            CREATE TABLE document_chunks (
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
                
                -- Vector embedding (pgvector) - 1536 dimensions for text-embedding-3-small
                embedding VECTOR(1536),
                
                -- Timestamps
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                -- Indexing for performance
                UNIQUE(document_id, chunk_index)
            );
        """)
        
        # Sessions table
        cursor.execute("""
            CREATE TABLE sessions (
                session_id VARCHAR(255) PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                is_active BOOLEAN DEFAULT TRUE
            );
        """)
        
        # File hashes table for deduplication
        cursor.execute("""
            CREATE TABLE file_hashes (
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
            CREATE TABLE upload_jobs (
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
        
        print("Creating indexes...")
        
        # Create indexes for performance
        indexes = [
            "CREATE INDEX idx_docs_user_id ON documents(user_id);",
            "CREATE INDEX idx_docs_project_id ON documents(project_id);", 
            "CREATE INDEX idx_docs_upload_date ON documents(upload_date);",
            "CREATE INDEX idx_docs_file_hash ON documents(file_hash);",
            "CREATE INDEX idx_docs_is_deleted ON documents(is_deleted);",
            "CREATE INDEX idx_chunks_document_id ON document_chunks(document_id);",
            "CREATE INDEX idx_chunks_user_id ON document_chunks(user_id);",
            "CREATE INDEX idx_chunks_project_id ON document_chunks(project_id);",
            "CREATE INDEX idx_sessions_user_id ON sessions(user_id);",
            "CREATE INDEX idx_sessions_expires_at ON sessions(expires_at);",
            # Vector index for similarity search
            "CREATE INDEX idx_chunks_embedding ON document_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);"
        ]
        
        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
            except Exception as e:
                print(f"Index creation warning: {e}")
                # Continue with other indexes
        
        cursor.close()
        conn.close()
        
        print("✅ PostgreSQL schema fixed successfully!")
        print("- All ID fields now use VARCHAR(255)")
        print("- Vector dimension set to 1536 for text-embedding-3-small")
        print("- All tables and indexes created")
        
    except Exception as e:
        print(f"❌ Error fixing schema: {e}")
        return False
        
    return True

if __name__ == "__main__":
    fix_schema()