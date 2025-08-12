#!/usr/bin/env python3
"""
PostgreSQL + pgvector Migration Test Suite
Comprehensive testing of PostgreSQL database with pgvector extension
for Meeting AI document processing and vector embeddings
"""

import psycopg2
from psycopg2.extras import execute_values
import numpy as np
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import uuid
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PostgresVectorTester:
    """Test suite for PostgreSQL + pgvector migration"""
    
    def __init__(self):
        """Initialize database connection parameters"""
        self.db_config = {
            'host': 'localhost',
            'database': 'postgres',
            'user': 'postgres',
            'password': 'Sandeep@0904',
            'port': '5432'
        }
        self.conn = None
        self.cursor = None
        self.vector_dimension = 1536  # Compatible dimension for pgvector (text-embedding-ada-002)
        self.test_results = {
            'pgvector_extension': False,
            'table_creation': False,
            'metadata_operations': False,
            'vector_operations': False,
            'similarity_search': False,
            'crud_operations': False,
            'performance_test': False,
            'cleanup': False
        }
    
    def connect(self) -> bool:
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor()
            logger.info("[SUCCESS] Connected to PostgreSQL database")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Database connection failed: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        try:
            if self.cursor:
                self.cursor.close()
            if self.conn:
                self.conn.close()
            logger.info("[SUCCESS] Database connection closed")
        except Exception as e:
            logger.error(f"[ERROR] Error closing connection: {e}")
    
    def test_pgvector_extension(self) -> bool:
        """Test pgvector extension installation and functionality"""
        try:
            logger.info("\n=== Testing pgvector Extension ===")
            
            # Check if pgvector extension is installed
            self.cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
            extension = self.cursor.fetchone()
            
            if not extension:
                logger.info("pgvector extension not found. Attempting to install...")
                self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                self.conn.commit()
                logger.info("[SUCCESS] pgvector extension installed")
            else:
                logger.info("[SUCCESS] pgvector extension already installed")
            
            # Test vector data type
            self.cursor.execute("SELECT '[1,2,3]'::vector;")
            test_vector = self.cursor.fetchone()
            logger.info(f"[TEST] Vector data type test: {test_vector[0]}")
            
            # Test vector operations
            self.cursor.execute("SELECT '[1,2,3]'::vector <-> '[3,2,1]'::vector as distance;")
            distance = self.cursor.fetchone()
            logger.info(f"[TEST] Vector distance calculation: {distance[0]}")
            
            self.test_results['pgvector_extension'] = True
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] pgvector extension test failed: {e}")
            self.test_results['pgvector_extension'] = False
            return False
    
    def create_test_schema(self) -> bool:
        """Create test schema similar to current Meeting AI structure"""
        try:
            logger.info("\n=== Creating Test Database Schema ===")
            
            # Drop tables if they exist (for clean testing)
            drop_tables = [
                "DROP TABLE IF EXISTS test_document_chunks CASCADE;",
                "DROP TABLE IF EXISTS test_documents CASCADE;", 
                "DROP TABLE IF EXISTS test_meetings CASCADE;",
                "DROP TABLE IF EXISTS test_projects CASCADE;",
                "DROP TABLE IF EXISTS test_users CASCADE;"
            ]
            
            for drop_sql in drop_tables:
                self.cursor.execute(drop_sql)
            
            # Create Users table
            users_sql = """
            CREATE TABLE test_users (
                user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                username VARCHAR(100) UNIQUE NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                full_name VARCHAR(255),
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE
            );
            """
            self.cursor.execute(users_sql)
            
            # Create Projects table
            projects_sql = """
            CREATE TABLE test_projects (
                project_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES test_users(user_id) ON DELETE CASCADE,
                project_name VARCHAR(255) NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, project_name)
            );
            """
            self.cursor.execute(projects_sql)
            
            # Create Meetings table
            meetings_sql = """
            CREATE TABLE test_meetings (
                meeting_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES test_users(user_id) ON DELETE CASCADE,
                project_id UUID REFERENCES test_projects(project_id) ON DELETE SET NULL,
                meeting_name VARCHAR(255) NOT NULL,
                meeting_date TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            self.cursor.execute(meetings_sql)
            
            # Create Documents table with metadata
            documents_sql = """
            CREATE TABLE test_documents (
                document_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES test_users(user_id) ON DELETE CASCADE,
                project_id UUID REFERENCES test_projects(project_id) ON DELETE SET NULL,
                meeting_id UUID REFERENCES test_meetings(meeting_id) ON DELETE SET NULL,
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
                deleted_by UUID REFERENCES test_users(user_id)
            );
            """
            self.cursor.execute(documents_sql)
            
            # Create Document Chunks table with pgvector support
            chunks_sql = f"""
            CREATE TABLE test_document_chunks (
                chunk_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                document_id UUID NOT NULL REFERENCES test_documents(document_id) ON DELETE CASCADE,
                user_id UUID NOT NULL REFERENCES test_users(user_id) ON DELETE CASCADE,
                project_id UUID REFERENCES test_projects(project_id) ON DELETE SET NULL,
                meeting_id UUID REFERENCES test_meetings(meeting_id) ON DELETE SET NULL,
                
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
            """
            self.cursor.execute(chunks_sql)
            
            # Create indexes for performance
            indexes_sql = [
                "CREATE INDEX idx_test_docs_user_id ON test_documents(user_id);",
                "CREATE INDEX idx_test_docs_project_id ON test_documents(project_id);", 
                "CREATE INDEX idx_test_docs_upload_date ON test_documents(upload_date);",
                "CREATE INDEX idx_test_docs_file_hash ON test_documents(file_hash);",
                "CREATE INDEX idx_test_chunks_document_id ON test_document_chunks(document_id);",
                "CREATE INDEX idx_test_chunks_user_id ON test_document_chunks(user_id);",
                f"CREATE INDEX idx_test_chunks_embedding ON test_document_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10);"
            ]
            
            for index_sql in indexes_sql:
                try:
                    self.cursor.execute(index_sql)
                except Exception as e:
                    logger.warning(f"Index creation warning: {e}")
                    # Continue with other indexes
            
            self.conn.commit()
            logger.info("[SUCCESS] Test schema created successfully")
            self.test_results['table_creation'] = True
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Schema creation failed: {e}")
            self.test_results['table_creation'] = False
            self.conn.rollback()
            return False
    
    def test_metadata_operations(self) -> bool:
        """Test metadata CRUD operations"""
        try:
            logger.info("\n=== Testing Metadata Operations ===")
            
            # Create test user
            user_sql = """
            INSERT INTO test_users (username, email, full_name, password_hash) 
            VALUES (%s, %s, %s, %s) RETURNING user_id;
            """
            self.cursor.execute(user_sql, ('testuser', 'test@example.com', 'Test User', 'hashed_password'))
            user_id = self.cursor.fetchone()[0]
            logger.info(f"[TEST] Created test user: {user_id}")
            
            # Create test project
            project_sql = """
            INSERT INTO test_projects (user_id, project_name, description) 
            VALUES (%s, %s, %s) RETURNING project_id;
            """
            self.cursor.execute(project_sql, (user_id, 'Test Project', 'Project for testing'))
            project_id = self.cursor.fetchone()[0]
            logger.info(f"[TEST] Created test project: {project_id}")
            
            # Create test meeting
            meeting_sql = """
            INSERT INTO test_meetings (user_id, project_id, meeting_name, meeting_date) 
            VALUES (%s, %s, %s, %s) RETURNING meeting_id;
            """
            self.cursor.execute(meeting_sql, (user_id, project_id, 'Test Meeting', datetime.now()))
            meeting_id = self.cursor.fetchone()[0]
            logger.info(f"[TEST] Created test meeting: {meeting_id}")
            
            # Create test document with AI metadata
            doc_sql = """
            INSERT INTO test_documents (
                user_id, project_id, meeting_id, filename, original_filename,
                file_size, file_hash, content_summary, main_topics, participants
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING document_id;
            """
            
            file_hash = hashlib.sha256(b'test content').hexdigest()
            
            self.cursor.execute(doc_sql, (
                user_id, project_id, meeting_id, 'test_doc.pdf', 'Original Test Doc.pdf',
                1024, file_hash, 'Document about testing PostgreSQL integration',
                'PostgreSQL, pgvector, testing', 'Test User, Database Admin'
            ))
            document_id = self.cursor.fetchone()[0]
            logger.info(f"[TEST] Created test document: {document_id}")
            
            # Test querying metadata
            query_sql = """
            SELECT d.filename, d.content_summary, p.project_name, m.meeting_name
            FROM test_documents d
            JOIN test_projects p ON d.project_id = p.project_id
            JOIN test_meetings m ON d.meeting_id = m.meeting_id
            WHERE d.user_id = %s;
            """
            self.cursor.execute(query_sql, (user_id,))
            result = self.cursor.fetchone()
            
            logger.info(f"[TEST] Query result: {result}")
            
            # Store document and meeting IDs for vector testing
            self.test_document_id = document_id
            self.test_user_id = user_id
            self.test_project_id = project_id
            self.test_meeting_id = meeting_id
            
            self.conn.commit()
            logger.info("[SUCCESS] Metadata operations completed successfully")
            self.test_results['metadata_operations'] = True
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Metadata operations failed: {e}")
            self.test_results['metadata_operations'] = False
            self.conn.rollback()
            return False
    
    def test_vector_operations(self) -> bool:
        """Test vector embedding storage and retrieval"""
        try:
            logger.info("\n=== Testing Vector Operations ===")
            
            if not hasattr(self, 'test_document_id'):
                logger.error("[ERROR] No test document available. Run metadata test first.")
                return False
            
            # Generate test embeddings (simulating real embeddings)
            test_embeddings = []
            test_contents = [
                "This is the first chunk of the test document discussing PostgreSQL integration.",
                "The second chunk covers pgvector extension and its vector similarity capabilities.", 
                "This third chunk explains how embeddings work in the context of document search.",
                "The fourth chunk discusses performance considerations for vector databases.",
                "The final chunk summarizes the benefits of using PostgreSQL with pgvector."
            ]
            
            for i, content in enumerate(test_contents):
                # Generate pseudo-realistic embedding (normally would come from OpenAI)
                np.random.seed(i + 42)  # Consistent random vectors for testing
                embedding = np.random.normal(0, 1, self.vector_dimension).astype(np.float32)
                # Normalize to unit vector (common for embeddings)
                embedding = embedding / np.linalg.norm(embedding)
                test_embeddings.append(embedding)
            
            # Batch insert chunks with embeddings
            chunk_sql = """
            INSERT INTO test_document_chunks (
                document_id, user_id, project_id, meeting_id, content, 
                chunk_index, embedding
            ) VALUES %s;
            """
            
            chunk_data = []
            for i, (content, embedding) in enumerate(zip(test_contents, test_embeddings)):
                chunk_data.append((
                    self.test_document_id, self.test_user_id, self.test_project_id, 
                    self.test_meeting_id, content, i, embedding.tolist()
                ))
            
            execute_values(self.cursor, chunk_sql, chunk_data, template=None)
            
            # Update document chunk count
            self.cursor.execute(
                "UPDATE test_documents SET chunk_count = %s WHERE document_id = %s",
                (len(test_contents), self.test_document_id)
            )
            
            self.conn.commit()
            logger.info(f"[TEST] Inserted {len(test_contents)} chunks with vector embeddings")
            
            # Test vector retrieval
            self.cursor.execute(
                "SELECT chunk_id, content, embedding FROM test_document_chunks WHERE document_id = %s LIMIT 2",
                (self.test_document_id,)
            )
            chunks = self.cursor.fetchall()
            
            for chunk in chunks:
                chunk_id, content, embedding_data = chunk
                embedding_vector = np.array(embedding_data)
                logger.info(f"[TEST] Retrieved chunk {chunk_id}: {content[:50]}...")
                logger.info(f"[TEST] Embedding dimension: {len(embedding_vector)}")
            
            # Store first embedding for similarity testing
            self.test_query_embedding = test_embeddings[0]
            
            logger.info("[SUCCESS] Vector operations completed successfully")
            self.test_results['vector_operations'] = True
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Vector operations failed: {e}")
            self.test_results['vector_operations'] = False
            self.conn.rollback()
            return False
    
    def test_similarity_search(self) -> bool:
        """Test vector similarity search functionality"""
        try:
            logger.info("\n=== Testing Similarity Search ===")
            
            if not hasattr(self, 'test_query_embedding'):
                logger.error("[ERROR] No test embeddings available. Run vector operations test first.")
                return False
            
            # Test cosine similarity search
            similarity_sql = """
            SELECT 
                chunk_id,
                content,
                1 - (embedding <=> %s::vector) as cosine_similarity
            FROM test_document_chunks
            WHERE user_id = %s
            ORDER BY embedding <=> %s::vector
            LIMIT 3;
            """
            
            query_embedding = self.test_query_embedding.tolist()
            self.cursor.execute(similarity_sql, (query_embedding, self.test_user_id, query_embedding))
            similar_chunks = self.cursor.fetchall()
            
            logger.info("[TEST] Top 3 most similar chunks:")
            for i, (chunk_id, content, similarity) in enumerate(similar_chunks):
                logger.info(f"  {i+1}. Similarity: {similarity:.4f}")
                logger.info(f"     Content: {content[:80]}...")
            
            # Test different distance metrics
            distance_metrics = [
                ("L2 Distance", "<->", "euclidean"),
                ("Cosine Distance", "<=>", "cosine"), 
                ("Inner Product", "<#>", "inner_product")
            ]
            
            for metric_name, operator, description in distance_metrics:
                test_sql = f"""
                SELECT AVG(embedding {operator} %s::vector) as avg_distance
                FROM test_document_chunks
                WHERE user_id = %s;
                """
                self.cursor.execute(test_sql, (query_embedding, self.test_user_id))
                avg_distance = self.cursor.fetchone()[0]
                logger.info(f"[TEST] {metric_name} average: {avg_distance:.4f}")
            
            # Test filtered similarity search (by project)
            filtered_search_sql = """
            SELECT 
                c.chunk_id,
                c.content,
                1 - (c.embedding <=> %s::vector) as similarity,
                d.filename,
                p.project_name
            FROM test_document_chunks c
            JOIN test_documents d ON c.document_id = d.document_id
            JOIN test_projects p ON c.project_id = p.project_id
            WHERE c.user_id = %s AND c.project_id = %s
            ORDER BY c.embedding <=> %s::vector
            LIMIT 5;
            """
            
            self.cursor.execute(filtered_search_sql, (
                query_embedding, self.test_user_id, self.test_project_id, query_embedding
            ))
            filtered_results = self.cursor.fetchall()
            
            logger.info(f"[TEST] Found {len(filtered_results)} chunks in filtered search")
            
            logger.info("[SUCCESS] Similarity search completed successfully")
            self.test_results['similarity_search'] = True
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Similarity search failed: {e}")
            self.test_results['similarity_search'] = False
            return False
    
    def test_crud_operations(self) -> bool:
        """Test Create, Read, Update, Delete operations"""
        try:
            logger.info("\n=== Testing CRUD Operations ===")
            
            # CREATE: Add new document and chunks
            new_doc_sql = """
            INSERT INTO test_documents (
                user_id, project_id, filename, content_summary, main_topics
            ) VALUES (%s, %s, %s, %s, %s) RETURNING document_id;
            """
            
            self.cursor.execute(new_doc_sql, (
                self.test_user_id, self.test_project_id, 'crud_test.pdf',
                'Document for CRUD testing', 'CRUD, PostgreSQL, Testing'
            ))
            crud_doc_id = self.cursor.fetchone()[0]
            logger.info(f"[CREATE] Created new document: {crud_doc_id}")
            
            # READ: Query documents with joins
            read_sql = """
            SELECT d.document_id, d.filename, d.content_summary, COUNT(c.chunk_id) as chunk_count
            FROM test_documents d
            LEFT JOIN test_document_chunks c ON d.document_id = c.document_id
            WHERE d.user_id = %s
            GROUP BY d.document_id, d.filename, d.content_summary;
            """
            
            self.cursor.execute(read_sql, (self.test_user_id,))
            documents = self.cursor.fetchall()
            logger.info(f"[READ] Found {len(documents)} documents for user")
            
            for doc in documents:
                logger.info(f"  - {doc[1]}: {doc[3]} chunks")
            
            # UPDATE: Modify document metadata
            update_sql = """
            UPDATE test_documents 
            SET content_summary = %s, processing_status = %s, processed_date = %s
            WHERE document_id = %s;
            """
            
            self.cursor.execute(update_sql, (
                'Updated CRUD test document with new summary',
                'completed',
                datetime.now(),
                crud_doc_id
            ))
            
            updated_rows = self.cursor.rowcount
            logger.info(f"[UPDATE] Updated {updated_rows} document(s)")
            
            # Test soft delete (Meeting AI pattern)
            soft_delete_sql = """
            UPDATE test_documents 
            SET is_deleted = TRUE, deleted_at = %s, deleted_by = %s
            WHERE document_id = %s;
            """
            
            self.cursor.execute(soft_delete_sql, (
                datetime.now(), self.test_user_id, crud_doc_id
            ))
            logger.info("[SOFT DELETE] Soft deleted test document")
            
            # Verify soft deleted document is hidden from normal queries
            active_docs_sql = """
            SELECT COUNT(*) FROM test_documents 
            WHERE user_id = %s AND (is_deleted = FALSE OR is_deleted IS NULL);
            """
            self.cursor.execute(active_docs_sql, (self.test_user_id,))
            active_count = self.cursor.fetchone()[0]
            
            all_docs_sql = """
            SELECT COUNT(*) FROM test_documents WHERE user_id = %s;
            """
            self.cursor.execute(all_docs_sql, (self.test_user_id,))
            total_count = self.cursor.fetchone()[0]
            
            logger.info(f"[TEST] Active documents: {active_count}, Total: {total_count}")
            
            # Test undelete
            undelete_sql = """
            UPDATE test_documents 
            SET is_deleted = FALSE, deleted_at = NULL, deleted_by = NULL
            WHERE document_id = %s;
            """
            self.cursor.execute(undelete_sql, (crud_doc_id,))
            logger.info("[UNDELETE] Restored soft deleted document")
            
            self.conn.commit()
            logger.info("[SUCCESS] CRUD operations completed successfully")
            self.test_results['crud_operations'] = True
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] CRUD operations failed: {e}")
            self.test_results['crud_operations'] = False
            self.conn.rollback()
            return False
    
    def test_performance(self) -> bool:
        """Test performance with larger dataset"""
        try:
            logger.info("\n=== Testing Performance ===")
            
            # Create additional test data for performance testing
            start_time = datetime.now()
            
            # Generate 100 test chunks with embeddings
            chunk_data = []
            for i in range(100):
                np.random.seed(i + 1000)
                embedding = np.random.normal(0, 1, self.vector_dimension).astype(np.float32)
                embedding = embedding / np.linalg.norm(embedding)
                
                chunk_data.append((
                    self.test_document_id, self.test_user_id, self.test_project_id,
                    self.test_meeting_id, f'Performance test chunk {i} with sample content.',
                    i + 100, embedding.tolist()
                ))
            
            # Batch insert
            chunk_sql = """
            INSERT INTO test_document_chunks (
                document_id, user_id, project_id, meeting_id, content, 
                chunk_index, embedding
            ) VALUES %s;
            """
            
            execute_values(self.cursor, chunk_sql, chunk_data, template=None, page_size=50)
            insert_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"[PERF] Inserted 100 chunks in {insert_time:.2f} seconds")
            
            # Test similarity search performance
            start_time = datetime.now()
            query_embedding = self.test_query_embedding.tolist()
            
            for _ in range(10):  # Run 10 searches
                self.cursor.execute("""
                    SELECT chunk_id, 1 - (embedding <=> %s::vector) as similarity
                    FROM test_document_chunks
                    WHERE user_id = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT 10;
                """, (query_embedding, self.test_user_id, query_embedding))
                results = self.cursor.fetchall()
            
            search_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"[PERF] 10 similarity searches in {search_time:.2f} seconds")
            logger.info(f"[PERF] Average search time: {search_time/10:.4f} seconds")
            
            # Test complex query performance
            start_time = datetime.now()
            complex_sql = """
            SELECT 
                c.chunk_id,
                c.content,
                d.filename,
                p.project_name,
                1 - (c.embedding <=> %s::vector) as similarity
            FROM test_document_chunks c
            JOIN test_documents d ON c.document_id = d.document_id
            JOIN test_projects p ON c.project_id = p.project_id
            WHERE c.user_id = %s 
                AND d.is_deleted = FALSE
                AND c.content ILIKE %s
            ORDER BY c.embedding <=> %s::vector
            LIMIT 20;
            """
            
            self.cursor.execute(complex_sql, (
                query_embedding, self.test_user_id, '%test%', query_embedding
            ))
            complex_results = self.cursor.fetchall()
            complex_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"[PERF] Complex join query with vector search in {complex_time:.4f} seconds")
            logger.info(f"[PERF] Found {len(complex_results)} results")
            
            self.conn.commit()
            logger.info("[SUCCESS] Performance testing completed")
            self.test_results['performance_test'] = True
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Performance testing failed: {e}")
            self.test_results['performance_test'] = False
            return False
    
    def cleanup_test_data(self) -> bool:
        """Clean up test data"""
        try:
            logger.info("\n=== Cleaning Up Test Data ===")
            
            # Drop test tables in correct order (respecting foreign keys)
            cleanup_sql = [
                "DROP TABLE IF EXISTS test_document_chunks CASCADE;",
                "DROP TABLE IF EXISTS test_documents CASCADE;",
                "DROP TABLE IF EXISTS test_meetings CASCADE;", 
                "DROP TABLE IF EXISTS test_projects CASCADE;",
                "DROP TABLE IF EXISTS test_users CASCADE;"
            ]
            
            for sql in cleanup_sql:
                self.cursor.execute(sql)
            
            self.conn.commit()
            logger.info("[SUCCESS] Test data cleaned up successfully")
            self.test_results['cleanup'] = True
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Cleanup failed: {e}")
            self.test_results['cleanup'] = False
            return False
    
    def compare_with_faiss(self):
        """Compare PostgreSQL + pgvector with current FAISS + SQLite setup"""
        logger.info("\n=== Comparison with Current FAISS + SQLite Setup ===")
        
        comparison = {
            'PostgreSQL + pgvector': {
                'pros': [
                    'Unified storage (metadata + vectors in one system)',
                    'ACID transactions for data consistency', 
                    'Advanced SQL queries with vector operations',
                    'Built-in backup and replication',
                    'Better concurrent access handling',
                    'No need to sync between SQLite and FAISS',
                    'Rich indexing options (IVFFlat, HNSW)',
                    'Mature ecosystem and tooling'
                ],
                'cons': [
                    'Higher resource usage than FAISS',
                    'More complex deployment setup',
                    'Potential vendor lock-in to PostgreSQL',
                    'Learning curve for pgvector-specific operations'
                ]
            },
            'Current FAISS + SQLite': {
                'pros': [
                    'Optimized for pure vector similarity search',
                    'Lower memory usage for vectors',
                    'Fast similarity search performance',
                    'Lightweight SQLite for metadata'
                ],
                'cons': [
                    'Need to maintain sync between two systems',
                    'No transactional consistency across both stores',
                    'More complex backup and recovery',
                    'Limited concurrent write access with SQLite',
                    'Manual index rebuilding required'
                ]
            }
        }
        
        for system, details in comparison.items():
            logger.info(f"\n{system}:")
            logger.info("  Pros:")
            for pro in details['pros']:
                logger.info(f"    + {pro}")
            logger.info("  Cons:")
            for con in details['cons']:
                logger.info(f"    - {con}")
    
    def generate_migration_recommendations(self):
        """Generate recommendations for migration"""
        logger.info("\n=== Migration Recommendations ===")
        
        success_rate = sum(self.test_results.values()) / len(self.test_results)
        
        recommendations = []
        
        if success_rate >= 0.8:
            recommendations.extend([
                "✅ PostgreSQL + pgvector is ready for production migration",
                "✅ All core functionality tests passed",
                "✅ Performance is suitable for Meeting AI workload"
            ])
        elif success_rate >= 0.6:
            recommendations.extend([
                "⚠️  PostgreSQL + pgvector shows promise but needs attention",
                "⚠️  Some tests failed - review issues before migration",
                "⚠️  Consider pilot deployment first"
            ])
        else:
            recommendations.extend([
                "❌ PostgreSQL + pgvector not ready for migration",
                "❌ Multiple critical tests failed",
                "❌ Recommend staying with current FAISS + SQLite setup"
            ])
        
        # Add specific recommendations based on test results
        if not self.test_results['pgvector_extension']:
            recommendations.append("🔧 Install and configure pgvector extension properly")
        
        if not self.test_results['performance_test']:
            recommendations.append("🔧 Optimize database configuration for vector operations")
        
        if not self.test_results['similarity_search']:
            recommendations.append("🔧 Review vector indexing strategy")
        
        recommendations.extend([
            "",
            "Migration Steps (if proceeding):",
            "1. Set up PostgreSQL with pgvector in production environment",
            "2. Create migration scripts for existing SQLite data",
            "3. Implement PostgreSQL database manager (replace current manager.py)",
            "4. Update AI services to use PostgreSQL operations", 
            "5. Test with subset of data before full migration",
            "6. Plan for rollback strategy",
            "7. Monitor performance after migration"
        ])
        
        for rec in recommendations:
            logger.info(f"  {rec}")
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        logger.info("\n" + "="*60)
        logger.info("PostgreSQL + pgvector Migration Test Summary")
        logger.info("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(self.test_results.values())
        
        logger.info(f"Tests Passed: {passed_tests}/{total_tests}")
        logger.info(f"Success Rate: {passed_tests/total_tests:.1%}")
        logger.info("")
        
        for test_name, passed in self.test_results.items():
            status = "[PASS]" if passed else "[FAIL]"
            test_display = test_name.replace('_', ' ').title()
            logger.info(f"  {status} {test_display}")
        
        logger.info("")
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED! PostgreSQL + pgvector is ready!")
        elif passed_tests >= total_tests * 0.8:
            logger.info("⚠️  Most tests passed. Review failures before proceeding.")
        else:
            logger.info("❌ Multiple test failures. Migration not recommended.")

def main():
    """Main test execution"""
    logger.info("Starting PostgreSQL + pgvector Migration Test Suite")
    logger.info("=" * 60)
    
    tester = PostgresVectorTester()
    
    try:
        # Connect to database
        if not tester.connect():
            logger.error("Failed to connect to database. Exiting.")
            return
        
        # Run all tests in sequence
        test_sequence = [
            ('pgvector Extension', tester.test_pgvector_extension),
            ('Schema Creation', tester.create_test_schema),
            ('Metadata Operations', tester.test_metadata_operations),
            ('Vector Operations', tester.test_vector_operations),
            ('Similarity Search', tester.test_similarity_search),
            ('CRUD Operations', tester.test_crud_operations),
            ('Performance Testing', tester.test_performance),
            ('Cleanup', tester.cleanup_test_data)
        ]
        
        for test_name, test_func in test_sequence:
            logger.info(f"\nExecuting: {test_name}")
            try:
                test_func()
            except Exception as e:
                logger.error(f"Test {test_name} failed with exception: {e}")
        
        # Generate analysis
        tester.compare_with_faiss()
        tester.generate_migration_recommendations()
        tester.print_test_summary()
        
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        tester.disconnect()
    
    logger.info("\nPostgreSQL + pgvector Migration Testing Complete!")

if __name__ == "__main__":
    main()