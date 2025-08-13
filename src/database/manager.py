"""
Database manager compatibility layer for PostgreSQL+pgvector architecture.
This module provides compatibility with existing service imports while delegating to PostgresManager.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Import PostgreSQL manager
from .postgres_manager import PostgresManager

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Compatibility layer that delegates all operations to PostgresManager.
    Maintains API compatibility with existing services while using PostgreSQL+pgvector only.
    """
    
    def __init__(self, host: str = "localhost", database: str = "meetingsai", 
                 user: str = "postgres", password: str = "Sandeep@0904", 
                 port: int = 5432, vector_dimension: int = 1536):
        """
        Initialize database manager with PostgreSQL connection.
        
        Args:
            host: PostgreSQL host
            database: Database name
            user: Database user
            password: Database password
            port: Database port
            vector_dimension: Vector dimension for pgvector
        """
        # Initialize PostgreSQL manager
        self.postgres_manager = PostgresManager(
            host=host,
            database=database, 
            user=user,
            password=password,
            port=port,
            vector_dimension=vector_dimension
        )
        
        logger.info("DatabaseManager initialized with PostgreSQL+pgvector backend")
    
    # Delegate all methods to PostgresManager
    def __getattr__(self, name):
        """Delegate any missing methods to postgres_manager"""
        return getattr(self.postgres_manager, name)
    
    # Explicit delegation for commonly used methods
    def get_all_documents(self, user_id: str = None) -> List[Dict[str, Any]]:
        """Get all documents with metadata for document selection"""
        return self.postgres_manager.get_all_documents(user_id)
    
    def get_user_projects(self, user_id: str):
        """Get all projects for a user"""
        return self.postgres_manager.get_user_projects(user_id)
    
    def create_project(self, user_id: str, project_name: str, description: str = "") -> str:
        """Create a new project for a user"""
        return self.postgres_manager.create_project(user_id, project_name, description)
    
    def get_user_meetings(self, user_id: str, project_id: str = None):
        """Get meetings for a user, optionally filtered by project"""
        return self.postgres_manager.get_user_meetings(user_id, project_id)
    
    def create_user(self, username: str, email: str, full_name: str, password_hash: str) -> str:
        """Create a new user"""
        return self.postgres_manager.create_user(username, email, full_name, password_hash)
    
    def get_user_by_username(self, username: str):
        """Get user by username"""
        return self.postgres_manager.get_user_by_username(username)
    
    def get_user_by_id(self, user_id: str):
        """Get user by user_id"""
        return self.postgres_manager.get_user_by_id(user_id)
    
    def update_user_last_login(self, user_id: str):
        """Update user's last login timestamp"""
        return self.postgres_manager.update_user_last_login(user_id)
    
    def create_session(self, user_id: str, session_id: str, expires_at: datetime) -> bool:
        """Create a new user session"""
        return self.postgres_manager.create_session(user_id, session_id, expires_at)
    
    def validate_session(self, session_id: str) -> Optional[str]:
        """Validate a session and return user_id if valid"""
        return self.postgres_manager.validate_session(session_id)
    
    def deactivate_session(self, session_id: str) -> bool:
        """Deactivate a session"""
        return self.postgres_manager.deactivate_session(session_id)
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions and return count of cleaned sessions"""
        return self.postgres_manager.cleanup_expired_sessions()
    
    # Vector search compatibility methods (delegate to PostgreSQL pgvector)
    def search_similar_chunks(self, query_embedding, top_k: int = 20):
        """Search for similar chunks using pgvector"""
        return self.postgres_manager.search_similar_chunks(query_embedding, top_k)
    
    def enhanced_search_with_metadata(self, query_embedding, user_id: str, 
                                    filters: Dict = None, top_k: int = 20):
        """Enhanced search with metadata filtering"""
        return self.postgres_manager.enhanced_search_with_metadata(
            query_embedding, user_id, filters, top_k
        )
    
    def get_chunks_by_ids(self, chunk_ids: List[str]):
        """Retrieve chunks by their IDs"""
        return self.postgres_manager.get_chunks_by_ids(chunk_ids)
    
    def add_document(self, document, chunks: List):
        """Add a document and all its chunks to PostgreSQL+pgvector"""
        return self.postgres_manager.add_document(document, chunks)
    
    # Job management methods
    def create_upload_job(self, user_id: str, total_files: int, project_id: str = None, 
                         meeting_id: str = None) -> str:
        """Create a new upload job for tracking batch processing"""
        return self.postgres_manager.create_upload_job(user_id, total_files, project_id, meeting_id)
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get current job status and progress"""
        return self.postgres_manager.get_job_status(job_id)
    
    def update_job_status(self, job_id: str, status: str, processed_files: int = None, 
                         failed_files: int = None, error_message: str = None):
        """Update job status and progress"""
        return self.postgres_manager.update_job_status(
            job_id, status, processed_files, failed_files, error_message
        )
    
    # Additional compatibility methods for legacy code
    def store_document_metadata(self, filename: str, content: str, user_id: str, 
                              project_id: str = None, meeting_id: str = None) -> str:
        """Store document metadata and return document_id"""
        return self.postgres_manager.store_document_metadata(
            filename, content, user_id, project_id, meeting_id
        )
    
    def get_document_metadata(self, document_id: str) -> Dict[str, Any]:
        """Get metadata for a specific document"""
        return self.postgres_manager.get_document_metadata(document_id)
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        return self.postgres_manager.get_database_stats()