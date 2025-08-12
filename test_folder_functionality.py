#!/usr/bin/env python3
"""
Test script to verify folder-based # functionality with PostgreSQL backend
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.database.postgres_manager import PostgresManager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_folder_functionality():
    """Test the folder-based search functionality"""
    try:
        # Initialize PostgreSQL manager
        logger.info("Initializing PostgreSQL manager...")
        db_manager = PostgresManager(
            host="localhost",
            database="meetingsai",
            user="postgres", 
            password="Sandeep@0904",
            port=5432,
            vector_dimension=1536
        )
        
        # Test user ID (using the testuser we created earlier)
        test_user_id = "test-user-001"
        
        # 1. Test get_all_documents to see folder_path values
        logger.info("=== Testing get_all_documents ===")
        all_docs = db_manager.get_all_documents(test_user_id)
        logger.info(f"Found {len(all_docs)} documents for user {test_user_id}")
        
        for doc in all_docs:
            logger.info(f"Document: {doc['filename']}")
            logger.info(f"  - Document ID: {doc['document_id']}")
            logger.info(f"  - Project ID: {doc['project_id']}")
            logger.info(f"  - Folder Path: {doc.get('folder_path', 'NOT SET')}")
            logger.info(f"  - Upload Date: {doc['upload_date']}")
            logger.info("")
        
        # 2. Test folder-based document retrieval
        if all_docs:
            # Try to find documents with specific folder paths
            folder_paths = set()
            for doc in all_docs:
                folder_path = doc.get('folder_path')
                if folder_path:
                    folder_paths.add(folder_path)
            
            logger.info(f"=== Found folder paths: {folder_paths} ===")
            
            for folder_path in folder_paths:
                logger.info(f"\nTesting folder: {folder_path}")
                folder_docs = db_manager.get_user_documents_by_folder(test_user_id, folder_path)
                logger.info(f"Found {len(folder_docs)} documents in folder '{folder_path}':")
                for doc_id in folder_docs:
                    # Find the document details
                    doc_details = next((d for d in all_docs if d['document_id'] == doc_id), None)
                    if doc_details:
                        logger.info(f"  - {doc_details['filename']}")
                    else:
                        logger.info(f"  - Document ID: {doc_id}")
            
            # 3. Test folder-based search methods
            if folder_paths:
                test_folder = list(folder_paths)[0]
                logger.info(f"\n=== Testing search methods with folder: {test_folder} ===")
                
                # Test keyword search
                test_keywords = ["document", "meeting"]
                keyword_results = db_manager.keyword_search_chunks_by_folder(
                    test_keywords, test_user_id, test_folder
                )
                logger.info(f"Keyword search results: {len(keyword_results)} chunks found")
                
                # Test vector search (need a sample embedding)
                try:
                    import numpy as np
                    # Create a dummy embedding for testing
                    dummy_embedding = np.random.rand(1536).astype(np.float32)
                    vector_results = db_manager.search_similar_chunks_by_folder(
                        dummy_embedding, test_user_id, test_folder
                    )
                    logger.info(f"Vector search results: {len(vector_results)} chunks found")
                except Exception as e:
                    logger.error(f"Vector search test failed: {e}")
        
        else:
            logger.warning("No documents found for testing!")
            
            # Let's check all users to see if documents exist
            logger.info("=== Checking documents for all users ===")
            all_docs_all_users = db_manager.get_all_documents()
            logger.info(f"Found {len(all_docs_all_users)} total documents")
            
            user_ids = set()
            for doc in all_docs_all_users:
                user_ids.add(doc['user_id'])
                logger.info(f"Document: {doc['filename']} (User: {doc['user_id']}, Folder: {doc.get('folder_path', 'NOT SET')})")
            
            logger.info(f"Users with documents: {user_ids}")
        
        logger.info("=== Test completed successfully! ===")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_folder_functionality()
    if success:
        print("\n✅ Folder functionality test PASSED")
    else:
        print("\n❌ Folder functionality test FAILED")
        sys.exit(1)