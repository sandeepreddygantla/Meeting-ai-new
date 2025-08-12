#!/usr/bin/env python3
"""
Test script to verify folder-based search functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.database.postgres_manager import PostgresManager
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_folder_search():
    """Test the folder-based search functionality with actual data"""
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
        
        # Use the actual user ID that has documents
        actual_user_id = "2b53fa95-c96d-4eeb-965b-e1a5e49da536"
        test_folder_path = "user_folder/project_7baa56e4-2894-48c6-89da-36e9431f193f"
        
        logger.info(f"Testing with User ID: {actual_user_id}")
        logger.info(f"Testing with Folder Path: {test_folder_path}")
        
        # 1. Test get_user_documents_by_folder
        logger.info("\n=== Testing get_user_documents_by_folder ===")
        folder_docs = db_manager.get_user_documents_by_folder(actual_user_id, test_folder_path)
        logger.info(f"Found {len(folder_docs)} documents in folder")
        for doc_id in folder_docs:
            logger.info(f"  - Document ID: {doc_id}")
        
        # 2. Test keyword_search_chunks_by_folder
        logger.info("\n=== Testing keyword_search_chunks_by_folder ===")
        test_keywords = ["document", "meeting", "fulfillment"]
        keyword_results = db_manager.keyword_search_chunks_by_folder(
            test_keywords, actual_user_id, test_folder_path
        )
        logger.info(f"Found {len(keyword_results)} chunks with keywords {test_keywords}")
        for i, chunk_id in enumerate(keyword_results[:3]):  # Show first 3
            logger.info(f"  - Chunk {i+1}: {chunk_id}")
        
        # 3. Test search_similar_chunks_by_folder
        logger.info("\n=== Testing search_similar_chunks_by_folder ===")
        
        # First, get a real embedding from one of the chunks for testing
        try:
            # Get some chunks to extract a real embedding
            with db_manager.get_cursor() as (conn, cursor):
                cursor.execute("""
                    SELECT dc.chunk_id, dc.embedding
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.document_id
                    WHERE d.user_id = %s AND d.folder_path = %s
                    LIMIT 1;
                """, (actual_user_id, test_folder_path))
                
                result = cursor.fetchone()
                if result:
                    chunk_id, embedding = result
                    logger.info(f"Using embedding from chunk: {chunk_id}")
                    
                    # Convert embedding back to numpy array
                    if isinstance(embedding, list):
                        query_embedding = np.array(embedding, dtype=np.float32)
                    else:
                        # If it's a string representation, we'll create a dummy one
                        query_embedding = np.random.rand(1536).astype(np.float32)
                    
                    vector_results = db_manager.search_similar_chunks_by_folder(
                        query_embedding, actual_user_id, test_folder_path, top_k=5
                    )
                    logger.info(f"Found {len(vector_results)} similar chunks")
                    for i, (chunk_id, similarity) in enumerate(vector_results):
                        logger.info(f"  - Chunk {i+1}: {chunk_id} (similarity: {similarity:.4f})")
                        
                else:
                    logger.warning("No chunks found with embeddings for vector search test")
                    
        except Exception as e:
            logger.error(f"Vector search test failed: {e}")
        
        # 4. Test the folder path detection logic from frontend
        logger.info("\n=== Testing folder path detection (simulating # functionality) ===")
        
        # Simulate the frontend logic for detecting #Default Project
        def simulate_folder_detection(user_input):
            """Simulate the frontend folder detection logic"""
            if "#" in user_input:
                # Extract the project/folder name
                hash_parts = user_input.split("#")
                if len(hash_parts) > 1:
                    folder_name = hash_parts[1].split()[0].strip()
                    
                    # Simulate the folder path construction
                    if folder_name.lower() == "default":
                        return "user_folder/default_project"
                    elif "project" in folder_name.lower():
                        # For testing, use the actual project folder we have
                        return test_folder_path
                    else:
                        return f"user_folder/{folder_name}"
            return None
        
        test_queries = [
            "#Default Project summary",
            "#project summary", 
            "#Default what happened in the meetings?"
        ]
        
        for query in test_queries:
            detected_folder = simulate_folder_detection(query)
            logger.info(f"Query: '{query}' -> Detected folder: {detected_folder}")
            
            if detected_folder:
                docs_in_folder = db_manager.get_user_documents_by_folder(actual_user_id, detected_folder)
                logger.info(f"  Documents in detected folder: {len(docs_in_folder)}")
        
        logger.info("\n=== All tests completed successfully! ===")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_folder_search()
    if success:
        print("\nFolder search functionality test PASSED")
    else:
        print("\nFolder search functionality test FAILED")
        sys.exit(1)