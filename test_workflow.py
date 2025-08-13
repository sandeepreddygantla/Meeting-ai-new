#!/usr/bin/env python3
"""
Test script to verify the pending->completed workflow fix.
This script simulates the document upload workflow to test our fix.
"""

import os
import sys
from datetime import datetime
from src.database.postgres_manager import PostgresManager
from src.models.document import MeetingDocument

def test_pending_completed_workflow():
    """Test the pending->completed workflow fix."""
    print("Testing pending->completed workflow...")
    
    # Initialize database manager
    db_manager = PostgresManager()
    
    # Create test user first
    test_user_id = "ebdc7ace-a59a-4fa5-ba8b-237354a3ecd3"  # Use existing testuser ID
    test_filename = "test-workflow-document.docx"
    
    # Test document 1: Create initial pending record
    doc1 = MeetingDocument(
        document_id="doc-123-pending",
        user_id=test_user_id,
        project_id="test-project-1",
        meeting_id=None,
        filename=test_filename,
        original_filename=test_filename,
        file_path=f"/uploads/{test_filename}",
        file_size=1000,
        file_hash="abc123",
        content="Test content",
        processed_content="Test processed content",
        processing_status="pending"
    )
    
    print("1. Adding initial pending document...")
    try:
        db_manager.add_document(doc1, [])  # No chunks for this test
        print("   [SUCCESS] Pending document added successfully")
    except Exception as e:
        print(f"   [ERROR] Error adding pending document: {e}")
        return False
    
    # Test document 2: Simulate completion workflow (same filename)
    doc2 = MeetingDocument(
        document_id="doc-123-completed",
        user_id=test_user_id,
        project_id="test-project-1", 
        meeting_id=None,
        filename=test_filename,  # Same filename
        original_filename=test_filename,
        file_path=f"/uploads/{test_filename}",
        file_size=1000,
        file_hash="abc123",
        content="Test content with processing complete",
        processed_content="Test processed content complete",
        processing_status="completed",
        summary="Test summary",
        chunk_count=5
    )
    
    print("2. Processing completion (should UPDATE existing pending record)...")
    try:
        db_manager.add_document(doc2, [])  # No chunks for this test
        print("   [SUCCESS] Document completion processed successfully")
    except Exception as e:
        print(f"   [ERROR] Error processing completion: {e}")
        return False
    
    # Verify the workflow: Check that we only have one record with 'completed' status
    print("3. Verifying workflow results...")
    try:
        with db_manager.get_cursor() as (conn, cursor):
            cursor.execute("""
                SELECT document_id, processing_status, COUNT(*) OVER () as total_count
                FROM documents 
                WHERE user_id = %s AND filename = %s;
            """, (test_user_id, test_filename))
            
            records = cursor.fetchall()
            
            if len(records) == 1:
                doc_id, status, total = records[0]
                if status == 'completed':
                    print("   [SUCCESS] Found exactly 1 record with 'completed' status")
                    print(f"     Document ID: {doc_id}")
                    return True
                else:
                    print(f"   [FAIL] Record has status '{status}', expected 'completed'")
                    return False
            elif len(records) == 0:
                print("   [FAIL] No records found")
                return False
            else:
                print(f"   [FAIL] Found {len(records)} records, expected exactly 1")
                for i, (doc_id, status, total) in enumerate(records):
                    print(f"     Record {i+1}: ID={doc_id}, Status={status}")
                return False
                
    except Exception as e:
        print(f"   [ERROR] Error verifying results: {e}")
        return False
    
    finally:
        # Cleanup test data
        try:
            with db_manager.get_cursor() as (conn, cursor):
                cursor.execute("DELETE FROM documents WHERE user_id = %s AND filename = %s", (test_user_id, test_filename))
                print("   [CLEANUP] Test data cleaned up")
        except Exception as e:
            print(f"   [WARNING] Could not clean up test data: {e}")

def main():
    """Main test function."""
    print("=== Document Workflow Test ===")
    print("Testing the pending->completed workflow fix...")
    print()
    
    try:
        success = test_pending_completed_workflow()
        print()
        if success:
            print("[PASS] TEST PASSED: Pending->completed workflow is working correctly!")
            print("   - No duplicate records created")
            print("   - Pending record properly updated to completed")
            return 0
        else:
            print("[FAIL] TEST FAILED: Workflow fix needs attention")
            return 1
            
    except Exception as e:
        print(f"[ERROR] TEST ERROR: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())