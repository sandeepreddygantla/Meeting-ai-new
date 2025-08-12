#!/usr/bin/env python3
"""
PostgreSQL Connection Test Script
Tests connection to PostgreSQL database running on localhost
"""

import psycopg2
from psycopg2 import sql
import sys
import os

def test_postgres_connection():
    """Test connection to PostgreSQL database"""
    
    # Database connection parameters
    db_config = {
        'host': 'localhost',
        'database': 'postgres',  # Default database
        'user': 'postgres',
        'password': 'Sandeep@0904',
        'port': '5432'  # Default PostgreSQL port
    }
    
    try:
        print("Attempting to connect to PostgreSQL...")
        print(f"Host: {db_config['host']}")
        print(f"Database: {db_config['database']}")
        print(f"User: {db_config['user']}")
        print(f"Port: {db_config['port']}")
        print("-" * 50)
        
        # Establish connection
        conn = psycopg2.connect(**db_config)
        
        # Create cursor
        cursor = conn.cursor()
        
        # Test basic queries
        print("[SUCCESS] Connection successful!")
        
        # Get PostgreSQL version
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"PostgreSQL Version: {version[0]}")
        
        # Get current database
        cursor.execute("SELECT current_database();")
        current_db = cursor.fetchone()
        print(f"Current Database: {current_db[0]}")
        
        # Get current user
        cursor.execute("SELECT current_user;")
        current_user = cursor.fetchone()
        print(f"Current User: {current_user[0]}")
        
        # List all databases
        cursor.execute("SELECT datname FROM pg_database WHERE datistemplate = false;")
        databases = cursor.fetchall()
        print(f"Available Databases: {[db[0] for db in databases]}")
        
        # Test creating a simple table
        print("\nTesting table creation...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_connection (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Insert test data
        cursor.execute("""
            INSERT INTO test_connection (name) VALUES (%s)
            ON CONFLICT DO NOTHING;
        """, ("Connection Test",))
        
        # Retrieve test data
        cursor.execute("SELECT * FROM test_connection LIMIT 5;")
        test_data = cursor.fetchall()
        print(f"Test table data: {test_data}")
        
        # Clean up test table
        cursor.execute("DROP TABLE IF EXISTS test_connection;")
        
        # Commit changes
        conn.commit()
        
        print("\n[SUCCESS] All tests passed! PostgreSQL connection is working correctly.")
        
    except psycopg2.Error as e:
        print(f"[ERROR] PostgreSQL Error: {e}")
        return False
        
    except Exception as e:
        print(f"[ERROR] General Error: {e}")
        return False
        
    finally:
        # Close connections
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()
        print("Connection closed.")
    
    return True

def check_psycopg2_installation():
    """Check if psycopg2 is properly installed"""
    try:
        import psycopg2
        print(f"[SUCCESS] psycopg2 version: {psycopg2.__version__}")
        return True
    except ImportError:
        print("[ERROR] psycopg2 is not installed. Please run: pip install psycopg2-binary")
        return False

if __name__ == "__main__":
    print("PostgreSQL Connection Test")
    print("=" * 50)
    
    # Check if psycopg2 is installed
    if not check_psycopg2_installation():
        sys.exit(1)
    
    # Test connection
    if test_postgres_connection():
        print("\n[SUCCESS] PostgreSQL setup is complete and working!")
        sys.exit(0)
    else:
        print("\n[ERROR] PostgreSQL connection failed. Please check your configuration.")
        sys.exit(1)