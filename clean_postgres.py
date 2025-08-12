#!/usr/bin/env python3
"""
Clean PostgreSQL database completely for fresh testing
"""
import psycopg2
import logging

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'meetingsai',
    'user': 'postgres',
    'password': 'Sandeep@0904'
}

def clean_database():
    """Clean all data from PostgreSQL tables"""
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        print("Starting PostgreSQL database cleanup...")
        
        # Get all table names
        cursor.execute("""
            SELECT tablename FROM pg_tables 
            WHERE schemaname = 'public' 
            ORDER BY tablename;
        """)
        tables = cursor.fetchall()
        
        print(f"Found {len(tables)} tables to clean:")
        for table in tables:
            print(f"  - {table[0]}")
        
        # Disable foreign key constraints temporarily
        cursor.execute("SET session_replication_role = replica;")
        
        # Clean all tables
        for table in tables:
            table_name = table[0]
            cursor.execute(f"TRUNCATE TABLE {table_name} CASCADE;")
            print(f"Cleaned table: {table_name}")
        
        # Re-enable foreign key constraints
        cursor.execute("SET session_replication_role = DEFAULT;")
        
        # Commit changes
        conn.commit()
        
        # Verify cleanup
        print("\nVerifying cleanup:")
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            count = cursor.fetchone()[0]
            print(f"  {table_name}: {count} rows")
        
        cursor.close()
        conn.close()
        
        print("\nPostgreSQL database cleanup completed successfully!")
        
    except Exception as e:
        print(f"Error cleaning database: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = clean_database()
    if success:
        print("\nDatabase is now clean and ready for fresh testing!")
    else:
        print("\nDatabase cleanup failed!")