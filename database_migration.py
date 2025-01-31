#!/usr/bin/env python3
"""
CineBrain Database Migration Script
Adds updated_at column to admin_recommendation table
"""

import os
import sys
import psycopg2
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_database_url():
    """Get database URL from environment or input"""
    db_url = "postgresql://database_zlr3_user:SYIQRe2v60hCePHpdz0CKNXRlGCVuAG0@dpg-d45i1k15pdvs73c2tqh0-a.singapore-postgres.render.com/database_zlr3"
    
    if not db_url:
        print("DATABASE_URL not found in environment variables.")
        db_url = input("Please enter your PostgreSQL connection URL: ")
    
    # Handle both postgres:// and postgresql:// formats
    if db_url.startswith('postgres://'):
        db_url = db_url.replace('postgres://', 'postgresql://', 1)
    
    return db_url

def test_connection(db_url):
    """Test database connection"""
    try:
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        logger.info(f"✅ Database connection successful: {version[0]}")
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

def check_column_exists(db_url, table_name, column_name):
    """Check if column exists in table"""
    try:
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = %s AND column_name = %s;
        """, (table_name, column_name))
        
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        return result is not None
    except Exception as e:
        logger.error(f"❌ Error checking column existence: {e}")
        return False

def check_table_exists(db_url, table_name):
    """Check if table exists"""
    try:
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_name = %s;
        """, (table_name,))
        
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        return result is not None
    except Exception as e:
        logger.error(f"❌ Error checking table existence: {e}")
        return False

def add_updated_at_column(db_url):
    """Add updated_at column to admin_recommendation table"""
    try:
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        # Check if table exists first
        if not check_table_exists(db_url, 'admin_recommendation'):
            logger.warning("⚠️ admin_recommendation table does not exist")
            return False
        
        # Check if column already exists
        if check_column_exists(db_url, 'admin_recommendation', 'updated_at'):
            logger.info("⚠️ updated_at column already exists in admin_recommendation table")
            cursor.close()
            conn.close()
            return True
        
        # Add the column
        logger.info("📝 Adding updated_at column to admin_recommendation table...")
        cursor.execute("""
            ALTER TABLE admin_recommendation 
            ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
        """)
        
        # Update existing records to set updated_at = created_at
        logger.info("📝 Updating existing records...")
        cursor.execute("""
            UPDATE admin_recommendation 
            SET updated_at = created_at 
            WHERE updated_at IS NULL;
        """)
        
        # Get count of updated records
        cursor.execute("SELECT COUNT(*) FROM admin_recommendation;")
        total_records = cursor.fetchone()[0]
        
        # Commit the changes
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"✅ Successfully added updated_at column and updated {total_records} existing records")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error adding updated_at column: {e}")
        try:
            conn.rollback()
            cursor.close()
            conn.close()
        except:
            pass
        return False

def verify_migration(db_url):
    """Verify the migration was successful"""
    try:
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        # Check column exists
        cursor.execute("""
            SELECT column_name, data_type, column_default
            FROM information_schema.columns 
            WHERE table_name = 'admin_recommendation' AND column_name = 'updated_at';
        """)
        
        result = cursor.fetchone()
        if result:
            logger.info(f"✅ Column verification: {result[0]} ({result[1]}) with default: {result[2]}")
        
        # Check sample data
        cursor.execute("""
            SELECT id, created_at, updated_at 
            FROM admin_recommendation 
            LIMIT 3;
        """)
        
        records = cursor.fetchall()
        logger.info(f"📊 Sample records ({len(records)} found):")
        for record in records:
            logger.info(f"   ID: {record[0]}, Created: {record[1]}, Updated: {record[2]}")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        return False

def show_table_structure(db_url):
    """Show current table structure"""
    try:
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns 
            WHERE table_name = 'admin_recommendation'
            ORDER BY ordinal_position;
        """)
        
        columns = cursor.fetchall()
        logger.info("📋 Current admin_recommendation table structure:")
        logger.info("   Column Name          | Data Type    | Nullable | Default")
        logger.info("   ---------------------|--------------|----------|----------")
        for col in columns:
            nullable = "YES" if col[2] == "YES" else "NO"
            default = str(col[3]) if col[3] else "None"
            logger.info(f"   {col[0]:<20} | {col[1]:<12} | {nullable:<8} | {default}")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Error showing table structure: {e}")
        return False

def main():
    """Main migration function"""
    print("🎬 CineBrain Database Migration Tool")
    print("=====================================")
    print("Adding 'updated_at' column to admin_recommendation table")
    print()
    
    # Get database URL
    db_url = get_database_url()
    
    if not db_url:
        logger.error("❌ No database URL provided")
        sys.exit(1)
    
    # Test connection
    logger.info("🔍 Testing database connection...")
    if not test_connection(db_url):
        logger.error("❌ Cannot connect to database")
        sys.exit(1)
    
    # Show current table structure
    logger.info("📋 Checking current table structure...")
    show_table_structure(db_url)
    print()
    
    # Confirm migration
    response = input("Do you want to proceed with the migration? (yes/no): ").lower().strip()
    if response not in ['yes', 'y']:
        logger.info("❌ Migration cancelled by user")
        sys.exit(0)
    
    # Perform migration
    logger.info("🚀 Starting migration...")
    if add_updated_at_column(db_url):
        logger.info("✅ Migration completed successfully!")
        
        # Verify migration
        logger.info("🔍 Verifying migration...")
        if verify_migration(db_url):
            logger.info("✅ Migration verification successful!")
            print()
            print("🎉 Migration completed successfully!")
            print("You can now uncomment the updated_at field in your AdminRecommendation model.")
        else:
            logger.warning("⚠️ Migration verification failed")
    else:
        logger.error("❌ Migration failed")
        sys.exit(1)

if __name__ == "__main__":
    main()