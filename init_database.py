#!/usr/bin/env python3
"""
Database initialization and seeding script for Construction Budget Management System
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from database import DatabaseManager
from data.seed_data import SeedDataManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_database():
    """
    Initialize and seed the database with required data
    """
    try:
        logger.info("Starting database initialization...")
        
        # Initialize database manager
        db_manager = DatabaseManager()
        logger.info("Database tables created successfully")
        
        # Initialize seed data manager
        seed_manager = SeedDataManager()
        
        # Seed all required data
        logger.info("Seeding database with initial data...")
        seed_manager.seed_all_data()
        
        # Verify seeded data
        logger.info("Verifying seeded data...")
        seed_manager.verify_seed_data()
        
        logger.info("Database initialization completed successfully!")
        
        # Display summary
        projects = db_manager.get_all_projects()
        logger.info(f"Current projects in database: {len(projects)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during database initialization: {str(e)}")
        return False

def clean_and_reset_database():
    """
    Clean and reset the database (useful for development)
    """
    try:
        logger.info("Cleaning and resetting database...")
        
        seed_manager = SeedDataManager()
        seed_manager.clean_database()
        
        # Re-initialize
        return initialize_database()
        
    except Exception as e:
        logger.error(f"Error during database reset: {str(e)}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize Construction Budget Database")
    parser.add_argument("--reset", action="store_true", help="Clean and reset the database")
    parser.add_argument("--verify", action="store_true", help="Only verify existing data")
    
    args = parser.parse_args()
    
    if args.reset:
        success = clean_and_reset_database()
    elif args.verify:
        try:
            seed_manager = SeedDataManager()
            seed_manager.verify_seed_data()
            success = True
        except Exception as e:
            logger.error(f"Verification failed: {str(e)}")
            success = False
    else:
        success = initialize_database()
    
    if success:
        logger.info("Database operation completed successfully!")
        sys.exit(0)
    else:
        logger.error("Database operation failed!")
        sys.exit(1)