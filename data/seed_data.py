import sqlite3
import os
import logging
from datetime import datetime, timedelta
import random
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SeedDataManager:
    """
    Manages seeding of initial data for the construction budget management system
    """
    
    def __init__(self, db_path: str = "construction_budget.db"):
        self.db_path = db_path
        
    def seed_all_data(self):
        """
        Seed all initial data required for the system
        """
        try:
            logger.info("Starting database seeding process...")
            
            # Ensure database exists
            self._ensure_database_exists()
            
            # Seed NBR 12721 standards
            self._seed_nbr_standards()
            
            # Seed historical data for ML training
            self._seed_historical_data()
            
            # Seed regional adjustment factors
            self._seed_regional_factors()
            
            # Seed material cost references
            self._seed_material_references()
            
            logger.info("Database seeding completed successfully!")
            
        except Exception as e:
            logger.error(f"Error seeding database: {str(e)}")
            raise
    
    def _ensure_database_exists(self):
        """
        Ensure database and tables exist
        """
        try:
            # Import DatabaseManager to initialize tables
            import sys
            sys.path.append('..')
            from database import DatabaseManager
            
            db_manager = DatabaseManager(self.db_path)
            logger.info("Database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def _seed_nbr_standards(self):
        """
        Seed NBR 12721 compliance standards and requirements
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create NBR standards table if not exists
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS nbr_standards (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        standard_code TEXT NOT NULL,
                        category TEXT NOT NULL,
                        subcategory TEXT,
                        requirement TEXT NOT NULL,
                        min_value REAL,
                        max_value REAL,
                        typical_value REAL,
                        unit TEXT,
                        project_type TEXT,
                        notes TEXT,
                        created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Clear existing standards
                cursor.execute('DELETE FROM nbr_standards')
                
                # NBR 12721 material ratio standards
                material_standards = [
                    # Concrete/Masonry ratios by project type
                    ('NBR 12721', 'Material Ratio', 'Concrete/Masonry', 'Percentage of total project cost for concrete and masonry work', 
                     15.0, 25.0, 20.0, '%', 'Residential', 'Standard residential construction ratios'),
                    ('NBR 12721', 'Material Ratio', 'Concrete/Masonry', 'Percentage of total project cost for concrete and masonry work', 
                     18.0, 28.0, 23.0, '%', 'Commercial', 'Commercial building construction ratios'),
                    ('NBR 12721', 'Material Ratio', 'Concrete/Masonry', 'Percentage of total project cost for concrete and masonry work', 
                     20.0, 30.0, 25.0, '%', 'Industrial', 'Industrial construction ratios'),
                    
                    # Steel/Concrete structural ratios
                    ('NBR 12721', 'Material Ratio', 'Steel/Concrete', 'Percentage of total project cost for structural elements', 
                     10.0, 18.0, 14.0, '%', 'Residential', 'Residential structural work ratios'),
                    ('NBR 12721', 'Material Ratio', 'Steel/Concrete', 'Percentage of total project cost for structural elements', 
                     12.0, 20.0, 16.0, '%', 'Commercial', 'Commercial structural work ratios'),
                    ('NBR 12721', 'Material Ratio', 'Steel/Concrete', 'Percentage of total project cost for structural elements', 
                     15.0, 25.0, 20.0, '%', 'Industrial', 'Industrial structural work ratios'),
                    
                    # Roofing materials
                    ('NBR 12721', 'Material Ratio', 'Roofing Materials', 'Percentage of total project cost for roofing systems', 
                     3.0, 8.0, 5.0, '%', 'Residential', 'Residential roofing ratios'),
                    ('NBR 12721', 'Material Ratio', 'Roofing Materials', 'Percentage of total project cost for roofing systems', 
                     4.0, 9.0, 6.0, '%', 'Commercial', 'Commercial roofing ratios'),
                    ('NBR 12721', 'Material Ratio', 'Roofing Materials', 'Percentage of total project cost for roofing systems', 
                     5.0, 10.0, 7.0, '%', 'Industrial', 'Industrial roofing ratios'),
                    
                    # Doors and Windows
                    ('NBR 12721', 'Material Ratio', 'Doors/Windows', 'Percentage of total project cost for openings', 
                     5.0, 12.0, 8.0, '%', 'Residential', 'Residential openings ratios'),
                    ('NBR 12721', 'Material Ratio', 'Doors/Windows', 'Percentage of total project cost for openings', 
                     8.0, 15.0, 11.0, '%', 'Commercial', 'Commercial openings ratios'),
                    ('NBR 12721', 'Material Ratio', 'Doors/Windows', 'Percentage of total project cost for openings', 
                     4.0, 10.0, 7.0, '%', 'Industrial', 'Industrial openings ratios'),
                    
                    # Finishes
                    ('NBR 12721', 'Material Ratio', 'Finishes', 'Percentage of total project cost for finishing work', 
                     8.0, 18.0, 13.0, '%', 'Residential', 'Residential finishing ratios'),
                    ('NBR 12721', 'Material Ratio', 'Finishes', 'Percentage of total project cost for finishing work', 
                     10.0, 20.0, 15.0, '%', 'Commercial', 'Commercial finishing ratios'),
                    ('NBR 12721', 'Material Ratio', 'Finishes', 'Percentage of total project cost for finishing work', 
                     5.0, 12.0, 8.0, '%', 'Industrial', 'Industrial finishing ratios'),
                    
                    # MEP Systems
                    ('NBR 12721', 'Material Ratio', 'MEP Systems', 'Percentage of total project cost for MEP installations', 
                     12.0, 20.0, 16.0, '%', 'Residential', 'Residential MEP ratios'),
                    ('NBR 12721', 'Material Ratio', 'MEP Systems', 'Percentage of total project cost for MEP installations', 
                     15.0, 25.0, 20.0, '%', 'Commercial', 'Commercial MEP ratios'),
                    ('NBR 12721', 'Material Ratio', 'MEP Systems', 'Percentage of total project cost for MEP installations', 
                     18.0, 30.0, 24.0, '%', 'Industrial', 'Industrial MEP ratios'),
                ]
                
                # Insert material standards
                cursor.executemany('''
                    INSERT INTO nbr_standards (standard_code, category, subcategory, requirement, 
                                             min_value, max_value, typical_value, unit, project_type, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', material_standards)
                
                # Unit cost standards (R$ values as of 2024)
                unit_cost_standards = [
                    # Concrete costs per m³
                    ('NBR 12721', 'Unit Cost', 'Concrete/Masonry', 'Standard concrete C20 cost per cubic meter', 
                     120.0, 200.0, 160.0, 'R$/m³', 'All', 'Includes materials and basic labor'),
                    
                    # Structural steel costs
                    ('NBR 12721', 'Unit Cost', 'Steel/Concrete', 'Structural steel cost per cubic meter equivalent', 
                     600.0, 1000.0, 800.0, 'R$/m³', 'All', 'Includes fabrication and installation'),
                    
                    # Roofing materials
                    ('NBR 12721', 'Unit Cost', 'Roofing Materials', 'Standard roofing materials per square meter', 
                     30.0, 60.0, 45.0, 'R$/m²', 'All', 'Ceramic tiles, basic waterproofing'),
                    
                    # Doors and windows
                    ('NBR 12721', 'Unit Cost', 'Doors/Windows', 'Standard door/window unit cost', 
                     300.0, 800.0, 550.0, 'R$/unit', 'All', 'Medium quality aluminum/wood'),
                    
                    # Finishes
                    ('NBR 12721', 'Unit Cost', 'Finishes', 'Standard finishing cost per square meter', 
                     50.0, 120.0, 85.0, 'R$/m²', 'All', 'Paint, basic flooring, standard fixtures'),
                    
                    # MEP Systems
                    ('NBR 12721', 'Unit Cost', 'MEP Systems', 'MEP installation cost per cubic meter of building', 
                     150.0, 300.0, 225.0, 'R$/m³', 'All', 'Electrical, plumbing, HVAC basic systems'),
                ]
                
                cursor.executemany('''
                    INSERT INTO nbr_standards (standard_code, category, subcategory, requirement, 
                                             min_value, max_value, typical_value, unit, project_type, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', unit_cost_standards)
                
                # Quality and safety standards
                quality_standards = [
                    ('NBR 12721', 'Quality', 'Concrete Strength', 'Minimum concrete compressive strength', 
                     20.0, None, 25.0, 'MPa', 'All', 'Standard structural concrete'),
                    ('NBR 12721', 'Quality', 'Steel Grade', 'Minimum steel grade for structural use', 
                     250.0, None, 300.0, 'MPa', 'All', 'Structural steel yield strength'),
                    ('NBR 12721', 'Safety', 'Load Factor', 'Safety factor for structural design', 
                     1.4, None, 1.4, 'factor', 'All', 'Dead load safety factor'),
                    ('NBR 12721', 'Safety', 'Live Load Factor', 'Safety factor for live loads', 
                     1.7, None, 1.7, 'factor', 'All', 'Live load safety factor'),
                ]
                
                cursor.executemany('''
                    INSERT INTO nbr_standards (standard_code, category, subcategory, requirement, 
                                             min_value, max_value, typical_value, unit, project_type, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', quality_standards)
                
                conn.commit()
                logger.info("NBR 12721 standards seeded successfully")
                
        except Exception as e:
            logger.error(f"Error seeding NBR standards: {str(e)}")
            raise
    
    def _seed_historical_data(self):
        """
        Seed historical data for ML model training
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Clear existing historical data
                cursor.execute('DELETE FROM historical_data')
                
                # Generate 3 years of historical data
                start_date = datetime.now() - timedelta(days=3*365)
                
                material_categories = [
                    'Concrete/Masonry', 'Steel/Concrete', 'Roofing Materials',
                    'Doors/Windows', 'Finishes', 'MEP Systems'
                ]
                
                project_types = ['Residential', 'Commercial', 'Industrial', 'Infrastructure']
                
                locations = [
                    'São Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'Brasília',
                    'Salvador', 'Porto Alegre', 'Curitiba', 'Fortaleza', 'Recife', 'Manaus'
                ]
                
                # Base unit costs (R$)
                base_unit_costs = {
                    'Concrete/Masonry': 160.0,
                    'Steel/Concrete': 800.0,
                    'Roofing Materials': 45.0,
                    'Doors/Windows': 550.0,
                    'Finishes': 85.0,
                    'MEP Systems': 225.0
                }
                
                # Project type multipliers
                type_multipliers = {
                    'Residential': 1.0,
                    'Commercial': 1.2,
                    'Industrial': 1.4,
                    'Infrastructure': 1.6
                }
                
                # Location multipliers (cost of living adjustments)
                location_multipliers = {
                    'São Paulo': 1.15,
                    'Rio de Janeiro': 1.10,
                    'Belo Horizonte': 1.00,
                    'Brasília': 1.05,
                    'Salvador': 0.95,
                    'Porto Alegre': 1.03,
                    'Curitiba': 1.01,
                    'Fortaleza': 0.90,
                    'Recife': 0.92,
                    'Manaus': 1.20
                }
                
                # Generate 2000 historical records
                historical_records = []
                
                for i in range(2000):
                    # Random selections
                    material = random.choice(material_categories)
                    project_type = random.choice(project_types)
                    location = random.choice(locations)
                    
                    # Random date in the last 3 years
                    random_days = random.randint(0, 3*365)
                    project_date = start_date + timedelta(days=random_days)
                    
                    # Calculate quantity (log-normal distribution)
                    quantity = max(1.0, np.random.lognormal(2.0, 1.0))
                    
                    # Calculate unit cost with variations
                    base_cost = base_unit_costs[material]
                    
                    # Apply multipliers
                    unit_cost = (base_cost * 
                               type_multipliers[project_type] * 
                               location_multipliers[location])
                    
                    # Add time-based inflation (2% per year)
                    years_ago = (datetime.now() - project_date).days / 365.25
                    inflation_factor = (1.02) ** years_ago
                    unit_cost = unit_cost / inflation_factor
                    
                    # Add market volatility (±15%)
                    volatility = np.random.normal(1.0, 0.10)
                    unit_cost = unit_cost * volatility
                    
                    # Seasonal adjustments
                    month = project_date.month
                    if month in [6, 7, 8]:  # Winter (dry season)
                        seasonal_factor = 0.95
                    elif month in [12, 1, 2]:  # Summer (rainy season)
                        seasonal_factor = 1.05
                    else:
                        seasonal_factor = 1.0
                    
                    unit_cost = unit_cost * seasonal_factor
                    
                    # Ensure positive costs
                    unit_cost = max(unit_cost, base_cost * 0.5)
                    total_cost = unit_cost * quantity
                    
                    historical_records.append((
                        material, project_type, location, quantity,
                        unit_cost, total_cost, project_date.date()
                    ))
                
                # Insert historical data
                cursor.executemany('''
                    INSERT INTO historical_data 
                    (material_category, project_type, location, quantity, 
                     unit_cost, total_cost, project_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', historical_records)
                
                conn.commit()
                logger.info(f"Generated {len(historical_records)} historical records for ML training")
                
        except Exception as e:
            logger.error(f"Error seeding historical data: {str(e)}")
            raise
    
    def _seed_regional_factors(self):
        """
        Seed regional adjustment factors
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create regional factors table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS regional_factors (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        location TEXT NOT NULL UNIQUE,
                        cost_multiplier REAL NOT NULL,
                        labor_availability TEXT,
                        material_availability TEXT,
                        logistics_difficulty TEXT,
                        market_maturity TEXT,
                        notes TEXT,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Clear existing factors
                cursor.execute('DELETE FROM regional_factors')
                
                regional_data = [
                    ('São Paulo', 1.15, 'High', 'Excellent', 'Low', 'Mature', 'Major economic center, highest costs'),
                    ('Rio de Janeiro', 1.10, 'High', 'Good', 'Medium', 'Mature', 'Major city, good infrastructure'),
                    ('Belo Horizonte', 1.00, 'Medium', 'Good', 'Low', 'Mature', 'Regional center, balanced costs'),
                    ('Brasília', 1.05, 'Medium', 'Good', 'Medium', 'Developing', 'Capital city, planned infrastructure'),
                    ('Salvador', 0.95, 'Medium', 'Fair', 'Medium', 'Developing', 'Regional center, lower costs'),
                    ('Porto Alegre', 1.03, 'Medium', 'Good', 'Low', 'Mature', 'Southern hub, stable market'),
                    ('Curitiba', 1.01, 'Medium', 'Good', 'Low', 'Mature', 'Planned city, efficient logistics'),
                    ('Fortaleza', 0.90, 'Low', 'Fair', 'High', 'Developing', 'Coastal city, emerging market'),
                    ('Recife', 0.92, 'Low', 'Fair', 'Medium', 'Developing', 'Regional center, growing market'),
                    ('Manaus', 1.20, 'Low', 'Poor', 'Very High', 'Emerging', 'Amazon region, high logistics costs'),
                    ('Goiânia', 0.98, 'Medium', 'Good', 'Medium', 'Developing', 'Agricultural region center'),
                    ('Campinas', 1.08, 'High', 'Excellent', 'Low', 'Mature', 'Industrial hub near São Paulo'),
                    ('Florianópolis', 1.06, 'Medium', 'Good', 'Medium', 'Mature', 'Technology center, island location'),
                    ('Campo Grande', 0.93, 'Low', 'Fair', 'High', 'Developing', 'Agricultural region, emerging market'),
                    ('João Pessoa', 0.88, 'Low', 'Fair', 'High', 'Developing', 'Coastal city, limited infrastructure')
                ]
                
                cursor.executemany('''
                    INSERT INTO regional_factors 
                    (location, cost_multiplier, labor_availability, material_availability, 
                     logistics_difficulty, market_maturity, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', regional_data)
                
                conn.commit()
                logger.info("Regional adjustment factors seeded successfully")
                
        except Exception as e:
            logger.error(f"Error seeding regional factors: {str(e)}")
            raise
    
    def _seed_material_references(self):
        """
        Seed material reference data and specifications
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create material references table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS material_references (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        category TEXT NOT NULL,
                        subcategory TEXT,
                        specification TEXT NOT NULL,
                        unit TEXT NOT NULL,
                        typical_unit_cost REAL,
                        min_cost REAL,
                        max_cost REAL,
                        supplier_grade TEXT,
                        nbr_standard TEXT,
                        environmental_rating TEXT,
                        lead_time_days INTEGER,
                        notes TEXT,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Clear existing references
                cursor.execute('DELETE FROM material_references')
                
                material_specs = [
                    # Concrete/Masonry
                    ('Concrete/Masonry', 'Concrete', 'Concrete C20 - fck=20MPa', 'm³', 160.0, 120.0, 200.0, 
                     'Standard', 'NBR 6118', 'B', 7, 'Standard structural concrete'),
                    ('Concrete/Masonry', 'Concrete', 'Concrete C25 - fck=25MPa', 'm³', 180.0, 140.0, 220.0, 
                     'Good', 'NBR 6118', 'B', 7, 'High strength concrete'),
                    ('Concrete/Masonry', 'Concrete', 'Concrete C30 - fck=30MPa', 'm³', 210.0, 170.0, 250.0, 
                     'Premium', 'NBR 6118', 'A', 10, 'High performance concrete'),
                    ('Concrete/Masonry', 'Masonry', 'Ceramic Block 14x19x29cm', 'm²', 45.0, 35.0, 55.0, 
                     'Standard', 'NBR 15270', 'C', 14, 'Standard masonry block'),
                    ('Concrete/Masonry', 'Masonry', 'Concrete Block 14x19x39cm', 'm²', 40.0, 30.0, 50.0, 
                     'Standard', 'NBR 6136', 'B', 10, 'Structural concrete block'),
                    
                    # Steel/Concrete
                    ('Steel/Concrete', 'Rebar', 'Steel Rebar CA-50 ϕ10mm', 'kg', 6.5, 5.5, 7.5, 
                     'Standard', 'NBR 7480', 'B', 21, 'Standard reinforcement steel'),
                    ('Steel/Concrete', 'Rebar', 'Steel Rebar CA-50 ϕ12.5mm', 'kg', 6.3, 5.3, 7.3, 
                     'Standard', 'NBR 7480', 'B', 21, 'Medium diameter rebar'),
                    ('Steel/Concrete', 'Structural', 'Steel Profile I 200mm', 'kg', 8.5, 7.0, 10.0, 
                     'Good', 'NBR 8800', 'A', 30, 'Structural steel profile'),
                    ('Steel/Concrete', 'Structural', 'Steel Profile H 300mm', 'kg', 9.2, 8.0, 10.5, 
                     'Premium', 'NBR 8800', 'A', 35, 'Heavy structural profile'),
                    
                    # Roofing Materials
                    ('Roofing Materials', 'Tiles', 'Ceramic Roof Tiles', 'm²', 35.0, 25.0, 45.0, 
                     'Standard', 'NBR 13310', 'C', 14, 'Traditional ceramic tiles'),
                    ('Roofing Materials', 'Tiles', 'Concrete Roof Tiles', 'm²', 30.0, 22.0, 38.0, 
                     'Standard', 'NBR 13858', 'B', 10, 'Concrete roof tiles'),
                    ('Roofing Materials', 'Metal', 'Galvanized Steel Sheet 0.5mm', 'm²', 28.0, 20.0, 36.0, 
                     'Good', 'NBR 7008', 'B', 7, 'Corrugated metal roofing'),
                    ('Roofing Materials', 'Waterproofing', 'Modified Bitumen Membrane', 'm²', 45.0, 35.0, 55.0, 
                     'Premium', 'NBR 9952', 'A', 14, 'High-performance waterproofing'),
                    
                    # Doors/Windows
                    ('Doors/Windows', 'Doors', 'Wooden Door 80x210cm', 'unit', 350.0, 250.0, 450.0, 
                     'Standard', 'NBR 15930', 'C', 21, 'Standard interior door'),
                    ('Doors/Windows', 'Doors', 'Steel Security Door 90x210cm', 'unit', 650.0, 500.0, 800.0, 
                     'Good', 'NBR 11742', 'B', 28, 'Security entrance door'),
                    ('Doors/Windows', 'Windows', 'Aluminum Window 120x100cm', 'unit', 420.0, 320.0, 520.0, 
                     'Standard', 'NBR 10821', 'B', 21, 'Standard aluminum window'),
                    ('Doors/Windows', 'Windows', 'PVC Window 150x120cm', 'unit', 580.0, 450.0, 710.0, 
                     'Good', 'NBR 10821', 'A', 28, 'Energy efficient PVC window'),
                    
                    # Finishes
                    ('Finishes', 'Flooring', 'Ceramic Floor Tiles 45x45cm', 'm²', 65.0, 50.0, 80.0, 
                     'Standard', 'NBR 13817', 'C', 14, 'Standard ceramic flooring'),
                    ('Finishes', 'Flooring', 'Porcelain Tiles 60x60cm', 'm²', 85.0, 70.0, 100.0, 
                     'Good', 'NBR 15463', 'B', 21, 'High-quality porcelain tiles'),
                    ('Finishes', 'Paint', 'Acrylic Paint - Internal', 'm²', 12.0, 8.0, 16.0, 
                     'Standard', 'NBR 11702', 'B', 7, 'Standard internal paint'),
                    ('Finishes', 'Paint', 'External Facade Paint', 'm²', 18.0, 14.0, 22.0, 
                     'Good', 'NBR 11702', 'A', 10, 'Weather-resistant external paint'),
                    
                    # MEP Systems
                    ('MEP Systems', 'Electrical', 'Electrical Installation Basic', 'm²', 85.0, 65.0, 105.0, 
                     'Standard', 'NBR 5410', 'B', 21, 'Basic electrical installation'),
                    ('MEP Systems', 'Electrical', 'Electrical Installation Premium', 'm²', 135.0, 110.0, 160.0, 
                     'Premium', 'NBR 5410', 'A', 28, 'Advanced electrical with automation'),
                    ('MEP Systems', 'Plumbing', 'Plumbing Installation Basic', 'm²', 75.0, 55.0, 95.0, 
                     'Standard', 'NBR 5626', 'B', 14, 'Basic plumbing installation'),
                    ('MEP Systems', 'HVAC', 'HVAC System Commercial', 'm²', 180.0, 140.0, 220.0, 
                     'Good', 'NBR 16401', 'A', 35, 'Commercial HVAC system'),
                ]
                
                cursor.executemany('''
                    INSERT INTO material_references 
                    (category, subcategory, specification, unit, typical_unit_cost, 
                     min_cost, max_cost, supplier_grade, nbr_standard, 
                     environmental_rating, lead_time_days, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', material_specs)
                
                conn.commit()
                logger.info("Material reference data seeded successfully")
                
        except Exception as e:
            logger.error(f"Error seeding material references: {str(e)}")
            raise
    
    def verify_seed_data(self):
        """
        Verify that seed data was inserted correctly
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check NBR standards
                cursor.execute('SELECT COUNT(*) FROM nbr_standards')
                nbr_count = cursor.fetchone()[0]
                
                # Check historical data
                cursor.execute('SELECT COUNT(*) FROM historical_data')
                historical_count = cursor.fetchone()[0]
                
                # Check regional factors
                cursor.execute('SELECT COUNT(*) FROM regional_factors')
                regional_count = cursor.fetchone()[0]
                
                # Check material references
                cursor.execute('SELECT COUNT(*) FROM material_references')
                material_count = cursor.fetchone()[0]
                
                verification_results = {
                    'nbr_standards': nbr_count,
                    'historical_data': historical_count,
                    'regional_factors': regional_count,
                    'material_references': material_count
                }
                
                logger.info(f"Seed data verification: {verification_results}")
                
                # Minimum expected counts
                expected_minimums = {
                    'nbr_standards': 20,
                    'historical_data': 1500,
                    'regional_factors': 10,
                    'material_references': 20
                }
                
                success = True
                for table, count in verification_results.items():
                    if count < expected_minimums[table]:
                        logger.error(f"Insufficient data in {table}: {count} < {expected_minimums[table]}")
                        success = False
                
                if success:
                    logger.info("All seed data verification checks passed!")
                else:
                    logger.error("Some seed data verification checks failed!")
                
                return success, verification_results
                
        except Exception as e:
            logger.error(f"Error verifying seed data: {str(e)}")
            return False, {}
    
    def clean_database(self):
        """
        Clean all seeded data (useful for re-seeding)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                tables_to_clean = [
                    'nbr_standards',
                    'historical_data', 
                    'regional_factors',
                    'material_references'
                ]
                
                for table in tables_to_clean:
                    try:
                        cursor.execute(f'DELETE FROM {table}')
                        logger.info(f"Cleaned table: {table}")
                    except sqlite3.Error as e:
                        logger.warning(f"Could not clean table {table}: {str(e)}")
                
                conn.commit()
                logger.info("Database cleaning completed")
                
        except Exception as e:
            logger.error(f"Error cleaning database: {str(e)}")
            raise

def main():
    """
    Main function to run the seeding process
    """
    try:
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        seeder = SeedDataManager()
        
        print("🌱 Starting Construction Budget Database Seeding...")
        print("=" * 50)
        
        # Clean existing data (optional)
        user_input = input("Clean existing seed data? (y/N): ").lower()
        if user_input == 'y':
            print("🧹 Cleaning existing data...")
            seeder.clean_database()
        
        # Seed all data
        print("📊 Seeding database with initial data...")
        seeder.seed_all_data()
        
        # Verify seeding
        print("✅ Verifying seed data...")
        success, results = seeder.verify_seed_data()
        
        if success:
            print("\n🎉 Database seeding completed successfully!")
            print("\nData Summary:")
            for table, count in results.items():
                print(f"  • {table.replace('_', ' ').title()}: {count:,} records")
        else:
            print("\n❌ Database seeding completed with warnings!")
            print("Please check the logs for details.")
        
        print("\n" + "=" * 50)
        print("The system is now ready with:")
        print("• NBR 12721 compliance standards")
        print("• Historical data for ML training")
        print("• Regional cost adjustment factors")
        print("• Material specification references")
        
    except Exception as e:
        print(f"\n❌ Error during seeding process: {str(e)}")
        logger.error(f"Seeding process failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
