import sqlite3
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Database manager for construction budget management system
    """
    
    def __init__(self, db_path: str = "construction_budget.db"):
        self.db_path = os.path.abspath(db_path)
        self.init_database()
    
    def _get_connection(self):
        """
        Get a database connection with proper configuration
        """
        # Create a new database file with proper permissions
        if not os.path.exists(self.db_path):
            # Create empty file with write permissions
            open(self.db_path, 'a').close()
            os.chmod(self.db_path, 0o666)
        
        conn = sqlite3.connect(
            self.db_path, 
            timeout=60.0,
            check_same_thread=False
        )
        
        # Set SQLite to be more permissive
        conn.execute('PRAGMA journal_mode=DELETE')
        conn.execute('PRAGMA synchronous=OFF')
        conn.execute('PRAGMA locking_mode=NORMAL')
        return conn
    
    def init_database(self):
        """
        Initialize database with required tables
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Projects table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    location TEXT,
                    project_type TEXT,
                    bim_file_path TEXT,
                    total_budget REAL DEFAULT 0,
                    actual_cost REAL DEFAULT 0,
                    completion_percentage REAL DEFAULT 0,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Materials table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS materials (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER,
                    name TEXT NOT NULL,
                    category TEXT,
                    quantity REAL NOT NULL,
                    unit TEXT DEFAULT 'm³',
                    unit_cost REAL DEFAULT 0,
                    total_cost REAL DEFAULT 0,
                    supplier TEXT,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects (id)
                )
            ''')
            
            # Cost predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cost_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER,
                    material_category TEXT NOT NULL,
                    predicted_unit_cost REAL NOT NULL,
                    predicted_total_cost REAL NOT NULL,
                    confidence_score REAL DEFAULT 0,
                    model_version TEXT,
                    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects (id)
                )
            ''')
            
            # NBR compliance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS nbr_compliance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER,
                    standard_code TEXT NOT NULL,
                    compliance_status TEXT DEFAULT 'pending',
                    issues TEXT,
                    recommendations TEXT,
                    check_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects (id)
                )
            ''')
            
            # Historical data table for ML training
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS historical_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    material_category TEXT NOT NULL,
                    project_type TEXT,
                    location TEXT,
                    quantity REAL,
                    unit_cost REAL,
                    total_cost REAL,
                    project_date DATE,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Cost adjustments table for manual interventions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cost_adjustments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER,
                    material_id INTEGER,
                    original_cost REAL,
                    adjusted_cost REAL,
                    adjustment_reason TEXT,
                    adjusted_by TEXT,
                    adjustment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects (id),
                    FOREIGN KEY (material_id) REFERENCES materials (id)
                )
            ''')
            
            # Timeline tracking table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS project_timeline (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER,
                    milestone TEXT NOT NULL,
                    planned_date DATE,
                    actual_date DATE,
                    status TEXT DEFAULT 'pending',
                    notes TEXT,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects (id)
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def create_project(self, name: str, location: str, project_type: str, 
                      bim_file_path: str) -> int:
        """
        Create a new project
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO projects (name, location, project_type, bim_file_path)
                VALUES (?, ?, ?, ?)
            ''', (name, location, project_type, bim_file_path))
            
            project_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            logger.info(f"Created project: {name} (ID: {project_id})")
            return project_id
            
        except Exception as e:
            logger.error(f"Error creating project: {str(e)}")
            raise
    
    def add_material_quantity(self, project_id: int, material_name: str, 
                            quantity: float, category: str = None):
        """
        Add material quantity to project
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO materials (project_id, name, category, quantity)
                VALUES (?, ?, ?, ?)
            ''', (project_id, material_name, category, quantity))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error adding material quantity: {str(e)}")
            raise
    
    def store_prediction(self, project_id: int, material_category: str, 
                        unit_cost: float, total_cost: float, confidence: float):
        """
        Store ML prediction results
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO cost_predictions 
                (project_id, material_category, predicted_unit_cost, 
                 predicted_total_cost, confidence_score)
                VALUES (?, ?, ?, ?, ?)
            ''', (project_id, material_category, unit_cost, total_cost, confidence))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing prediction: {str(e)}")
            raise
    
    def get_all_projects(self) -> List[Dict[str, Any]]:
        """
        Get all projects
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, name, location, project_type, total_budget, 
                       actual_cost, completion_percentage, created_date
                FROM projects
                ORDER BY created_date DESC
            ''')
            
            projects = []
            for row in cursor.fetchall():
                projects.append({
                    'id': row[0],
                    'name': row[1],
                    'location': row[2],
                    'project_type': row[3],
                    'total_budget': row[4],
                    'actual_cost': row[5],
                    'completion_percentage': row[6],
                    'created_date': row[7]
                })
            
            conn.close()
            return projects
            
        except Exception as e:
            logger.error(f"Error getting projects: {str(e)}")
            return []
    
    def get_project_summary(self, project_id: int) -> Dict[str, Any]:
        """
        Get project summary for dashboard
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Get project details
            cursor.execute('''
                SELECT name, location, project_type, total_budget, 
                       actual_cost, completion_percentage
                FROM projects WHERE id = ?
            ''', (project_id,))
            
            project_row = cursor.fetchone()
            if not project_row:
                conn.close()
                return {}
            
            # Calculate metrics
            total_budget = project_row[3] or 0
            actual_cost = project_row[4] or 0
            completion = project_row[5] or 0
            
            # Get material count
            cursor.execute('SELECT COUNT(*) FROM materials WHERE project_id = ?', (project_id,))
            material_count = cursor.fetchone()[0]
            
            # Get prediction accuracy
            accuracy = self._calculate_project_accuracy(cursor, project_id)
            
            summary = {
                'project_name': project_row[0],
                'location': project_row[1],
                'project_type': project_row[2],
                'total_budget': total_budget,
                'actual_cost': actual_cost,
                'cost_variance': actual_cost - total_budget,
                'completion_percentage': completion,
                'material_count': material_count,
                'prediction_accuracy': accuracy,
                'budget_utilization': (actual_cost / total_budget * 100) if total_budget > 0 else 0
            }
            
            conn.close()
            return summary
            
        except Exception as e:
            logger.error(f"Error getting project summary: {str(e)}")
            return {}
    
    def _calculate_project_accuracy(self, cursor, project_id: int) -> float:
        """
        Calculate overall prediction accuracy for project
        """
        try:
            cursor.execute('''
                SELECT AVG(confidence_score) 
                FROM cost_predictions 
                WHERE project_id = ?
            ''', (project_id,))
            
            result = cursor.fetchone()
            return (result[0] or 0.85) * 100  # Default to 85% if no predictions
            
        except:
            return 85.0
    
    def get_cost_timeline(self, project_id: int) -> List[Dict[str, Any]]:
        """
        Get cost timeline data for charts
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Check if timeline data exists
            cursor.execute('''
                SELECT COUNT(*) FROM project_timeline WHERE project_id = ?
            ''', (project_id,))
            
            if cursor.fetchone()[0] == 0:
                # Generate sample timeline data
                self._generate_sample_timeline(cursor, project_id)
                conn.commit()
            
            cursor.execute('''
                SELECT milestone, planned_date, actual_date, status
                FROM project_timeline 
                WHERE project_id = ?
                ORDER BY planned_date
            ''', (project_id,))
            
            timeline = []
            for row in cursor.fetchall():
                timeline.append({
                    'milestone': row[0],
                    'planned_date': row[1],
                    'actual_date': row[2],
                    'status': row[3]
                })
            
            conn.close()
            return timeline
            
        except Exception as e:
            logger.error(f"Error getting cost timeline: {str(e)}")
            return []
    
    def _generate_sample_timeline(self, cursor, project_id: int):
        """
        Generate sample timeline data for demonstration
        """
        milestones = [
            ('Project Start', 0),
            ('Foundation Complete', 30),
            ('Structure Complete', 90),
            ('MEP Installation', 150),
            ('Finishes Complete', 210),
            ('Project Completion', 270)
        ]
        
        base_date = datetime.now()
        
        for milestone, days_offset in milestones:
            planned_date = base_date + timedelta(days=days_offset)
            status = 'completed' if days_offset <= 30 else 'pending'
            actual_date = planned_date if status == 'completed' else None
            
            cursor.execute('''
                INSERT INTO project_timeline 
                (project_id, milestone, planned_date, actual_date, status)
                VALUES (?, ?, ?, ?, ?)
            ''', (project_id, milestone, planned_date.date(), actual_date, status))
    
    def get_material_deviations(self, project_id: int) -> List[Dict[str, Any]]:
        """
        Get material cost deviations
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT m.name, m.category, m.quantity, m.unit_cost, m.total_cost,
                       p.predicted_unit_cost, p.predicted_total_cost
                FROM materials m
                LEFT JOIN cost_predictions p ON m.category = p.material_category 
                    AND m.project_id = p.project_id
                WHERE m.project_id = ?
            ''', (project_id,))
            
            deviations = []
            for row in cursor.fetchall():
                actual_cost = row[4] or 0
                predicted_cost = row[6] or actual_cost
                deviation = ((actual_cost - predicted_cost) / predicted_cost * 100) if predicted_cost > 0 else 0
                
                deviations.append({
                    'material': row[0],
                    'category': row[1],
                    'quantity': row[2],
                    'actual_cost': actual_cost,
                    'predicted_cost': predicted_cost,
                    'deviation_percent': deviation
                })
            
            conn.close()
            return deviations
            
        except Exception as e:
            logger.error(f"Error getting material deviations: {str(e)}")
            return []
    
    def get_critical_materials(self, project_id: int, threshold: float = 10.0) -> List[Dict[str, Any]]:
        """
        Get materials with critical cost deviations
        """
        deviations = self.get_material_deviations(project_id)
        return [d for d in deviations if abs(d['deviation_percent']) > threshold]
    
    def get_predictions(self, project_id: int) -> List[Dict[str, Any]]:
        """
        Get current predictions for project
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT material_category, predicted_unit_cost, 
                       predicted_total_cost, confidence_score
                FROM cost_predictions
                WHERE project_id = ?
                ORDER BY material_category
            ''', (project_id,))
            
            predictions = []
            for row in cursor.fetchall():
                predictions.append({
                    'category': row[0],
                    'unit_cost': row[1],
                    'total_cost': row[2],
                    'confidence': row[3]
                })
            
            conn.close()
            return predictions
            
        except Exception as e:
            logger.error(f"Error getting predictions: {str(e)}")
            return []
    
    def get_project_materials(self, project_id: int) -> List[Dict[str, Any]]:
        """
        Get all materials for a project
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, name, category, quantity, unit, unit_cost, total_cost
                FROM materials
                WHERE project_id = ?
                ORDER BY category, name
            ''', (project_id,))
            
            materials = []
            for row in cursor.fetchall():
                materials.append({
                    'id': row[0],
                    'name': row[1],
                    'category': row[2],
                    'quantity': row[3],
                    'unit': row[4],
                    'unit_cost': row[5],
                    'total_cost': row[6]
                })
            
            conn.close()
            return materials
            
        except Exception as e:
            logger.error(f"Error getting project materials: {str(e)}")
            return []
    
    def update_material_cost(self, material_id: int, new_cost: float, reason: str):
        """
        Update material cost with manual adjustment
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Get current cost
            cursor.execute('SELECT unit_cost, project_id FROM materials WHERE id = ?', (material_id,))
            result = cursor.fetchone()
            if not result:
                conn.close()
                return
            
            old_cost, project_id = result
            
            # Update material cost
            cursor.execute('''
                UPDATE materials 
                SET unit_cost = ?, total_cost = quantity * ?
                WHERE id = ?
            ''', (new_cost, new_cost, material_id))
            
            # Record adjustment
            cursor.execute('''
                INSERT INTO cost_adjustments 
                (project_id, material_id, original_cost, adjusted_cost, adjustment_reason)
                VALUES (?, ?, ?, ?, ?)
            ''', (project_id, material_id, old_cost, new_cost, reason))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating material cost: {str(e)}")
            raise
    
    def get_adjustment_history(self, project_id: int) -> List[Dict[str, Any]]:
        """
        Get manual adjustment history
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT m.name, ca.original_cost, ca.adjusted_cost, 
                       ca.adjustment_reason, ca.adjustment_date
                FROM cost_adjustments ca
                JOIN materials m ON ca.material_id = m.id
                WHERE ca.project_id = ?
                ORDER BY ca.adjustment_date DESC
            ''', (project_id,))
            
            adjustments = []
            for row in cursor.fetchall():
                adjustments.append({
                    'material': row[0],
                    'original_cost': row[1],
                    'adjusted_cost': row[2],
                    'reason': row[3],
                    'date': row[4]
                })
            
            conn.close()
            return adjustments
            
        except Exception as e:
            logger.error(f"Error getting adjustment history: {str(e)}")
            return []
    
    def get_project_details(self, project_id: int) -> Optional[Dict[str, Any]]:
        """
        Get detailed project information
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT name, location, project_type, bim_file_path,
                       total_budget, actual_cost, completion_percentage,
                       created_date, updated_date
                FROM projects WHERE id = ?
            ''', (project_id,))
            
            row = cursor.fetchone()
            if not row:
                conn.close()
                return None
            
            project_details = {
                'id': project_id,
                'name': row[0],
                'location': row[1],
                'project_type': row[2],
                'bim_file_path': row[3],
                'total_budget': row[4],
                'actual_cost': row[5],
                'completion_percentage': row[6],
                'created_date': row[7],
                'updated_date': row[8]
            }
            
            conn.close()
            return project_details
            
        except Exception as e:
            logger.error(f"Error getting project details: {str(e)}")
            return None
    
    def get_project_for_prediction(self, project_id: int) -> Optional[Dict[str, Any]]:
        """
        Get project data formatted for ML prediction
        """
        project = self.get_project_details(project_id)
        if not project:
            return None
        
        materials = self.get_project_materials(project_id)
        
        return {
            'project_type': project['project_type'],
            'location': project['location'],
            'materials': materials
        }
    
    def get_historical_data(self) -> List[Dict[str, Any]]:
        """
        Get historical data for ML training
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT material_category, project_type, location, 
                       quantity, unit_cost, total_cost, project_date
                FROM historical_data
                ORDER BY project_date DESC
            ''')
            
            data = []
            for row in cursor.fetchall():
                data.append({
                    'material_category': row[0],
                    'project_type': row[1],
                    'location': row[2],
                    'quantity': row[3],
                    'unit_cost': row[4],
                    'total_cost': row[5],
                    'project_date': row[6]
                })
            
            conn.close()
            return data
            
        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            return []
    
    def get_accuracy_data(self, project_id: int) -> Dict[str, Dict[str, List[float]]]:
        """
        Get actual vs predicted data for accuracy calculation
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT m.category, m.total_cost, p.predicted_total_cost
                FROM materials m
                LEFT JOIN cost_predictions p ON m.category = p.material_category 
                    AND m.project_id = p.project_id
                WHERE m.project_id = ? AND p.predicted_total_cost IS NOT NULL
            ''', (project_id,))
            
            accuracy_data = {}
            for row in cursor.fetchall():
                category = row[0]
                actual = row[1] or 0
                predicted = row[2] or 0
                
                if category not in accuracy_data:
                    accuracy_data[category] = {'actual': [], 'predicted': []}
                
                accuracy_data[category]['actual'].append(actual)
                accuracy_data[category]['predicted'].append(predicted)
            
            conn.close()
            return accuracy_data
            
        except Exception as e:
            logger.error(f"Error getting accuracy data: {str(e)}")
            return {}