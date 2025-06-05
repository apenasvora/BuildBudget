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
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """
        Initialize database with required tables
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
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
                        quantity REAL,
                        unit_cost REAL,
                        total_cost REAL,
                        created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (project_id) REFERENCES projects (id)
                    )
                ''')
                
                # Predictions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        project_id INTEGER,
                        material_category TEXT,
                        predicted_unit_cost REAL,
                        predicted_total_cost REAL,
                        confidence_level REAL,
                        prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (project_id) REFERENCES projects (id)
                    )
                ''')
                
                # Cost tracking table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS cost_tracking (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        project_id INTEGER,
                        date DATE,
                        predicted_cost REAL,
                        actual_cost REAL,
                        cumulative_predicted REAL,
                        cumulative_actual REAL,
                        created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (project_id) REFERENCES projects (id)
                    )
                ''')
                
                # Manual adjustments table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS manual_adjustments (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        project_id INTEGER,
                        material_id INTEGER,
                        old_cost REAL,
                        new_cost REAL,
                        reason TEXT,
                        adjusted_by TEXT,
                        adjustment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (project_id) REFERENCES projects (id),
                        FOREIGN KEY (material_id) REFERENCES materials (id)
                    )
                ''')
                
                # Historical data table for ML training
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS historical_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        material_category TEXT,
                        project_type TEXT,
                        location TEXT,
                        quantity REAL,
                        unit_cost REAL,
                        total_cost REAL,
                        project_date DATE,
                        created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # NBR compliance table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS nbr_compliance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        project_id INTEGER,
                        standard_code TEXT,
                        requirement TEXT,
                        status TEXT,
                        notes TEXT,
                        checked_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (project_id) REFERENCES projects (id)
                    )
                ''')
                
                conn.commit()
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
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO projects (name, location, project_type, bim_file_path)
                    VALUES (?, ?, ?, ?)
                ''', (name, location, project_type, bim_file_path))
                
                project_id = cursor.lastrowid
                conn.commit()
                
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
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO materials (project_id, name, category, quantity, unit_cost, total_cost)
                    VALUES (?, ?, ?, ?, 0, 0)
                ''', (project_id, material_name, category or material_name, quantity))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error adding material quantity: {str(e)}")
            raise
    
    def store_prediction(self, project_id: int, material_category: str, 
                        unit_cost: float, total_cost: float, confidence: float):
        """
        Store ML prediction results
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete existing prediction for this material
                cursor.execute('''
                    DELETE FROM predictions 
                    WHERE project_id = ? AND material_category = ?
                ''', (project_id, material_category))
                
                # Insert new prediction
                cursor.execute('''
                    INSERT INTO predictions 
                    (project_id, material_category, predicted_unit_cost, 
                     predicted_total_cost, confidence_level)
                    VALUES (?, ?, ?, ?, ?)
                ''', (project_id, material_category, unit_cost, total_cost, confidence))
                
                # Update material costs
                cursor.execute('''
                    UPDATE materials 
                    SET unit_cost = ?, total_cost = ?
                    WHERE project_id = ? AND (name = ? OR category = ?)
                ''', (unit_cost, total_cost, project_id, material_category, material_category))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error storing prediction: {str(e)}")
            raise
    
    def get_all_projects(self) -> List[Dict[str, Any]]:
        """
        Get all projects
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
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
                        'type': row[3],
                        'total_budget': row[4],
                        'actual_cost': row[5],
                        'completion_percentage': row[6],
                        'created_date': row[7]
                    })
                
                return projects
                
        except Exception as e:
            logger.error(f"Error getting projects: {str(e)}")
            return []
    
    def get_project_summary(self, project_id: int) -> Dict[str, Any]:
        """
        Get project summary for dashboard
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get project basic info
                cursor.execute('''
                    SELECT total_budget, actual_cost, completion_percentage
                    FROM projects WHERE id = ?
                ''', (project_id,))
                
                project_row = cursor.fetchone()
                if not project_row:
                    return {}
                
                total_budget, actual_cost, completion = project_row
                
                # Calculate total predicted cost from materials
                cursor.execute('''
                    SELECT SUM(total_cost) FROM materials WHERE project_id = ?
                ''', (project_id,))
                
                predicted_total = cursor.fetchone()[0] or 0
                
                # Update project total budget if not set
                if total_budget == 0 and predicted_total > 0:
                    cursor.execute('''
                        UPDATE projects SET total_budget = ? WHERE id = ?
                    ''', (predicted_total, project_id))
                    total_budget = predicted_total
                    conn.commit()
                
                # Calculate variances
                budget_change = predicted_total - total_budget if total_budget > 0 else 0
                cost_variance = ((actual_cost - predicted_total) / predicted_total * 100) if predicted_total > 0 else 0
                
                # Get prediction accuracy
                accuracy = self._calculate_project_accuracy(cursor, project_id)
                
                return {
                    'total_budget': total_budget,
                    'actual_cost': actual_cost,
                    'completion_percentage': completion,
                    'budget_change': budget_change,
                    'cost_variance': cost_variance,
                    'progress_change': 0,  # This would need historical tracking
                    'prediction_accuracy': accuracy
                }
                
        except Exception as e:
            logger.error(f"Error getting project summary: {str(e)}")
            return {}
    
    def _calculate_project_accuracy(self, cursor, project_id: int) -> float:
        """
        Calculate overall prediction accuracy for project
        """
        try:
            cursor.execute('''
                SELECT AVG(confidence_level) FROM predictions WHERE project_id = ?
            ''', (project_id,))
            
            result = cursor.fetchone()
            return result[0] if result and result[0] else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating accuracy: {str(e)}")
            return 0.0
    
    def get_cost_timeline(self, project_id: int) -> List[Dict[str, Any]]:
        """
        Get cost timeline data for charts
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if we have tracking data
                cursor.execute('''
                    SELECT COUNT(*) FROM cost_tracking WHERE project_id = ?
                ''', (project_id,))
                
                if cursor.fetchone()[0] == 0:
                    # Generate sample timeline data
                    self._generate_sample_timeline(cursor, project_id)
                    conn.commit()
                
                # Get timeline data
                cursor.execute('''
                    SELECT date, predicted_cost, actual_cost, 
                           cumulative_predicted, cumulative_actual
                    FROM cost_tracking 
                    WHERE project_id = ?
                    ORDER BY date
                ''', (project_id,))
                
                timeline = []
                for row in cursor.fetchall():
                    timeline.append({
                        'date': row[0],
                        'predicted_cost': row[1],
                        'actual_cost': row[2],
                        'cumulative_predicted': row[3],
                        'cumulative_actual': row[4]
                    })
                
                return timeline
                
        except Exception as e:
            logger.error(f"Error getting cost timeline: {str(e)}")
            return []
    
    def _generate_sample_timeline(self, cursor, project_id: int):
        """
        Generate sample timeline data for demonstration
        """
        try:
            # Get project total cost
            cursor.execute('''
                SELECT SUM(total_cost) FROM materials WHERE project_id = ?
            ''', (project_id,))
            
            total_cost = cursor.fetchone()[0] or 100000
            
            # Generate 30 days of sample data
            start_date = datetime.now() - timedelta(days=30)
            
            cumulative_predicted = 0
            cumulative_actual = 0
            
            for i in range(30):
                current_date = start_date + timedelta(days=i)
                
                # Progressive cost accumulation
                daily_predicted = total_cost * (i + 1) / 30
                daily_actual = daily_predicted * (0.9 + 0.2 * (i / 30))  # Some variance
                
                daily_increment_pred = daily_predicted - cumulative_predicted
                daily_increment_actual = daily_actual - cumulative_actual
                
                cumulative_predicted = daily_predicted
                cumulative_actual = daily_actual
                
                cursor.execute('''
                    INSERT INTO cost_tracking 
                    (project_id, date, predicted_cost, actual_cost, 
                     cumulative_predicted, cumulative_actual)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (project_id, current_date.date(), daily_increment_pred, 
                      daily_increment_actual, cumulative_predicted, cumulative_actual))
                
        except Exception as e:
            logger.warning(f"Error generating sample timeline: {str(e)}")
    
    def get_material_deviations(self, project_id: int) -> List[Dict[str, Any]]:
        """
        Get material cost deviations
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT m.category, m.total_cost, p.predicted_total_cost
                    FROM materials m
                    LEFT JOIN predictions p ON m.project_id = p.project_id 
                        AND (m.category = p.material_category OR m.name = p.material_category)
                    WHERE m.project_id = ?
                ''', (project_id,))
                
                deviations = []
                for row in cursor.fetchall():
                    category, actual_cost, predicted_cost = row
                    
                    if predicted_cost and predicted_cost > 0:
                        deviation = ((actual_cost - predicted_cost) / predicted_cost) * 100
                    else:
                        deviation = 0
                    
                    deviations.append({
                        'material_category': category,
                        'actual_cost': actual_cost,
                        'predicted_cost': predicted_cost,
                        'deviation_percentage': deviation
                    })
                
                return deviations
                
        except Exception as e:
            logger.error(f"Error getting material deviations: {str(e)}")
            return []
    
    def get_critical_materials(self, project_id: int, threshold: float = 10.0) -> List[Dict[str, Any]]:
        """
        Get materials with critical cost deviations
        """
        try:
            deviations = self.get_material_deviations(project_id)
            
            critical = []
            for deviation in deviations:
                if abs(deviation['deviation_percentage']) > threshold:
                    critical.append({
                        'name': deviation['material_category'],
                        'deviation': deviation['deviation_percentage'],
                        'actual_cost': deviation['actual_cost'],
                        'predicted_cost': deviation['predicted_cost']
                    })
            
            # Sort by deviation magnitude
            critical.sort(key=lambda x: abs(x['deviation']), reverse=True)
            
            return critical
            
        except Exception as e:
            logger.error(f"Error getting critical materials: {str(e)}")
            return []
    
    def get_predictions(self, project_id: int) -> List[Dict[str, Any]]:
        """
        Get current predictions for project
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT material_category, predicted_unit_cost, 
                           predicted_total_cost, confidence_level
                    FROM predictions 
                    WHERE project_id = ?
                    ORDER BY predicted_total_cost DESC
                ''', (project_id,))
                
                predictions = []
                for row in cursor.fetchall():
                    predictions.append({
                        'material_category': row[0],
                        'predicted_unit_cost': row[1],
                        'predicted_cost': row[2],
                        'confidence_level': row[3]
                    })
                
                return predictions
                
        except Exception as e:
            logger.error(f"Error getting predictions: {str(e)}")
            return []
    
    def get_project_materials(self, project_id: int) -> List[Dict[str, Any]]:
        """
        Get all materials for a project
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT id, name, category, quantity, unit_cost, total_cost
                    FROM materials 
                    WHERE project_id = ?
                    ORDER BY total_cost DESC
                ''', (project_id,))
                
                materials = []
                for row in cursor.fetchall():
                    materials.append({
                        'id': row[0],
                        'name': row[1],
                        'category': row[2],
                        'quantity': row[3],
                        'unit_cost': row[4],
                        'cost': row[5]
                    })
                
                return materials
                
        except Exception as e:
            logger.error(f"Error getting project materials: {str(e)}")
            return []
    
    def update_material_cost(self, material_id: int, new_cost: float, reason: str):
        """
        Update material cost with manual adjustment
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get current cost
                cursor.execute('''
                    SELECT project_id, total_cost FROM materials WHERE id = ?
                ''', (material_id,))
                
                result = cursor.fetchone()
                if not result:
                    raise ValueError(f"Material {material_id} not found")
                
                project_id, old_cost = result
                
                # Update material cost
                cursor.execute('''
                    UPDATE materials SET total_cost = ? WHERE id = ?
                ''', (new_cost, material_id))
                
                # Record adjustment
                cursor.execute('''
                    INSERT INTO manual_adjustments 
                    (project_id, material_id, old_cost, new_cost, reason)
                    VALUES (?, ?, ?, ?, ?)
                ''', (project_id, material_id, old_cost, new_cost, reason))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error updating material cost: {str(e)}")
            raise
    
    def get_adjustment_history(self, project_id: int) -> List[Dict[str, Any]]:
        """
        Get manual adjustment history
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT ma.adjustment_date, m.name, ma.old_cost, 
                           ma.new_cost, ma.reason
                    FROM manual_adjustments ma
                    JOIN materials m ON ma.material_id = m.id
                    WHERE ma.project_id = ?
                    ORDER BY ma.adjustment_date DESC
                ''', (project_id,))
                
                history = []
                for row in cursor.fetchall():
                    history.append({
                        'date': row[0],
                        'material': row[1],
                        'old_cost': row[2],
                        'new_cost': row[3],
                        'reason': row[4]
                    })
                
                return history
                
        except Exception as e:
            logger.error(f"Error getting adjustment history: {str(e)}")
            return []
    
    def get_project_details(self, project_id: int) -> Optional[Dict[str, Any]]:
        """
        Get detailed project information
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT name, location, project_type, total_budget, 
                           actual_cost, completion_percentage, created_date
                    FROM projects WHERE id = ?
                ''', (project_id,))
                
                result = cursor.fetchone()
                if not result:
                    return None
                
                total_budget, actual_cost = result[3], result[4]
                variance = 0
                if total_budget > 0:
                    variance = ((actual_cost - total_budget) / total_budget) * 100
                
                return {
                    'name': result[0],
                    'location': result[1],
                    'type': result[2],
                    'total_budget': total_budget,
                    'actual_cost': actual_cost,
                    'completion': result[5],
                    'variance': variance,
                    'created_date': result[6]
                }
                
        except Exception as e:
            logger.error(f"Error getting project details: {str(e)}")
            return None
    
    def get_project_for_prediction(self, project_id: int) -> Optional[Dict[str, Any]]:
        """
        Get project data formatted for ML prediction
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get project info
                cursor.execute('''
                    SELECT name, location, project_type FROM projects WHERE id = ?
                ''', (project_id,))
                
                project_result = cursor.fetchone()
                if not project_result:
                    return None
                
                # Get materials
                cursor.execute('''
                    SELECT category, quantity FROM materials WHERE project_id = ?
                ''', (project_id,))
                
                materials = {}
                for row in cursor.fetchall():
                    category, quantity = row
                    materials[category] = quantity
                
                return {
                    'project_name': project_result[0],
                    'location': project_result[1],
                    'project_type': project_result[2],
                    'materials': materials
                }
                
        except Exception as e:
            logger.error(f"Error getting project for prediction: {str(e)}")
            return None
    
    def get_historical_data(self) -> List[Dict[str, Any]]:
        """
        Get historical data for ML training
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT material_category, project_type, location, 
                           quantity, unit_cost, project_date
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
                        'project_date': row[5]
                    })
                
                return data
                
        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            return []
    
    def get_accuracy_data(self, project_id: int) -> Dict[str, Dict[str, List[float]]]:
        """
        Get actual vs predicted data for accuracy calculation
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT m.category, m.total_cost, p.predicted_total_cost
                    FROM materials m
                    JOIN predictions p ON m.project_id = p.project_id 
                        AND (m.category = p.material_category OR m.name = p.material_category)
                    WHERE m.project_id = ?
                ''', (project_id,))
                
                accuracy_data = {}
                for row in cursor.fetchall():
                    category, actual, predicted = row
                    
                    if category not in accuracy_data:
                        accuracy_data[category] = {'actual': [], 'predicted': []}
                    
                    if actual and predicted:
                        accuracy_data[category]['actual'].append(actual)
                        accuracy_data[category]['predicted'].append(predicted)
                
                return accuracy_data
                
        except Exception as e:
            logger.error(f"Error getting accuracy data: {str(e)}")
            return {}
