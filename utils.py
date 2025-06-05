import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import logging
from datetime import datetime, timedelta
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_currency(value: float, currency: str = "R$") -> str:
    """
    Format currency values for display
    
    Args:
        value: The monetary value to format
        currency: Currency symbol (default: R$ for Brazilian Real)
    
    Returns:
        Formatted currency string
    """
    try:
        if value == 0:
            return f"{currency} 0,00"
        
        # Handle negative values
        is_negative = value < 0
        abs_value = abs(value)
        
        # Format with thousands separator and 2 decimal places
        if abs_value >= 1000000:
            # Millions
            formatted = f"{abs_value/1000000:.2f}M"
        elif abs_value >= 1000:
            # Thousands  
            formatted = f"{abs_value/1000:.1f}K"
        else:
            formatted = f"{abs_value:.2f}"
        
        # Add currency symbol and handle negative
        result = f"{currency} {formatted}"
        if is_negative:
            result = f"-{result}"
        
        return result
        
    except Exception as e:
        logger.warning(f"Error formatting currency: {str(e)}")
        return f"{currency} 0,00"

def calculate_mape(actual: List[float], predicted: List[float]) -> float:
    """
    Calculate Mean Absolute Percentage Error
    
    Args:
        actual: List of actual values
        predicted: List of predicted values
    
    Returns:
        MAPE as percentage
    """
    try:
        if len(actual) == 0 or len(predicted) == 0:
            return 100.0
        
        if len(actual) != len(predicted):
            logger.warning("Actual and predicted arrays have different lengths")
            min_len = min(len(actual), len(predicted))
            actual = actual[:min_len]
            predicted = predicted[:min_len]
        
        # Convert to numpy arrays
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        # Avoid division by zero
        mask = actual != 0
        if not np.any(mask):
            return 100.0
        
        # Calculate MAPE
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        
        return float(mape)
        
    except Exception as e:
        logger.error(f"Error calculating MAPE: {str(e)}")
        return 100.0

def calculate_accuracy_from_mape(mape: float) -> float:
    """
    Convert MAPE to accuracy percentage
    
    Args:
        mape: Mean Absolute Percentage Error
    
    Returns:
        Accuracy percentage (0-100)
    """
    try:
        accuracy = max(0, 100 - mape)
        return min(100, accuracy)  # Cap at 100%
    except:
        return 0.0

def get_material_categories() -> List[str]:
    """
    Get standard material categories used in the system
    
    Returns:
        List of material category names
    """
    return [
        'Concrete/Masonry',
        'Steel/Concrete', 
        'Roofing Materials',
        'Doors/Windows',
        'Finishes',
        'MEP Systems',
        'Miscellaneous'
    ]

def validate_project_data(project_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate project data for completeness and consistency
    
    Args:
        project_data: Dictionary containing project information
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    try:
        # Required fields
        required_fields = ['name', 'location', 'project_type']
        for field in required_fields:
            if not project_data.get(field):
                errors.append(f"Missing required field: {field}")
        
        # Validate project type
        valid_types = ['Residential', 'Commercial', 'Industrial', 'Infrastructure']
        if project_data.get('project_type') not in valid_types:
            errors.append(f"Invalid project type. Must be one of: {', '.join(valid_types)}")
        
        # Validate materials if present
        if 'materials' in project_data:
            materials = project_data['materials']
            if not isinstance(materials, dict):
                errors.append("Materials must be a dictionary")
            else:
                for material, quantity in materials.items():
                    if not isinstance(quantity, (int, float)) or quantity < 0:
                        errors.append(f"Invalid quantity for material {material}: {quantity}")
        
        # Validate costs if present
        cost_fields = ['total_budget', 'actual_cost']
        for field in cost_fields:
            if field in project_data:
                value = project_data[field]
                if not isinstance(value, (int, float)) or value < 0:
                    errors.append(f"Invalid {field}: {value}")
        
        return len(errors) == 0, errors
        
    except Exception as e:
        errors.append(f"Validation error: {str(e)}")
        return False, errors

def calculate_project_metrics(materials: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate various project metrics from materials data
    
    Args:
        materials: List of material dictionaries
    
    Returns:
        Dictionary of calculated metrics
    """
    try:
        if not materials:
            return {'total_cost': 0, 'material_count': 0, 'avg_unit_cost': 0}
        
        total_cost = sum(m.get('cost', 0) for m in materials)
        total_quantity = sum(m.get('quantity', 0) for m in materials)
        material_count = len(materials)
        
        # Calculate average unit cost
        unit_costs = [m.get('unit_cost', 0) for m in materials if m.get('unit_cost', 0) > 0]
        avg_unit_cost = np.mean(unit_costs) if unit_costs else 0
        
        # Calculate cost distribution
        cost_distribution = {}
        for material in materials:
            category = material.get('category', 'Miscellaneous')
            cost = material.get('cost', 0)
            cost_distribution[category] = cost_distribution.get(category, 0) + cost
        
        # Find most expensive category
        most_expensive = max(cost_distribution.items(), key=lambda x: x[1]) if cost_distribution else ('None', 0)
        
        return {
            'total_cost': total_cost,
            'total_quantity': total_quantity,
            'material_count': material_count,
            'avg_unit_cost': avg_unit_cost,
            'cost_distribution': cost_distribution,
            'most_expensive_category': most_expensive[0],
            'most_expensive_cost': most_expensive[1]
        }
        
    except Exception as e:
        logger.error(f"Error calculating project metrics: {str(e)}")
        return {'total_cost': 0, 'material_count': 0, 'avg_unit_cost': 0}

def generate_project_timeline(start_date: datetime, duration_months: int = 12) -> List[Dict[str, Any]]:
    """
    Generate a project timeline with milestones
    
    Args:
        start_date: Project start date
        duration_months: Project duration in months
    
    Returns:
        List of timeline milestones
    """
    try:
        timeline = []
        current_date = start_date
        
        # Standard construction milestones
        milestones = [
            {'name': 'Foundation', 'duration_weeks': 4, 'cost_percentage': 15},
            {'name': 'Structure', 'duration_weeks': 12, 'cost_percentage': 35},
            {'name': 'Roofing', 'duration_weeks': 3, 'cost_percentage': 8},
            {'name': 'MEP Rough-in', 'duration_weeks': 6, 'cost_percentage': 15},
            {'name': 'Finishes', 'duration_weeks': 8, 'cost_percentage': 20},
            {'name': 'Final Systems', 'duration_weeks': 3, 'cost_percentage': 7}
        ]
        
        cumulative_cost = 0
        
        for milestone in milestones:
            end_date = current_date + timedelta(weeks=milestone['duration_weeks'])
            cumulative_cost += milestone['cost_percentage']
            
            timeline.append({
                'milestone': milestone['name'],
                'start_date': current_date,
                'end_date': end_date,
                'duration_weeks': milestone['duration_weeks'],
                'cost_percentage': milestone['cost_percentage'],
                'cumulative_cost_percentage': cumulative_cost
            })
            
            current_date = end_date
        
        return timeline
        
    except Exception as e:
        logger.error(f"Error generating timeline: {str(e)}")
        return []

def clean_material_name(name: str) -> str:
    """
    Clean and standardize material names
    
    Args:
        name: Raw material name
    
    Returns:
        Cleaned material name
    """
    try:
        if not name:
            return 'Miscellaneous'
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', name.strip())
        
        # Standardize common variations
        standardizations = {
            r'concrete|concreto': 'Concrete',
            r'steel|aço|aco': 'Steel',
            r'masonry|alvenaria': 'Masonry',
            r'roof|telhado': 'Roofing',
            r'door|porta': 'Door',
            r'window|janela': 'Window',
            r'finish|acabamento': 'Finish',
            r'electrical|elétrico|eletrico': 'Electrical',
            r'plumbing|hidráulico|hidraulico': 'Plumbing'
        }
        
        for pattern, replacement in standardizations.items():
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
        
        # Capitalize properly
        cleaned = ' '.join(word.capitalize() for word in cleaned.split())
        
        return cleaned
        
    except Exception as e:
        logger.warning(f"Error cleaning material name: {str(e)}")
        return name or 'Miscellaneous'

def calculate_cost_variance(actual: float, predicted: float) -> Dict[str, float]:
    """
    Calculate cost variance metrics
    
    Args:
        actual: Actual cost
        predicted: Predicted cost
    
    Returns:
        Dictionary with variance metrics
    """
    try:
        if predicted == 0:
            return {
                'absolute_variance': actual,
                'percentage_variance': 100.0 if actual > 0 else 0.0,
                'is_over_budget': actual > 0
            }
        
        absolute_variance = actual - predicted
        percentage_variance = (absolute_variance / predicted) * 100
        is_over_budget = actual > predicted
        
        return {
            'absolute_variance': absolute_variance,
            'percentage_variance': percentage_variance,
            'is_over_budget': is_over_budget
        }
        
    except Exception as e:
        logger.error(f"Error calculating cost variance: {str(e)}")
        return {'absolute_variance': 0, 'percentage_variance': 0, 'is_over_budget': False}

def get_cost_trend(cost_data: List[Dict[str, Any]], window_size: int = 7) -> List[float]:
    """
    Calculate cost trend using moving average
    
    Args:
        cost_data: List of cost data points with 'date' and 'cost' keys
        window_size: Size of moving average window
    
    Returns:
        List of trend values
    """
    try:
        if len(cost_data) < window_size:
            return [d['cost'] for d in cost_data]
        
        # Sort by date
        sorted_data = sorted(cost_data, key=lambda x: x['date'])
        costs = [d['cost'] for d in sorted_data]
        
        # Calculate moving average
        trend = []
        for i in range(len(costs)):
            if i < window_size - 1:
                trend.append(costs[i])
            else:
                window_avg = np.mean(costs[i-window_size+1:i+1])
                trend.append(window_avg)
        
        return trend
        
    except Exception as e:
        logger.error(f"Error calculating cost trend: {str(e)}")
        return []

def export_data_to_csv(data: List[Dict[str, Any]], filename: str) -> str:
    """
    Export data to CSV format
    
    Args:
        data: List of data dictionaries
        filename: Output filename
    
    Returns:
        Path to exported file or error message
    """
    try:
        if not data:
            return "No data to export"
        
        df = pd.DataFrame(data)
        
        # Clean column names
        df.columns = [col.replace('_', ' ').title() for col in df.columns]
        
        # Export to CSV
        output_path = f"exports/{filename}"
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error exporting data: {str(e)}")
        return f"Export error: {str(e)}"

def validate_ifc_file(file_path: str) -> Dict[str, Any]:
    """
    Validate IFC file format and basic structure
    
    Args:
        file_path: Path to IFC file
    
    Returns:
        Validation result dictionary
    """
    try:
        if not file_path.lower().endswith('.ifc'):
            return {'valid': False, 'error': 'File must have .ifc extension'}
        
        # Check if file exists and has reasonable size
        import os
        if not os.path.exists(file_path):
            return {'valid': False, 'error': 'File does not exist'}
        
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return {'valid': False, 'error': 'File is empty'}
        
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            return {'valid': False, 'error': 'File too large (max 100MB)'}
        
        # Basic IFC format validation
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline().strip()
            if not first_line.startswith('ISO-10303'):
                return {'valid': False, 'error': 'Invalid IFC format'}
        
        return {
            'valid': True,
            'file_size': file_size,
            'size_mb': file_size / (1024 * 1024)
        }
        
    except Exception as e:
        return {'valid': False, 'error': f'Validation error: {str(e)}'}

def get_system_health() -> Dict[str, Any]:
    """
    Get system health metrics
    
    Returns:
        Dictionary with system health information
    """
    try:
        import psutil
        import sqlite3
        
        # Memory usage
        memory = psutil.virtual_memory()
        
        # Disk usage
        disk = psutil.disk_usage('.')
        
        # Database status
        try:
            conn = sqlite3.connect('construction_budget.db')
            conn.execute('SELECT 1')
            conn.close()
            db_status = 'healthy'
        except:
            db_status = 'error'
        
        return {
            'memory_used_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'disk_used_percent': (disk.used / disk.total) * 100,
            'disk_free_gb': disk.free / (1024**3),
            'database_status': db_status,
            'timestamp': datetime.now().isoformat()
        }
        
    except ImportError:
        # psutil not available
        return {
            'memory_used_percent': 0,
            'memory_available_gb': 0,
            'disk_used_percent': 0,
            'disk_free_gb': 0,
            'database_status': 'unknown',
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
