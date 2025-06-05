import logging
from typing import Dict, List, Any
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NBRValidator:
    """
    NBR 12721 (Brazilian Construction Standards) validator
    """
    
    def __init__(self):
        self.standards = self._load_nbr_standards()
        self.tolerance_thresholds = {
            'unit_cost_variance': 15.0,  # 15% tolerance for unit costs
            'material_ratio': 10.0,      # 10% tolerance for material ratios
            'total_cost_variance': 12.0   # 12% tolerance for total costs
        }
    
    def _load_nbr_standards(self) -> Dict[str, Any]:
        """
        Load NBR 12721 standards and reference values
        """
        return {
            'material_coefficients': {
                # NBR 12721 reference coefficients for different materials
                'Concrete/Masonry': {
                    'residential': {'min': 0.15, 'max': 0.25, 'typical': 0.20},
                    'commercial': {'min': 0.18, 'max': 0.28, 'typical': 0.23},
                    'industrial': {'min': 0.20, 'max': 0.30, 'typical': 0.25}
                },
                'Steel/Concrete': {
                    'residential': {'min': 0.10, 'max': 0.18, 'typical': 0.14},
                    'commercial': {'min': 0.12, 'max': 0.20, 'typical': 0.16},
                    'industrial': {'min': 0.15, 'max': 0.25, 'typical': 0.20}
                },
                'Roofing Materials': {
                    'residential': {'min': 0.03, 'max': 0.08, 'typical': 0.05},
                    'commercial': {'min': 0.04, 'max': 0.09, 'typical': 0.06},
                    'industrial': {'min': 0.05, 'max': 0.10, 'typical': 0.07}
                },
                'Doors/Windows': {
                    'residential': {'min': 0.05, 'max': 0.12, 'typical': 0.08},
                    'commercial': {'min': 0.08, 'max': 0.15, 'typical': 0.11},
                    'industrial': {'min': 0.04, 'max': 0.10, 'typical': 0.07}
                },
                'Finishes': {
                    'residential': {'min': 0.08, 'max': 0.18, 'typical': 0.13},
                    'commercial': {'min': 0.10, 'max': 0.20, 'typical': 0.15},
                    'industrial': {'min': 0.05, 'max': 0.12, 'typical': 0.08}
                },
                'MEP Systems': {
                    'residential': {'min': 0.12, 'max': 0.20, 'typical': 0.16},
                    'commercial': {'min': 0.15, 'max': 0.25, 'typical': 0.20},
                    'industrial': {'min': 0.18, 'max': 0.30, 'typical': 0.24}
                }
            },
            'unit_cost_references': {
                # Reference unit costs (R$/m², R$/unit) based on NBR guidelines
                'Concrete/Masonry': {
                    'unit': 'm³',
                    'min_cost': 120.0,
                    'max_cost': 200.0,
                    'typical_cost': 160.0
                },
                'Steel/Concrete': {
                    'unit': 'm³',
                    'min_cost': 600.0,
                    'max_cost': 1000.0,
                    'typical_cost': 800.0
                },
                'Roofing Materials': {
                    'unit': 'm²',
                    'min_cost': 30.0,
                    'max_cost': 60.0,
                    'typical_cost': 45.0
                },
                'Doors/Windows': {
                    'unit': 'unit',
                    'min_cost': 300.0,
                    'max_cost': 800.0,
                    'typical_cost': 550.0
                },
                'Finishes': {
                    'unit': 'm²',
                    'min_cost': 50.0,
                    'max_cost': 120.0,
                    'typical_cost': 85.0
                },
                'MEP Systems': {
                    'unit': 'm³',
                    'min_cost': 150.0,
                    'max_cost': 300.0,
                    'typical_cost': 225.0
                }
            },
            'mandatory_checks': [
                'material_ratio_compliance',
                'unit_cost_reasonableness',
                'total_cost_validation',
                'structural_adequacy',
                'safety_factors'
            ],
            'regional_adjustments': {
                'São Paulo': 1.10,
                'Rio de Janeiro': 1.05,
                'Belo Horizonte': 1.00,
                'Brasília': 1.02,
                'Salvador': 0.95,
                'Porto Alegre': 1.03,
                'Curitiba': 1.01,
                'Fortaleza': 0.90,
                'Recife': 0.92,
                'Manaus': 1.15
            }
        }
    
    def validate_project(self, project_id: int) -> List[str]:
        """
        Validate project against NBR 12721 standards
        """
        try:
            from database import DatabaseManager
            db_manager = DatabaseManager()
            
            # Get project data
            project_data = db_manager.get_project_details(project_id)
            materials = db_manager.get_project_materials(project_id)
            
            if not project_data or not materials:
                return ["Insufficient project data for validation"]
            
            issues = []
            
            # Validate material ratios
            ratio_issues = self._validate_material_ratios(project_data, materials)
            issues.extend(ratio_issues)
            
            # Validate unit costs
            cost_issues = self._validate_unit_costs(project_data, materials)
            issues.extend(cost_issues)
            
            # Validate total project cost
            total_issues = self._validate_total_cost(project_data, materials)
            issues.extend(total_issues)
            
            # Store compliance results
            self._store_compliance_results(db_manager, project_id, issues)
            
            logger.info(f"NBR validation completed for project {project_id}: {len(issues)} issues found")
            
            return issues
            
        except Exception as e:
            logger.error(f"Error validating project: {str(e)}")
            return [f"Validation error: {str(e)}"]
    
    def _validate_material_ratios(self, project_data: Dict[str, Any], 
                                materials: List[Dict[str, Any]]) -> List[str]:
        """
        Validate material cost ratios against NBR standards
        """
        issues = []
        
        try:
            project_type = project_data.get('type', 'residential').lower()
            total_cost = sum(material['cost'] for material in materials)
            
            if total_cost <= 0:
                return ["Total project cost is zero or negative"]
            
            # Check each material category
            for material in materials:
                category = material.get('category', material.get('name', ''))
                material_cost = material.get('cost', 0)
                material_ratio = material_cost / total_cost
                
                # Get NBR reference ratios
                if category in self.standards['material_coefficients']:
                    reference = self.standards['material_coefficients'][category].get(
                        project_type, 
                        self.standards['material_coefficients'][category]['residential']
                    )
                    
                    min_ratio = reference['min']
                    max_ratio = reference['max']
                    typical_ratio = reference['typical']
                    
                    # Check if ratio is within acceptable range
                    tolerance = self.tolerance_thresholds['material_ratio'] / 100
                    
                    if material_ratio < min_ratio * (1 - tolerance):
                        issues.append(
                            f"{category}: Cost ratio {material_ratio:.1%} is below NBR minimum "
                            f"of {min_ratio:.1%} for {project_type} projects"
                        )
                    elif material_ratio > max_ratio * (1 + tolerance):
                        issues.append(
                            f"{category}: Cost ratio {material_ratio:.1%} exceeds NBR maximum "
                            f"of {max_ratio:.1%} for {project_type} projects"
                        )
                    
                    # Warning for significant deviations from typical
                    deviation_from_typical = abs(material_ratio - typical_ratio) / typical_ratio
                    if deviation_from_typical > 0.30:  # 30% deviation threshold
                        issues.append(
                            f"{category}: Cost ratio {material_ratio:.1%} deviates significantly "
                            f"from typical {typical_ratio:.1%} (deviation: {deviation_from_typical:.1%})"
                        )
            
        except Exception as e:
            issues.append(f"Error validating material ratios: {str(e)}")
        
        return issues
    
    def _validate_unit_costs(self, project_data: Dict[str, Any], 
                           materials: List[Dict[str, Any]]) -> List[str]:
        """
        Validate unit costs against NBR reference values
        """
        issues = []
        
        try:
            location = project_data.get('location', 'São Paulo')
            regional_factor = self.standards['regional_adjustments'].get(location, 1.0)
            
            for material in materials:
                category = material.get('category', material.get('name', ''))
                unit_cost = material.get('unit_cost', 0)
                
                if unit_cost <= 0:
                    continue
                
                # Get NBR reference costs
                if category in self.standards['unit_cost_references']:
                    reference = self.standards['unit_cost_references'][category]
                    
                    # Apply regional adjustment
                    min_cost = reference['min_cost'] * regional_factor
                    max_cost = reference['max_cost'] * regional_factor
                    typical_cost = reference['typical_cost'] * regional_factor
                    
                    # Check reasonableness
                    tolerance = self.tolerance_thresholds['unit_cost_variance'] / 100
                    
                    if unit_cost < min_cost * (1 - tolerance):
                        issues.append(
                            f"{category}: Unit cost R${unit_cost:.2f}/{reference['unit']} "
                            f"is below reasonable minimum of R${min_cost:.2f} for {location}"
                        )
                    elif unit_cost > max_cost * (1 + tolerance):
                        issues.append(
                            f"{category}: Unit cost R${unit_cost:.2f}/{reference['unit']} "
                            f"exceeds reasonable maximum of R${max_cost:.2f} for {location}"
                        )
                    
                    # Check significant deviation from typical
                    deviation = abs(unit_cost - typical_cost) / typical_cost
                    if deviation > 0.40:  # 40% deviation threshold
                        issues.append(
                            f"{category}: Unit cost R${unit_cost:.2f} deviates significantly "
                            f"from typical R${typical_cost:.2f} (deviation: {deviation:.1%})"
                        )
            
        except Exception as e:
            issues.append(f"Error validating unit costs: {str(e)}")
        
        return issues
    
    def _validate_total_cost(self, project_data: Dict[str, Any], 
                           materials: List[Dict[str, Any]]) -> List[str]:
        """
        Validate total project cost reasonableness
        """
        issues = []
        
        try:
            total_cost = sum(material['cost'] for material in materials)
            project_type = project_data.get('type', 'residential').lower()
            
            # Estimate reasonable cost ranges based on project type
            # These are simplified estimates - in practice, you'd use more sophisticated methods
            cost_per_sqm_ranges = {
                'residential': {'min': 800, 'max': 2500, 'typical': 1500},
                'commercial': {'min': 1200, 'max': 3500, 'typical': 2200},
                'industrial': {'min': 1000, 'max': 3000, 'typical': 1800},
                'infrastructure': {'min': 1500, 'max': 4000, 'typical': 2500}
            }
            
            if project_type in cost_per_sqm_ranges:
                range_data = cost_per_sqm_ranges[project_type]
                
                # Assume 100m² as default area (this should come from BIM data)
                estimated_area = 100  # This should be extracted from BIM model
                cost_per_sqm = total_cost / estimated_area
                
                tolerance = self.tolerance_thresholds['total_cost_variance'] / 100
                
                if cost_per_sqm < range_data['min'] * (1 - tolerance):
                    issues.append(
                        f"Total cost R${cost_per_sqm:.2f}/m² appears low for {project_type} "
                        f"projects (expected minimum: R${range_data['min']:.2f}/m²)"
                    )
                elif cost_per_sqm > range_data['max'] * (1 + tolerance):
                    issues.append(
                        f"Total cost R${cost_per_sqm:.2f}/m² appears high for {project_type} "
                        f"projects (expected maximum: R${range_data['max']:.2f}/m²)"
                    )
            
            # Check for missing essential categories
            essential_categories = ['Concrete/Masonry', 'Steel/Concrete', 'MEP Systems']
            present_categories = {material.get('category', material.get('name', '')) 
                                for material in materials}
            
            missing_essential = set(essential_categories) - present_categories
            if missing_essential:
                issues.append(
                    f"Missing essential material categories: {', '.join(missing_essential)}"
                )
            
        except Exception as e:
            issues.append(f"Error validating total cost: {str(e)}")
        
        return issues
    
    def _store_compliance_results(self, db_manager, project_id: int, issues: List[str]):
        """
        Store compliance check results in database
        """
        try:
            import sqlite3
            
            with sqlite3.connect(db_manager.db_path) as conn:
                cursor = conn.cursor()
                
                # Clear previous compliance records
                cursor.execute('''
                    DELETE FROM nbr_compliance WHERE project_id = ?
                ''', (project_id,))
                
                # Store current compliance status
                if not issues:
                    cursor.execute('''
                        INSERT INTO nbr_compliance 
                        (project_id, standard_code, requirement, status, notes)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (project_id, 'NBR 12721', 'Overall Compliance', 'COMPLIANT', 
                          'All checks passed'))
                else:
                    for issue in issues:
                        cursor.execute('''
                            INSERT INTO nbr_compliance 
                            (project_id, standard_code, requirement, status, notes)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (project_id, 'NBR 12721', 'Compliance Check', 'NON_COMPLIANT', issue))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error storing compliance results: {str(e)}")
    
    def get_compliance_summary(self, project_id: int) -> Dict[str, Any]:
        """
        Get compliance summary for a project
        """
        try:
            from database import DatabaseManager
            db_manager = DatabaseManager()
            
            with sqlite3.connect(db_manager.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT status, COUNT(*) as count
                    FROM nbr_compliance 
                    WHERE project_id = ?
                    GROUP BY status
                ''', (project_id,))
                
                status_counts = dict(cursor.fetchall())
                
                cursor.execute('''
                    SELECT requirement, status, notes
                    FROM nbr_compliance 
                    WHERE project_id = ?
                    ORDER BY checked_date DESC
                ''', (project_id,))
                
                details = []
                for row in cursor.fetchall():
                    details.append({
                        'requirement': row[0],
                        'status': row[1],
                        'notes': row[2]
                    })
                
                total_checks = sum(status_counts.values())
                compliant_checks = status_counts.get('COMPLIANT', 0)
                compliance_percentage = (compliant_checks / total_checks * 100) if total_checks > 0 else 0
                
                return {
                    'compliance_percentage': compliance_percentage,
                    'total_checks': total_checks,
                    'compliant_checks': compliant_checks,
                    'non_compliant_checks': status_counts.get('NON_COMPLIANT', 0),
                    'details': details
                }
                
        except Exception as e:
            logger.error(f"Error getting compliance summary: {str(e)}")
            return {'compliance_percentage': 0, 'total_checks': 0, 'details': []}
    
    def get_nbr_recommendations(self, project_type: str, material_category: str) -> Dict[str, Any]:
        """
        Get NBR recommendations for specific material category and project type
        """
        try:
            project_type = project_type.lower()
            
            recommendations = {}
            
            # Material ratio recommendations
            if material_category in self.standards['material_coefficients']:
                ratio_data = self.standards['material_coefficients'][material_category].get(
                    project_type,
                    self.standards['material_coefficients'][material_category]['residential']
                )
                recommendations['ratio'] = ratio_data
            
            # Unit cost recommendations
            if material_category in self.standards['unit_cost_references']:
                cost_data = self.standards['unit_cost_references'][material_category]
                recommendations['unit_cost'] = cost_data
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting NBR recommendations: {str(e)}")
            return {}
    
    def export_compliance_report(self, project_id: int) -> str:
        """
        Export compliance report (simplified version)
        """
        try:
            summary = self.get_compliance_summary(project_id)
            
            report = f"""
NBR 12721 COMPLIANCE REPORT
Project ID: {project_id}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY:
- Compliance Percentage: {summary['compliance_percentage']:.1f}%
- Total Checks: {summary['total_checks']}
- Compliant: {summary['compliant_checks']}
- Non-Compliant: {summary['non_compliant_checks']}

DETAILED RESULTS:
"""
            
            for detail in summary['details']:
                status_symbol = "✓" if detail['status'] == 'COMPLIANT' else "✗"
                report += f"{status_symbol} {detail['requirement']}: {detail['notes']}\n"
            
            return report
            
        except Exception as e:
            logger.error(f"Error exporting compliance report: {str(e)}")
            return f"Error generating report: {str(e)}"
