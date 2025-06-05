import ifcopenshell
import ifcopenshell.util.element
import ifcopenshell.util.unit
from typing import Dict, List, Any
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BIMProcessor:
    """
    BIM processor for extracting quantities and material information from IFC files
    """
    
    def __init__(self):
        self.supported_formats = ['.ifc']
        self.material_mappings = {
            # Standard IFC element types to material categories
            'IfcWall': 'Concrete/Masonry',
            'IfcSlab': 'Concrete',
            'IfcBeam': 'Steel/Concrete',
            'IfcColumn': 'Steel/Concrete',
            'IfcRoof': 'Roofing Materials',
            'IfcDoor': 'Doors/Windows',
            'IfcWindow': 'Doors/Windows',
            'IfcStair': 'Concrete',
            'IfcRailing': 'Steel',
            'IfcCovering': 'Finishes',
            'IfcFurnishingElement': 'Furniture/Fixtures',
            'IfcBuildingElementProxy': 'Miscellaneous'
        }
    
    def extract_quantities(self, file_path: str) -> Dict[str, float]:
        """
        Extract quantities from IFC BIM model
        
        Args:
            file_path: Path to the IFC file
            
        Returns:
            Dictionary mapping material categories to quantities
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"BIM file not found: {file_path}")
            
            logger.info(f"Processing BIM file: {file_path}")
            
            # Load IFC file
            model = ifcopenshell.open(file_path)
            
            # Initialize quantities dictionary
            quantities = {}
            
            # Get all building elements
            elements = model.by_type("IfcBuildingElement")
            
            for element in elements:
                try:
                    # Get element type
                    element_type = element.is_a()
                    
                    # Map to material category
                    material_category = self.material_mappings.get(element_type, 'Miscellaneous')
                    
                    # Extract quantities
                    quantity = self._extract_element_quantity(element, model)
                    
                    # Accumulate quantities by category
                    if material_category in quantities:
                        quantities[material_category] += quantity
                    else:
                        quantities[material_category] = quantity
                        
                except Exception as e:
                    logger.warning(f"Error processing element {element.GlobalId}: {str(e)}")
                    continue
            
            # Process additional elements like spaces, zones
            self._process_spaces(model, quantities)
            
            logger.info(f"Extracted quantities for {len(quantities)} material categories")
            
            return quantities
            
        except Exception as e:
            logger.error(f"Error processing BIM file: {str(e)}")
            raise
    
    def _extract_element_quantity(self, element, model) -> float:
        """
        Extract quantity from individual element
        """
        try:
            # Try to get quantity from property sets
            psets = ifcopenshell.util.element.get_psets(element)
            
            # Look for common quantity properties
            quantity_keys = [
                'NetVolume', 'GrossVolume', 'Volume',
                'NetArea', 'GrossArea', 'Area',
                'Length', 'Width', 'Height',
                'Count', 'Quantity'
            ]
            
            for pset_name, pset_data in psets.items():
                for key in quantity_keys:
                    if key in pset_data and pset_data[key] is not None:
                        try:
                            return float(pset_data[key])
                        except (ValueError, TypeError):
                            continue
            
            # Fallback: calculate from geometry if available
            return self._calculate_geometric_quantity(element)
            
        except Exception as e:
            logger.warning(f"Could not extract quantity for element: {str(e)}")
            return 1.0  # Default quantity
    
    def _calculate_geometric_quantity(self, element) -> float:
        """
        Calculate quantity from element geometry
        """
        try:
            # Get element geometry
            if hasattr(element, 'Representation') and element.Representation:
                # This is a simplified approach - in practice, you'd need more
                # sophisticated geometry processing
                
                # For now, return a default value based on element type
                element_type = element.is_a()
                
                if 'Wall' in element_type or 'Slab' in element_type:
                    return 10.0  # m² or m³
                elif 'Beam' in element_type or 'Column' in element_type:
                    return 5.0   # m³
                elif 'Door' in element_type or 'Window' in element_type:
                    return 1.0   # units
                else:
                    return 1.0   # default
            
            return 1.0
            
        except Exception as e:
            logger.warning(f"Could not calculate geometric quantity: {str(e)}")
            return 1.0
    
    def _process_spaces(self, model, quantities: Dict[str, float]):
        """
        Process spaces and zones for additional material calculations
        """
        try:
            spaces = model.by_type("IfcSpace")
            
            total_area = 0
            total_volume = 0
            
            for space in spaces:
                try:
                    psets = ifcopenshell.util.element.get_psets(space)
                    
                    for pset_name, pset_data in psets.items():
                        if 'NetFloorArea' in pset_data:
                            total_area += float(pset_data['NetFloorArea'])
                        if 'NetVolume' in pset_data:
                            total_volume += float(pset_data['NetVolume'])
                            
                except Exception as e:
                    continue
            
            # Add derived material quantities based on spaces
            if total_area > 0:
                # Estimate flooring materials
                if 'Finishes' in quantities:
                    quantities['Finishes'] += total_area * 0.1  # 10cm thickness estimate
                else:
                    quantities['Finishes'] = total_area * 0.1
            
            if total_volume > 0:
                # Estimate MEP (Mechanical, Electrical, Plumbing) based on volume
                mep_estimate = total_volume * 0.05  # 5% of volume for MEP
                if 'MEP Systems' in quantities:
                    quantities['MEP Systems'] += mep_estimate
                else:
                    quantities['MEP Systems'] = mep_estimate
                    
        except Exception as e:
            logger.warning(f"Error processing spaces: {str(e)}")
    
    def get_element_details(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Get detailed information about all elements in the BIM model
        """
        try:
            model = ifcopenshell.open(file_path)
            elements = []
            
            building_elements = model.by_type("IfcBuildingElement")
            
            for element in building_elements:
                try:
                    element_info = {
                        'global_id': element.GlobalId,
                        'type': element.is_a(),
                        'name': getattr(element, 'Name', ''),
                        'description': getattr(element, 'Description', ''),
                        'material_category': self.material_mappings.get(element.is_a(), 'Miscellaneous'),
                        'properties': ifcopenshell.util.element.get_psets(element)
                    }
                    elements.append(element_info)
                    
                except Exception as e:
                    logger.warning(f"Error getting element details: {str(e)}")
                    continue
            
            return elements
            
        except Exception as e:
            logger.error(f"Error getting element details: {str(e)}")
            return []
    
    def validate_ifc_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate IFC file and return basic information
        """
        try:
            model = ifcopenshell.open(file_path)
            
            # Get basic model information
            project = model.by_type("IfcProject")[0] if model.by_type("IfcProject") else None
            
            validation_info = {
                'is_valid': True,
                'ifc_version': model.wrapped_data.header.file_description.description[0] if model.wrapped_data.header.file_description.description else 'Unknown',
                'project_name': project.Name if project and hasattr(project, 'Name') else 'Unknown',
                'element_count': len(model.by_type("IfcBuildingElement")),
                'space_count': len(model.by_type("IfcSpace")),
                'units': self._get_model_units(model),
                'errors': []
            }
            
            # Basic validation checks
            if validation_info['element_count'] == 0:
                validation_info['errors'].append("No building elements found in model")
            
            return validation_info
            
        except Exception as e:
            return {
                'is_valid': False,
                'error': str(e),
                'errors': [str(e)]
            }
    
    def _get_model_units(self, model) -> Dict[str, str]:
        """
        Extract units from IFC model
        """
        try:
            units = {}
            
            # Get unit assignments
            unit_assignments = model.by_type("IfcUnitAssignment")
            
            if unit_assignments:
                for unit_assignment in unit_assignments:
                    for unit in unit_assignment.Units:
                        if hasattr(unit, 'UnitType'):
                            unit_type = unit.UnitType
                            if hasattr(unit, 'Name'):
                                units[unit_type] = unit.Name
                            elif hasattr(unit, 'Prefix') and hasattr(unit, 'Name'):
                                units[unit_type] = f"{unit.Prefix}{unit.Name}"
            
            # Default units if not found
            if not units:
                units = {
                    'LENGTHUNIT': 'METRE',
                    'AREAUNIT': 'SQUARE_METRE', 
                    'VOLUMEUNIT': 'CUBIC_METRE'
                }
            
            return units
            
        except Exception as e:
            logger.warning(f"Could not extract units: {str(e)}")
            return {'LENGTHUNIT': 'METRE', 'AREAUNIT': 'SQUARE_METRE', 'VOLUMEUNIT': 'CUBIC_METRE'}
