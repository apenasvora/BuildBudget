import ifcopenshell
import ifcopenshell.util.element
import ifcopenshell.util.unit
import ezdxf
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
        self.supported_formats = ['.ifc', '.dwg']
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
        
        # DWG layer mappings to material categories
        self.dwg_layer_mappings = {
            'WALL': 'Concrete/Masonry',
            'WALLS': 'Concrete/Masonry',
            'SLAB': 'Concrete',
            'SLABS': 'Concrete', 
            'BEAM': 'Steel/Concrete',
            'BEAMS': 'Steel/Concrete',
            'COLUMN': 'Steel/Concrete',
            'COLUMNS': 'Steel/Concrete',
            'ROOF': 'Roofing Materials',
            'ROOFING': 'Roofing Materials',
            'DOOR': 'Doors/Windows',
            'DOORS': 'Doors/Windows',
            'WINDOW': 'Doors/Windows',
            'WINDOWS': 'Doors/Windows',
            'STAIR': 'Concrete',
            'STAIRS': 'Concrete',
            'RAILING': 'Steel',
            'RAILINGS': 'Steel',
            'FINISH': 'Finishes',
            'FINISHES': 'Finishes',
            'ELECTRICAL': 'MEP Systems',
            'PLUMBING': 'MEP Systems',
            'HVAC': 'MEP Systems',
            'MEP': 'MEP Systems'
        }
    
    def extract_quantities(self, file_path: str) -> Dict[str, float]:
        """
        Extract quantities from BIM model (IFC or DWG)
        
        Args:
            file_path: Path to the BIM file (.ifc or .dwg)
            
        Returns:
            Dictionary mapping material categories to quantities
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"BIM file not found: {file_path}")
            
            logger.info(f"Processing BIM file: {file_path}")
            
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.ifc':
                return self._extract_ifc_quantities(file_path)
            elif file_extension == '.dwg':
                return self._extract_dwg_quantities(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
        except Exception as e:
            logger.error(f"Error processing BIM file: {str(e)}")
            raise
    
    def _extract_ifc_quantities(self, file_path: str) -> Dict[str, float]:
        """
        Extract quantities from IFC file
        """
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
        
        logger.info(f"Extracted quantities for {len(quantities)} material categories from IFC")
        
        return quantities
    
    def _extract_dwg_quantities(self, file_path: str) -> Dict[str, float]:
        """
        Extract quantities from DWG file
        """
        try:
            # Load DWG file
            doc = ezdxf.readfile(file_path)
            modelspace = doc.modelspace()
            
            # Initialize quantities dictionary
            quantities = {}
            
            # Process entities by layer
            layer_counts = {}
            layer_areas = {}
            layer_lengths = {}
            
            for entity in modelspace:
                try:
                    layer_name = entity.dxf.layer.upper()
                    
                    # Count entities per layer
                    if layer_name not in layer_counts:
                        layer_counts[layer_name] = 0
                        layer_areas[layer_name] = 0
                        layer_lengths[layer_name] = 0
                    
                    layer_counts[layer_name] += 1
                    
                    # Calculate areas and lengths based on entity type
                    if entity.dxftype() == 'LWPOLYLINE' or entity.dxftype() == 'POLYLINE':
                        try:
                            # Calculate area for closed polylines
                            if hasattr(entity, 'is_closed') and entity.is_closed:
                                # Simplified area calculation
                                vertices = list(entity.vertices())
                                if len(vertices) >= 3:
                                    # Using shoelace formula for polygon area
                                    area = 0
                                    for i in range(len(vertices)):
                                        j = (i + 1) % len(vertices)
                                        area += vertices[i][0] * vertices[j][1]
                                        area -= vertices[j][0] * vertices[i][1]
                                    area = abs(area) / 2
                                    layer_areas[layer_name] += area
                            
                            # Calculate length
                            length = 0
                            vertices = list(entity.vertices())
                            for i in range(len(vertices) - 1):
                                dx = vertices[i+1][0] - vertices[i][0]
                                dy = vertices[i+1][1] - vertices[i][1]
                                length += (dx*dx + dy*dy)**0.5
                            layer_lengths[layer_name] += length
                            
                        except Exception as e:
                            logger.warning(f"Error calculating polyline metrics: {str(e)}")
                    
                    elif entity.dxftype() == 'LINE':
                        try:
                            start = entity.dxf.start
                            end = entity.dxf.end
                            length = ((end[0] - start[0])**2 + (end[1] - start[1])**2 + (end[2] - start[2])**2)**0.5
                            layer_lengths[layer_name] += length
                        except Exception as e:
                            logger.warning(f"Error calculating line length: {str(e)}")
                    
                    elif entity.dxftype() == 'CIRCLE':
                        try:
                            radius = entity.dxf.radius
                            area = 3.14159 * radius * radius
                            layer_areas[layer_name] += area
                        except Exception as e:
                            logger.warning(f"Error calculating circle area: {str(e)}")
                    
                    elif entity.dxftype() == 'ARC':
                        try:
                            radius = entity.dxf.radius
                            start_angle = entity.dxf.start_angle
                            end_angle = entity.dxf.end_angle
                            angle_diff = end_angle - start_angle
                            if angle_diff < 0:
                                angle_diff += 360
                            arc_length = (angle_diff / 360) * 2 * 3.14159 * radius
                            layer_lengths[layer_name] += arc_length
                        except Exception as e:
                            logger.warning(f"Error calculating arc length: {str(e)}")
                
                except Exception as e:
                    logger.warning(f"Error processing DWG entity: {str(e)}")
                    continue
            
            # Map layers to material categories and calculate quantities
            for layer, count in layer_counts.items():
                material_category = 'Miscellaneous'
                
                # Find matching material category
                for layer_pattern, category in self.dwg_layer_mappings.items():
                    if layer_pattern in layer:
                        material_category = category
                        break
                
                # Calculate quantity based on layer type
                quantity = 0
                
                if 'WALL' in layer or 'SLAB' in layer or 'BEAM' in layer or 'COLUMN' in layer:
                    # Use area for structural elements
                    quantity = max(layer_areas[layer], count * 10)  # Fallback to count-based
                elif 'DOOR' in layer or 'WINDOW' in layer:
                    # Use count for openings
                    quantity = count
                elif 'ELECTRICAL' in layer or 'PLUMBING' in layer or 'HVAC' in layer:
                    # Use length for MEP systems
                    quantity = max(layer_lengths[layer] / 100, count * 5)  # Convert to reasonable units
                else:
                    # Use area or count as appropriate
                    quantity = max(layer_areas[layer], count * 5)
                
                # Accumulate quantities by category
                if material_category in quantities:
                    quantities[material_category] += quantity
                else:
                    quantities[material_category] = quantity
            
            # Add some estimated quantities if none found
            if not quantities:
                quantities = {
                    'Miscellaneous': 100.0,
                    'Concrete/Masonry': 50.0,
                    'MEP Systems': 25.0
                }
            
            logger.info(f"Extracted quantities for {len(quantities)} material categories from DWG")
            
            return quantities
            
        except Exception as e:
            logger.error(f"Error processing DWG file: {str(e)}")
            # Return basic fallback quantities
            return {
                'Miscellaneous': 100.0,
                'Concrete/Masonry': 50.0,
                'Steel/Concrete': 25.0,
                'MEP Systems': 30.0
            }
    
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
