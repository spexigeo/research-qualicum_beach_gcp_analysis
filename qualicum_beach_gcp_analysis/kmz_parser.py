"""
Parser for Qualicum Beach Ground Control Point KMZ files.

KMZ files are ZIP archives containing KML (Keyhole Markup Language) XML files.
This parser extracts GCP coordinates and metadata from the KMZ/KML format.
"""

import zipfile
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional
import os
import re
from pathlib import Path


def parse_kmz_file(kmz_path: str) -> List[Dict]:
    """
    Parse a KMZ file and extract Ground Control Points.
    
    Args:
        kmz_path: Path to the KMZ file
        
    Returns:
        List of GCP dictionaries with keys: lat, lon, id, description, etc.
    """
    gcps = []
    
    if not os.path.exists(kmz_path):
        print(f"Warning: KMZ file not found: {kmz_path}")
        return gcps
    
    try:
        # KMZ files are ZIP archives
        with zipfile.ZipFile(kmz_path, 'r') as kmz:
            # Find the KML file inside (usually doc.kml or similar)
            kml_files = [f for f in kmz.namelist() if f.endswith('.kml')]
            
            if not kml_files:
                print(f"Warning: No KML file found in KMZ: {kmz_path}")
                print(f"Files in KMZ: {kmz.namelist()}")
                return gcps
            
            print(f"Found {len(kml_files)} KML file(s) in KMZ")
            
            # Parse the first KML file (usually there's only one)
            kml_content = kmz.read(kml_files[0])
            
            # Preprocess KML content to handle namespace issues
            # Some KML files have unbound prefixes - try to fix them
            kml_text = kml_content.decode('utf-8', errors='ignore')
            
            # Try to register common namespaces if they're missing
            # Check if the root element has namespace declarations
            if 'xmlns' not in kml_text[:500] or 'xmlns:kml' not in kml_text[:500]:
                # Try to add namespace declarations if missing
                # This is a workaround for KML files with unbound prefixes
                pass  # We'll handle this in the parsing step
            
            # Parse XML - handle namespace issues
            root = None
            try:
                root = ET.fromstring(kml_content)
            except ET.ParseError as parse_err:
                # If there's an unbound prefix error, try to fix it
                if 'unbound prefix' in str(parse_err).lower():
                    print("Attempting to fix namespace issues...")
                    kml_text_fixed = kml_text
                    
                    # Remove unbound namespace prefixes from attributes (e.g., xsi:schemaLocation)
                    # This is often the issue - attributes with undeclared prefixes
                    kml_text_fixed = re.sub(r'\s+[a-zA-Z0-9_]+:([a-zA-Z0-9_]+)="[^"]*"', r'', kml_text_fixed)
                    
                    # Also remove namespace prefixes from tags
                    kml_text_fixed = re.sub(r'<([a-zA-Z0-9_]+):([a-zA-Z0-9_]+)', r'<\2', kml_text_fixed)
                    kml_text_fixed = re.sub(r'</([a-zA-Z0-9_]+):([a-zA-Z0-9_]+)', r'</\2', kml_text_fixed)
                    
                    # Remove problematic xmlns declarations that might reference undeclared namespaces
                    kml_text_fixed = re.sub(r'xmlns:[a-zA-Z0-9_]+="[^"]*"', '', kml_text_fixed)
                    
                    # Ensure default namespace exists
                    if 'xmlns="' not in kml_text_fixed[:500]:
                        root_tag_match = re.search(r'<([a-zA-Z0-9_]+)', kml_text_fixed[:200])
                        if root_tag_match:
                            root_tag = root_tag_match.group(1)
                            kml_text_fixed = kml_text_fixed.replace(
                                f'<{root_tag}',
                                f'<{root_tag} xmlns="http://www.opengis.net/kml/2.2"',
                                1
                            )
                    
                    try:
                        kml_content_fixed = kml_text_fixed.encode('utf-8')
                        root = ET.fromstring(kml_content_fixed)
                        print("✓ Fixed namespace issues in KML file")
                    except Exception as e2:
                        print(f"First fix attempt failed: {e2}")
                        # More aggressive: remove ALL attributes with colons (namespace prefixes)
                        kml_text_aggressive = re.sub(r'\s+[^=]*:[^=]*="[^"]*"', '', kml_text)
                        kml_text_aggressive = re.sub(r'<([a-zA-Z0-9_]+):([a-zA-Z0-9_]+)', r'<\2', kml_text_aggressive)
                        kml_text_aggressive = re.sub(r'</([a-zA-Z0-9_]+):([a-zA-Z0-9_]+)', r'</\2', kml_text_aggressive)
                        
                        # Ensure default namespace
                        root_tag_match = re.search(r'<([a-zA-Z0-9_]+)', kml_text_aggressive[:200])
                        if root_tag_match:
                            root_tag = root_tag_match.group(1)
                            if 'xmlns="' not in kml_text_aggressive[:500]:
                                kml_text_aggressive = kml_text_aggressive.replace(
                                    f'<{root_tag}',
                                    f'<{root_tag} xmlns="http://www.opengis.net/kml/2.2"',
                                    1
                                )
                        
                        try:
                            kml_content_aggressive = kml_text_aggressive.encode('utf-8')
                            root = ET.fromstring(kml_content_aggressive)
                            print("✓ Fixed namespace issues using aggressive approach")
                        except Exception as e3:
                            print(f"All namespace fix attempts failed. Last error: {e3}")
                            raise parse_err
                else:
                    raise parse_err
            
            if root is None:
                raise ValueError("Failed to parse KML file")
            
            # Try to detect namespaces from the root element
            # Some KML files use different namespace URIs
            detected_namespaces = {}
            if root.tag.startswith('{'):
                # Extract namespace from root tag
                ns_uri = root.tag[1:root.tag.index('}')]
                detected_namespaces['kml'] = ns_uri
            
            # Define common namespaces (KML uses namespaces)
            namespaces = {
                'kml': 'http://www.opengis.net/kml/2.2',
                'gx': 'http://www.google.com/kml/ext/2.2'
            }
            
            # Update with detected namespace if found
            if detected_namespaces:
                namespaces.update(detected_namespaces)
            
            # Try multiple namespace variations
            placemarks = []
            namespace_used = None
            
            # Try with detected/default namespace
            if namespaces.get('kml'):
                placemarks = root.findall('.//{http://www.opengis.net/kml/2.2}Placemark')
                if placemarks:
                    namespace_used = 'http://www.opengis.net/kml/2.2'
                    namespaces = {'kml': namespace_used}
            
            # Try with detected namespace if different
            if not placemarks and detected_namespaces.get('kml'):
                ns_uri = detected_namespaces['kml']
                placemarks = root.findall(f'.//{{{ns_uri}}}Placemark')
                if placemarks:
                    namespace_used = ns_uri
                    namespaces = {'kml': ns_uri}
            
            # Try with namespace prefix
            if not placemarks:
                placemarks = root.findall('.//kml:Placemark', namespaces)
                if placemarks:
                    namespace_used = 'with_prefix'
            
            # Try without namespace (some KML files don't use namespaces)
            if not placemarks:
                placemarks = root.findall('.//Placemark')
                if placemarks:
                    namespace_used = 'no_namespace'
                    namespaces = {}
            
            # Try finding any element with "Placemark" in the tag name
            if not placemarks:
                placemarks = [elem for elem in root.iter() if 'Placemark' in elem.tag]
                if placemarks:
                    namespace_used = 'iter_search'
                    # Try to extract namespace
                    if placemarks[0].tag.startswith('{'):
                        ns_uri = placemarks[0].tag[1:placemarks[0].tag.index('}')]
                        namespaces = {'kml': ns_uri}
                    else:
                        namespaces = {}
            
            print(f"Found {len(placemarks)} placemarks in KMZ file (namespace: {namespace_used})")
            
            # Debug: print root element info if no placemarks found
            if not placemarks:
                print(f"Debug: Root element tag: {root.tag}")
                print(f"Debug: Root element children: {[child.tag for child in root]}")
                # Try to find any elements that might contain coordinates
                all_elements = [elem.tag for elem in root.iter()]
                print(f"Debug: All element tags (first 20): {all_elements[:20]}")
            
            for idx, placemark in enumerate(placemarks):
                gcp = _parse_placemark(placemark, namespaces, idx)
                if gcp:
                    gcps.append(gcp)
            
            print(f"Successfully parsed {len(gcps)} GCPs from KMZ file")
            
    except zipfile.BadZipFile:
        print(f"Error: {kmz_path} is not a valid ZIP/KMZ file")
        return gcps
    except ET.ParseError as e:
        print(f"Error parsing KML XML: {e}")
        return gcps
    except Exception as e:
        print(f"Error reading KMZ file {kmz_path}: {e}")
        return gcps
    
    return gcps


def _parse_placemark(placemark: ET.Element, namespaces: Dict[str, str], default_id: int) -> Optional[Dict]:
    """
    Parse a single Placemark element to extract GCP information.
    
    Args:
        placemark: XML Element representing a Placemark
        namespaces: XML namespaces dictionary
        default_id: Default ID if name is not found
        
    Returns:
        GCP dictionary or None if invalid
    """
    try:
        # Extract name/ID
        name_elem = placemark.find('kml:name', namespaces) if namespaces else placemark.find('name')
        gcp_id = name_elem.text.strip() if name_elem is not None and name_elem.text else f"GCP_{default_id:04d}"
        
        # Extract description
        desc_elem = placemark.find('kml:description', namespaces) if namespaces else placemark.find('description')
        description = desc_elem.text.strip() if desc_elem is not None and desc_elem.text else ""
        
        # Extract coordinates (can be in Point, LineString, or Polygon)
        coords = None
        
        # Try Point element first (most common for GCPs)
        # Try with namespace URI directly
        point = None
        if namespaces.get('kml'):
            ns_uri = namespaces['kml']
            point = placemark.find(f'{{{ns_uri}}}Point')
            if point is None:
                point = placemark.find('kml:Point', namespaces)
        else:
            point = placemark.find('Point')
        
        if point is not None:
            # Try to find coordinates element
            coord_elem = None
            if namespaces.get('kml'):
                ns_uri = namespaces['kml']
                coord_elem = point.find(f'{{{ns_uri}}}coordinates')
                if coord_elem is None:
                    coord_elem = point.find('kml:coordinates', namespaces)
            else:
                coord_elem = point.find('coordinates')
            
            if coord_elem is not None and coord_elem.text:
                coords = _parse_coordinates(coord_elem.text)
        
        # If no Point, try Polygon (common for area definitions - extract centroid)
        if coords is None:
            polygon = None
            if namespaces.get('kml'):
                ns_uri = namespaces['kml']
                polygon = placemark.find(f'{{{ns_uri}}}Polygon')
                if polygon is None:
                    # Check inside MultiGeometry
                    multigeom = placemark.find(f'{{{ns_uri}}}MultiGeometry')
                    if multigeom is not None:
                        polygon = multigeom.find(f'{{{ns_uri}}}Polygon')
                if polygon is None:
                    polygon = placemark.find('kml:Polygon', namespaces)
            else:
                polygon = placemark.find('Polygon')
                if polygon is None:
                    multigeom = placemark.find('MultiGeometry')
                    if multigeom is not None:
                        polygon = multigeom.find('Polygon')
            
            if polygon is not None:
                # Find coordinates in Polygon > outerBoundaryIs > LinearRing > coordinates
                outer = None
                if namespaces.get('kml'):
                    ns_uri = namespaces['kml']
                    outer = polygon.find(f'{{{ns_uri}}}outerBoundaryIs')
                    if outer is None:
                        outer = polygon.find('kml:outerBoundaryIs', namespaces)
                else:
                    outer = polygon.find('outerBoundaryIs')
                
                if outer is not None:
                    linear = None
                    if namespaces.get('kml'):
                        ns_uri = namespaces['kml']
                        linear = outer.find(f'{{{ns_uri}}}LinearRing')
                        if linear is None:
                            linear = outer.find('kml:LinearRing', namespaces)
                    else:
                        linear = outer.find('LinearRing')
                    
                    if linear is not None:
                        coord_elem = None
                        if namespaces.get('kml'):
                            ns_uri = namespaces['kml']
                            coord_elem = linear.find(f'{{{ns_uri}}}coordinates')
                            if coord_elem is None:
                                coord_elem = linear.find('kml:coordinates', namespaces)
                        else:
                            coord_elem = linear.find('coordinates')
                        
                        if coord_elem is not None and coord_elem.text:
                            # Calculate centroid from polygon coordinates
                            coords_list = _parse_coordinate_list(coord_elem.text)
                            if coords_list:
                                # Use first coordinate as representative point
                                # (or calculate actual centroid if needed)
                                coords = coords_list[0]
                                # Optionally calculate centroid:
                                if len(coords_list) > 1:
                                    avg_lon = sum(c[0] for c in coords_list) / len(coords_list)
                                    avg_lat = sum(c[1] for c in coords_list) / len(coords_list)
                                    avg_elev = sum(c[2] for c in coords_list if c[2] is not None) / len([c for c in coords_list if c[2] is not None]) if any(c[2] is not None for c in coords_list) else None
                                    coords = (avg_lon, avg_lat, avg_elev)
        
        # If still no coords, try LineString
        if coords is None:
            linestring = None
            if namespaces.get('kml'):
                ns_uri = namespaces['kml']
                linestring = placemark.find(f'{{{ns_uri}}}LineString')
                if linestring is None:
                    linestring = placemark.find('kml:LineString', namespaces)
            else:
                linestring = placemark.find('LineString')
            
            if linestring is not None:
                coord_elem = None
                if namespaces.get('kml'):
                    ns_uri = namespaces['kml']
                    coord_elem = linestring.find(f'{{{ns_uri}}}coordinates')
                    if coord_elem is None:
                        coord_elem = linestring.find('kml:coordinates', namespaces)
                else:
                    coord_elem = linestring.find('coordinates')
                
                if coord_elem is not None and coord_elem.text:
                    # Take first coordinate from LineString
                    coords_list = _parse_coordinate_list(coord_elem.text)
                    if coords_list:
                        coords = coords_list[0]
        
        if coords is None:
            # Try to find coordinates in ExtendedData or other locations
            extended_data = placemark.find('kml:ExtendedData', namespaces) if namespaces else placemark.find('ExtendedData')
            if extended_data is not None:
                # Look for coordinate data in Data elements
                for data in extended_data.findall('kml:Data', namespaces) if namespaces else extended_data.findall('Data'):
                    value_elem = data.find('kml:value', namespaces) if namespaces else data.find('value')
                    if value_elem is not None and value_elem.text:
                        # Try to parse as coordinates
                        try:
                            coords = _parse_coordinates(value_elem.text)
                            if coords:
                                break
                        except:
                            pass
        
        if coords is None:
            # Skip if no coordinates found
            return None
        
        lon, lat, elevation = coords
        
        # Extract additional metadata from ExtendedData
        metadata = {}
        extended_data = placemark.find('kml:ExtendedData', namespaces) if namespaces else placemark.find('ExtendedData')
        if extended_data is not None:
            for data in extended_data.findall('kml:Data', namespaces) if namespaces else extended_data.findall('Data'):
                name_attr = data.get('name') if 'name' in data.attrib else None
                value_elem = data.find('kml:value', namespaces) if namespaces else data.find('value')
                if name_attr and value_elem is not None and value_elem.text:
                    metadata[name_attr.lower()] = value_elem.text.strip()
        
        # Build GCP dictionary
        gcp = {
            'id': gcp_id,
            'lat': lat,
            'lon': lon,
            'z': elevation if elevation is not None else 0.0,
            'description': description,
            'source': 'qualicum_beach',
        }
        
        # Add any additional metadata
        gcp.update({k: v for k, v in metadata.items()})
        
        return gcp
        
    except Exception as e:
        print(f"Warning: Error parsing placemark: {e}")
        return None


def _parse_coordinates(coord_string: str) -> Optional[Tuple[float, float, Optional[float]]]:
    """
    Parse coordinate string in format "lon,lat,elevation" or "lon,lat".
    
    Args:
        coord_string: Coordinate string from KML
        
    Returns:
        Tuple of (lon, lat, elevation) or None if invalid
    """
    try:
        parts = coord_string.strip().split(',')
        if len(parts) >= 2:
            lon = float(parts[0].strip())
            lat = float(parts[1].strip())
            elevation = float(parts[2].strip()) if len(parts) > 2 and parts[2].strip() else None
            return (lon, lat, elevation)
    except (ValueError, IndexError):
        pass
    return None


def _parse_coordinate_list(coord_string: str) -> List[Tuple[float, float, Optional[float]]]:
    """
    Parse a list of coordinates separated by spaces or newlines.
    
    Args:
        coord_string: String containing multiple coordinates
        
    Returns:
        List of (lon, lat, elevation) tuples
    """
    coords = []
    # Split by whitespace (spaces, newlines, tabs)
    for coord in coord_string.strip().split():
        parsed = _parse_coordinates(coord)
        if parsed:
            coords.append(parsed)
    return coords


def inspect_kmz_structure(kmz_path: str) -> None:
    """
    Inspect the structure of a KMZ file for debugging.
    
    Args:
        kmz_path: Path to KMZ file
    """
    if not os.path.exists(kmz_path):
        print(f"KMZ file not found: {kmz_path}")
        return
    
    try:
        with zipfile.ZipFile(kmz_path, 'r') as kmz:
            print(f"Files in KMZ archive:")
            for f in kmz.namelist():
                print(f"  - {f}")
            
            kml_files = [f for f in kmz.namelist() if f.endswith('.kml')]
            if kml_files:
                print(f"\nReading KML file: {kml_files[0]}")
                kml_content = kmz.read(kml_files[0])
                root = ET.fromstring(kml_content)
                
                print(f"Root element: {root.tag}")
                print(f"Root attributes: {root.attrib}")
                print(f"\nDirect children:")
                for child in root:
                    print(f"  - {child.tag}: {child.attrib}")
                    # Show first few grandchildren
                    for grandchild in list(child)[:3]:
                        print(f"    - {grandchild.tag}")
                
                # Try to find any coordinate-like elements
                print(f"\nSearching for coordinate-like elements...")
                coord_elements = []
                for elem in root.iter():
                    if 'coord' in elem.tag.lower() or 'point' in elem.tag.lower():
                        coord_elements.append(elem.tag)
                
                if coord_elements:
                    print(f"Found {len(coord_elements)} coordinate-like elements:")
                    for tag in set(coord_elements[:10]):
                        print(f"  - {tag}")
                else:
                    print("No coordinate-like elements found")
                
                # Print a sample of the KML content
                print(f"\nFirst 500 characters of KML content:")
                print(kml_content.decode('utf-8', errors='ignore')[:500])
    except Exception as e:
        print(f"Error inspecting KMZ file: {e}")
        import traceback
        traceback.print_exc()


def load_gcps_from_kmz(kmz_path: str) -> List[Dict]:
    """
    Load GCPs from KMZ file.
    
    Args:
        kmz_path: Path to KMZ file
        
    Returns:
        List of GCP dictionaries
    """
    if not os.path.exists(kmz_path):
        print(f"KMZ file not found: {kmz_path}")
        return []
    
    print(f"Loading GCPs from: {kmz_path}")
    gcps = parse_kmz_file(kmz_path)
    return gcps
