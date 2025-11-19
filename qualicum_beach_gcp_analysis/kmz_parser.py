"""
Parser for Qualicum Beach Ground Control Point KMZ files.

KMZ files are ZIP archives containing KML (Keyhole Markup Language) XML files.
This parser extracts GCP coordinates and metadata from the KMZ/KML format.
"""

import zipfile
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional
import os
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
                return gcps
            
            # Parse the first KML file (usually there's only one)
            kml_content = kmz.read(kml_files[0])
            
            # Parse XML
            root = ET.fromstring(kml_content)
            
            # Define namespaces (KML uses namespaces)
            namespaces = {
                'kml': 'http://www.opengis.net/kml/2.2',
                'gx': 'http://www.google.com/kml/ext/2.2'
            }
            
            # Find all Placemark elements (these contain the GCPs)
            placemarks = root.findall('.//kml:Placemark', namespaces)
            if not placemarks:
                # Try without namespace (some KML files don't use namespaces)
                placemarks = root.findall('.//Placemark')
                namespaces = {}
            
            print(f"Found {len(placemarks)} placemarks in KMZ file")
            
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
        point = placemark.find('kml:Point', namespaces) if namespaces else placemark.find('Point')
        if point is not None:
            coord_elem = point.find('kml:coordinates', namespaces) if namespaces else point.find('coordinates')
            if coord_elem is not None and coord_elem.text:
                coords = _parse_coordinates(coord_elem.text)
        
        # If no Point, try LineString or Polygon (less common for GCPs)
        if coords is None:
            linestring = placemark.find('kml:LineString', namespaces) if namespaces else placemark.find('LineString')
            if linestring is not None:
                coord_elem = linestring.find('kml:coordinates', namespaces) if namespaces else linestring.find('coordinates')
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

