"""
Export GCPs to various formats for different software, especially MetaShape.
"""

from typing import List, Dict
import csv
import xml.etree.ElementTree as ET
from pathlib import Path


def export_to_metashape_csv(
    gcps: List[Dict],
    output_path: str,
    accuracy: float = 0.005
) -> str:
    """
    Export GCPs to MetaShape CSV format (tab-separated).
    
    MetaShape format: Label, X, Y, Z, Accuracy, Enabled
    - X, Y are longitude and latitude (WGS84)
    - Z is elevation
    - Accuracy is in meters
    - Enabled is 1 for enabled, 0 for disabled
    
    Args:
        gcps: List of GCP dictionaries with 'lat', 'lon', and optionally 'z', 'id', 'accuracy'
        output_path: Path to output file
        
    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        
        # Write header (MetaShape format)
        writer.writerow(['Label', 'X', 'Y', 'Z', 'Accuracy', 'Enabled'])
        
        for i, gcp in enumerate(gcps):
            label = gcp.get('id', gcp.get('label', f'GCP_{i+1:03d}'))
            lon = gcp.get('lon', gcp.get('longitude', 0.0))
            lat = gcp.get('lat', gcp.get('latitude', 0.0))
            z = gcp.get('z', gcp.get('elevation', gcp.get('altitude', 0.0)))
            # Use high accuracy (low value) for high weight in bundle adjustment
            # Default to 0.005m (5mm) if not specified - this gives very high weight
            # Can be overridden by accuracy parameter or individual GCP accuracy
            gcp_accuracy = gcp.get('accuracy', gcp.get('rmse', accuracy))
            # Use the more accurate (lower) value
            final_accuracy = min(gcp_accuracy, accuracy)
            enabled = '1'  # Default to enabled
            
            writer.writerow([label, lon, lat, z, final_accuracy, enabled])
    
    print(f"Exported {len(gcps)} GCPs to MetaShape CSV: {output_path}")
    return str(output_path)


def export_to_metashape_xml(
    gcps: List[Dict],
    output_path: str,
    accuracy: float = 0.005
) -> str:
    """
    Export GCPs as MetaShape marker file (XML format).
    
    This is an alternative format that MetaShape can import.
    
    Args:
        gcps: List of GCP dictionaries
        output_path: Path to output XML file
        
    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    root = ET.Element('document')
    chunks = ET.SubElement(root, 'chunks')
    chunk = ET.SubElement(chunks, 'chunk')
    markers = ET.SubElement(chunk, 'markers')
    
    for i, gcp in enumerate(gcps):
        marker = ET.SubElement(markers, 'marker')
        marker.set('label', gcp.get('id', gcp.get('label', f'GCP_{i+1:03d}')))
        marker.set('reference', 'true')
        
        # Position
        position = ET.SubElement(marker, 'position')
        lon = gcp.get('lon', gcp.get('longitude', 0.0))
        lat = gcp.get('lat', gcp.get('latitude', 0.0))
        z = gcp.get('z', gcp.get('elevation', gcp.get('altitude', 0.0)))
        position.set('x', str(lon))
        position.set('y', str(lat))
        position.set('z', str(z))
        
        # Accuracy - use high accuracy (low value) for high weight in bundle adjustment
        # Default to 0.005m (5mm) if not specified - this gives very high weight
        # Can be overridden by accuracy parameter or individual GCP accuracy
        gcp_accuracy = gcp.get('accuracy', gcp.get('rmse', accuracy))
        # Use the more accurate (lower) value
        final_accuracy = min(gcp_accuracy, accuracy)
        accuracy_elem = ET.SubElement(marker, 'accuracy')
        accuracy_elem.set('x', str(final_accuracy))
        accuracy_elem.set('y', str(final_accuracy))
        accuracy_elem.set('z', str(final_accuracy))
    
    tree = ET.ElementTree(root)
    ET.indent(tree, space='  ')
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    
    print(f"Exported {len(gcps)} GCPs to MetaShape XML: {output_path}")
    return str(output_path)


def export_to_metashape(
    gcps: List[Dict],
    output_path: str,
    format: str = 'csv',
    accuracy: float = 0.005
) -> str:
    """
    Export GCPs for MetaShape in the specified format.
    
    Args:
        gcps: List of GCP dictionaries
        output_path: Path to output file
        format: Export format ('csv' or 'xml')
        accuracy: GCP accuracy in meters. Lower values = higher weight in bundle adjustment.
                 Default 0.005m (5mm) gives very high weight.
        
    Returns:
        Path to saved file
    """
    if format == 'xml':
        return export_to_metashape_xml(gcps, output_path, accuracy=accuracy)
    else:
        return export_to_metashape_csv(gcps, output_path, accuracy=accuracy)
