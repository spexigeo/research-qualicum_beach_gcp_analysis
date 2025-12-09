"""Visualization utilities for GCPs on basemaps."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import rasterio
from rasterio.plot import show
from PIL import Image

try:
    import h3
    H3_AVAILABLE = True
except ImportError:
    H3_AVAILABLE = False
    h3 = None


def visualize_gcps_on_basemap(
    gcps: List[Dict],
    basemap_path: str,
    output_path: Optional[str] = None,
    title: str = "Ground Control Points on Basemap",
    point_size: int = 50,
    point_color: str = 'red',
    show_labels: bool = True
):
    """
    Visualize GCPs overlaid on a basemap.
    
    Args:
        gcps: List of GCP dictionaries with 'lat', 'lon', and optionally 'id' keys
        basemap_path: Path to basemap GeoTIFF file
        output_path: Path to save visualization (if None, displays interactively)
        title: Title for the plot
        point_size: Size of GCP markers
        point_color: Color of GCP markers
        show_labels: Whether to show GCP IDs as labels
    """
    # Load basemap
    with rasterio.open(basemap_path) as src:
        basemap_data = src.read()
        basemap_bounds = src.bounds
        transform = src.transform
        
        # Convert to RGB if needed
        if basemap_data.shape[0] == 1:
            # Grayscale, convert to RGB
            basemap_rgb = np.stack([basemap_data[0]] * 3, axis=0)
        elif basemap_data.shape[0] == 3:
            basemap_rgb = basemap_data
        else:
            # Take first 3 bands
            basemap_rgb = basemap_data[:3]
        
        # Normalize to 0-255 range
        basemap_rgb = np.clip(basemap_rgb, 0, 255).astype(np.uint8)
        
        # Transpose for matplotlib (height, width, channels)
        basemap_display = basemap_rgb.transpose(1, 2, 0)
    
    # Create figure with larger size for better quality
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Display basemap with proper extent
    extent = [basemap_bounds.left, basemap_bounds.right, 
              basemap_bounds.bottom, basemap_bounds.top]
    
    # Ensure basemap is displayed correctly
    ax.imshow(basemap_display, extent=extent, origin='upper', interpolation='bilinear')
    
    # Set axis limits to match basemap bounds
    ax.set_xlim(basemap_bounds.left, basemap_bounds.right)
    ax.set_ylim(basemap_bounds.bottom, basemap_bounds.top)
    
    # Extract GCP coordinates
    gcp_lons = [gcp['lon'] for gcp in gcps]
    gcp_lats = [gcp['lat'] for gcp in gcps]
    gcp_ids = [gcp.get('id', f'GCP_{i}') for i, gcp in enumerate(gcps)]
    
    # Plot GCPs
    ax.scatter(gcp_lons, gcp_lats, 
              s=point_size, c=point_color, 
              marker='o', edgecolors='white', linewidths=2,
              alpha=0.8, zorder=10)
    
    # Add labels if requested
    if show_labels:
        for lon, lat, gcp_id in zip(gcp_lons, gcp_lats, gcp_ids):
            ax.annotate(gcp_id, (lon, lat), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, color='white',
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='black', alpha=0.7),
                       zorder=11)
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if output_path:
        # Save with high resolution
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Visualization saved to {output_path}")
        print(f"  Basemap bounds: {basemap_bounds}")
        print(f"  Image size: {basemap_display.shape[1]}x{basemap_display.shape[0]} pixels")
    else:
        plt.show()
    
    plt.close()


def export_basemap_as_png(
    basemap_path: str,
    output_path: str,
    dpi: int = 200
) -> str:
    """
    Export basemap GeoTIFF as PNG image.
    
    Args:
        basemap_path: Path to basemap GeoTIFF file
        output_path: Path to save PNG file
        dpi: Resolution for PNG export
        
    Returns:
        Path to saved PNG file
    """
    # Load basemap
    with rasterio.open(basemap_path) as src:
        basemap_data = src.read()
        basemap_bounds = src.bounds
        
        # Convert to RGB if needed
        if basemap_data.shape[0] == 1:
            # Grayscale, convert to RGB
            basemap_rgb = np.stack([basemap_data[0]] * 3, axis=0)
        elif basemap_data.shape[0] == 3:
            basemap_rgb = basemap_data
        else:
            # Take first 3 bands
            basemap_rgb = basemap_data[:3]
        
        # Normalize to 0-255 range
        basemap_rgb = np.clip(basemap_rgb, 0, 255).astype(np.uint8)
        
        # Transpose for PIL (height, width, channels)
        basemap_display = basemap_rgb.transpose(1, 2, 0)
    
    # Convert to PIL Image and save
    basemap_image = Image.fromarray(basemap_display)
    basemap_image.save(output_path, 'PNG', dpi=(dpi, dpi))
    
    print(f"Basemap exported to PNG: {output_path}")
    print(f"  Image size: {basemap_display.shape[1]}x{basemap_display.shape[0]} pixels")
    print(f"  Bounds: {basemap_bounds}")
    
    return output_path


def calculate_gcp_bbox(gcps: List[Dict], padding: float = 0.01) -> Tuple[float, float, float, float]:
    """
    Calculate bounding box from GCPs with optional padding.
    
    Args:
        gcps: List of GCP dictionaries with 'lat' and 'lon' keys
        padding: Padding in degrees to add to bbox
        
    Returns:
        Tuple of (min_lat, min_lon, max_lat, max_lon)
    """
    if not gcps:
        raise ValueError("Cannot calculate bbox from empty GCP list")
    
    lats = [gcp['lat'] for gcp in gcps]
    lons = [gcp['lon'] for gcp in gcps]
    
    min_lat = min(lats) - padding
    max_lat = max(lats) + padding
    min_lon = min(lons) - padding
    max_lon = max(lons) + padding
    
    # Ensure minimum size (avoid zero-size bbox)
    if max_lat - min_lat < 0.0001:
        center_lat = (min_lat + max_lat) / 2
        min_lat = center_lat - 0.0001
        max_lat = center_lat + 0.0001
    
    if max_lon - min_lon < 0.0001:
        center_lon = (min_lon + max_lon) / 2
        min_lon = center_lon - 0.0001
        max_lon = center_lon + 0.0001
    
    return (min_lat, min_lon, max_lat, max_lon)


def bbox_to_h3_cells(
    bbox: Tuple[float, float, float, float],
    resolution: int = 12
) -> List[str]:
    """
    Calculate H3 cells that overlap a bounding box.
    
    Args:
        bbox: Bounding box as (min_lat, min_lon, max_lat, max_lon)
        resolution: H3 resolution level (0-15, default 12)
        
    Returns:
        List of H3 cell identifiers that overlap the bounding box
    """
    if not H3_AVAILABLE:
        raise ImportError("h3 library is not installed. Install it with: pip install h3")
    
    min_lat, min_lon, max_lat, max_lon = bbox
    
    # For H3 v4, try using H3Shape for polygon_to_cells
    try:
        from h3 import H3Shape
        
        # Create GeoJSON polygon from bounding box
        # GeoJSON uses [lon, lat] format
        geojson_polygon = {
            "type": "Polygon",
            "coordinates": [[
                [min_lon, max_lat],  # Top-left
                [max_lon, max_lat],  # Top-right
                [max_lon, min_lat],  # Bottom-right
                [min_lon, min_lat],  # Bottom-left
                [min_lon, max_lat],  # Close polygon
            ]]
        }
        
        # Create H3Shape from GeoJSON
        shape = H3Shape.from_geojson(geojson_polygon)
        
        # Get all H3 cells that intersect with the polygon
        h3_cells = h3.polygon_to_cells(shape, resolution)
        
        # Convert to sorted list
        return sorted(list(h3_cells))
        
    except (ImportError, AttributeError, Exception) as e:
        # Fallback method: sample points in the bbox and get their cells
        # This is less precise but works without H3Shape
        cells_set = set()
        
        # Sample points across the bbox
        # Use adaptive sampling based on bbox size
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon
        
        # Calculate number of samples needed (aim for ~10-15 samples per degree)
        num_lat_samples = max(5, int(lat_range * 10))
        num_lon_samples = max(5, int(lon_range * 10))
        
        lat_step = lat_range / num_lat_samples if lat_range > 0 else 0.001
        lon_step = lon_range / num_lon_samples if lon_range > 0 else 0.001
        
        for i in range(num_lat_samples + 1):
            for j in range(num_lon_samples + 1):
                lat = min_lat + i * lat_step
                lon = min_lon + j * lon_step
                cell = h3.latlng_to_cell(lat, lon, resolution)
                cells_set.add(cell)
        
        return sorted(list(cells_set))


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import rasterio
from rasterio.plot import show
from PIL import Image

try:
    import h3
    H3_AVAILABLE = True
except ImportError:
    H3_AVAILABLE = False
    h3 = None


def visualize_gcps_on_basemap(
    gcps: List[Dict],
    basemap_path: str,
    output_path: Optional[str] = None,
    title: str = "Ground Control Points on Basemap",
    point_size: int = 50,
    point_color: str = 'red',
    show_labels: bool = True
):
    """
    Visualize GCPs overlaid on a basemap.
    
    Args:
        gcps: List of GCP dictionaries with 'lat', 'lon', and optionally 'id' keys
        basemap_path: Path to basemap GeoTIFF file
        output_path: Path to save visualization (if None, displays interactively)
        title: Title for the plot
        point_size: Size of GCP markers
        point_color: Color of GCP markers
        show_labels: Whether to show GCP IDs as labels
    """
    # Load basemap
    with rasterio.open(basemap_path) as src:
        basemap_data = src.read()
        basemap_bounds = src.bounds
        transform = src.transform
        
        # Convert to RGB if needed
        if basemap_data.shape[0] == 1:
            # Grayscale, convert to RGB
            basemap_rgb = np.stack([basemap_data[0]] * 3, axis=0)
        elif basemap_data.shape[0] == 3:
            basemap_rgb = basemap_data
        else:
            # Take first 3 bands
            basemap_rgb = basemap_data[:3]
        
        # Normalize to 0-255 range
        basemap_rgb = np.clip(basemap_rgb, 0, 255).astype(np.uint8)
        
        # Transpose for matplotlib (height, width, channels)
        basemap_display = basemap_rgb.transpose(1, 2, 0)
    
    # Create figure with larger size for better quality
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Display basemap with proper extent
    extent = [basemap_bounds.left, basemap_bounds.right, 
              basemap_bounds.bottom, basemap_bounds.top]
    
    # Ensure basemap is displayed correctly
    ax.imshow(basemap_display, extent=extent, origin='upper', interpolation='bilinear')
    
    # Set axis limits to match basemap bounds
    ax.set_xlim(basemap_bounds.left, basemap_bounds.right)
    ax.set_ylim(basemap_bounds.bottom, basemap_bounds.top)
    
    # Extract GCP coordinates
    gcp_lons = [gcp['lon'] for gcp in gcps]
    gcp_lats = [gcp['lat'] for gcp in gcps]
    gcp_ids = [gcp.get('id', f'GCP_{i}') for i, gcp in enumerate(gcps)]
    
    # Plot GCPs
    ax.scatter(gcp_lons, gcp_lats, 
              s=point_size, c=point_color, 
              marker='o', edgecolors='white', linewidths=2,
              alpha=0.8, zorder=10)
    
    # Add labels if requested
    if show_labels:
        for lon, lat, gcp_id in zip(gcp_lons, gcp_lats, gcp_ids):
            ax.annotate(gcp_id, (lon, lat), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, color='white',
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='black', alpha=0.7),
                       zorder=11)
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if output_path:
        # Save with high resolution
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Visualization saved to {output_path}")
        print(f"  Basemap bounds: {basemap_bounds}")
        print(f"  Image size: {basemap_display.shape[1]}x{basemap_display.shape[0]} pixels")
    else:
        plt.show()
    
    plt.close()


def export_basemap_as_png(
    basemap_path: str,
    output_path: str,
    dpi: int = 200
) -> str:
    """
    Export basemap GeoTIFF as PNG image.
    
    Args:
        basemap_path: Path to basemap GeoTIFF file
        output_path: Path to save PNG file
        dpi: Resolution for PNG export
        
    Returns:
        Path to saved PNG file
    """
    # Load basemap
    with rasterio.open(basemap_path) as src:
        basemap_data = src.read()
        basemap_bounds = src.bounds
        
        # Convert to RGB if needed
        if basemap_data.shape[0] == 1:
            # Grayscale, convert to RGB
            basemap_rgb = np.stack([basemap_data[0]] * 3, axis=0)
        elif basemap_data.shape[0] == 3:
            basemap_rgb = basemap_data
        else:
            # Take first 3 bands
            basemap_rgb = basemap_data[:3]
        
        # Normalize to 0-255 range
        basemap_rgb = np.clip(basemap_rgb, 0, 255).astype(np.uint8)
        
        # Transpose for PIL (height, width, channels)
        basemap_display = basemap_rgb.transpose(1, 2, 0)
    
    # Convert to PIL Image and save
    basemap_image = Image.fromarray(basemap_display)
    basemap_image.save(output_path, 'PNG', dpi=(dpi, dpi))
    
    print(f"Basemap exported to PNG: {output_path}")
    print(f"  Image size: {basemap_display.shape[1]}x{basemap_display.shape[0]} pixels")
    print(f"  Bounds: {basemap_bounds}")
    
    return output_path


def calculate_gcp_bbox(gcps: List[Dict], padding: float = 0.01) -> Tuple[float, float, float, float]:
    """
    Calculate bounding box from GCPs with optional padding.
    
    Args:
        gcps: List of GCP dictionaries with 'lat' and 'lon' keys
        padding: Padding in degrees to add to bbox
        
    Returns:
        Tuple of (min_lat, min_lon, max_lat, max_lon)
    """
    if not gcps:
        raise ValueError("Cannot calculate bbox from empty GCP list")
    
    lats = [gcp['lat'] for gcp in gcps]
    lons = [gcp['lon'] for gcp in gcps]
    
    min_lat = min(lats) - padding
    max_lat = max(lats) + padding
    min_lon = min(lons) - padding
    max_lon = max(lons) + padding
    
    # Ensure minimum size (avoid zero-size bbox)
    if max_lat - min_lat < 0.0001:
        center_lat = (min_lat + max_lat) / 2
        min_lat = center_lat - 0.0001
        max_lat = center_lat + 0.0001
    
    if max_lon - min_lon < 0.0001:
        center_lon = (min_lon + max_lon) / 2
        min_lon = center_lon - 0.0001
        max_lon = center_lon + 0.0001
    
    return (min_lat, min_lon, max_lat, max_lon)


def bbox_to_h3_cells(
    bbox: Tuple[float, float, float, float],
    resolution: int = 12
) -> List[str]:
    """
    Calculate H3 cells that overlap a bounding box.
    
    Args:
        bbox: Bounding box as (min_lat, min_lon, max_lat, max_lon)
        resolution: H3 resolution level (0-15, default 12)
        
    Returns:
        List of H3 cell identifiers that overlap the bounding box
    """
    if not H3_AVAILABLE:
        raise ImportError("h3 library is not installed. Install it with: pip install h3")
    
    min_lat, min_lon, max_lat, max_lon = bbox
    
    # For H3 v4, try using H3Shape for polygon_to_cells
    try:
        from h3 import H3Shape
        
        # Create GeoJSON polygon from bounding box
        # GeoJSON uses [lon, lat] format
        geojson_polygon = {
            "type": "Polygon",
            "coordinates": [[
                [min_lon, max_lat],  # Top-left
                [max_lon, max_lat],  # Top-right
                [max_lon, min_lat],  # Bottom-right
                [min_lon, min_lat],  # Bottom-left
                [min_lon, max_lat],  # Close polygon
            ]]
        }
        
        # Create H3Shape from GeoJSON
        shape = H3Shape.from_geojson(geojson_polygon)
        
        # Get all H3 cells that intersect with the polygon
        h3_cells = h3.polygon_to_cells(shape, resolution)
        
        # Convert to sorted list
        return sorted(list(h3_cells))
        
    except (ImportError, AttributeError, Exception) as e:
        # Fallback method: sample points in the bbox and get their cells
        # This is less precise but works without H3Shape
        cells_set = set()
        
        # Sample points across the bbox
        # Use adaptive sampling based on bbox size
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon
        
        # Calculate number of samples needed (aim for ~10-15 samples per degree)
        num_lat_samples = max(5, int(lat_range * 10))
        num_lon_samples = max(5, int(lon_range * 10))
        
        lat_step = lat_range / num_lat_samples if lat_range > 0 else 0.001
        lon_step = lon_range / num_lon_samples if lon_range > 0 else 0.001
        
        for i in range(num_lat_samples + 1):
            for j in range(num_lon_samples + 1):
                lat = min_lat + i * lat_step
                lon = min_lon + j * lon_step
                cell = h3.latlng_to_cell(lat, lon, resolution)
                cells_set.add(cell)
        
        return sorted(list(cells_set))


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import rasterio
from rasterio.plot import show
from PIL import Image

try:
    import h3
    H3_AVAILABLE = True
except ImportError:
    H3_AVAILABLE = False
    h3 = None


def visualize_gcps_on_basemap(
    gcps: List[Dict],
    basemap_path: str,
    output_path: Optional[str] = None,
    title: str = "Ground Control Points on Basemap",
    point_size: int = 50,
    point_color: str = 'red',
    show_labels: bool = True
):
    """
    Visualize GCPs overlaid on a basemap.
    
    Args:
        gcps: List of GCP dictionaries with 'lat', 'lon', and optionally 'id' keys
        basemap_path: Path to basemap GeoTIFF file
        output_path: Path to save visualization (if None, displays interactively)
        title: Title for the plot
        point_size: Size of GCP markers
        point_color: Color of GCP markers
        show_labels: Whether to show GCP IDs as labels
    """
    # Load basemap
    with rasterio.open(basemap_path) as src:
        basemap_data = src.read()
        basemap_bounds = src.bounds
        transform = src.transform
        
        # Convert to RGB if needed
        if basemap_data.shape[0] == 1:
            # Grayscale, convert to RGB
            basemap_rgb = np.stack([basemap_data[0]] * 3, axis=0)
        elif basemap_data.shape[0] == 3:
            basemap_rgb = basemap_data
        else:
            # Take first 3 bands
            basemap_rgb = basemap_data[:3]
        
        # Normalize to 0-255 range
        basemap_rgb = np.clip(basemap_rgb, 0, 255).astype(np.uint8)
        
        # Transpose for matplotlib (height, width, channels)
        basemap_display = basemap_rgb.transpose(1, 2, 0)
    
    # Create figure with larger size for better quality
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Display basemap with proper extent
    extent = [basemap_bounds.left, basemap_bounds.right, 
              basemap_bounds.bottom, basemap_bounds.top]
    
    # Ensure basemap is displayed correctly
    ax.imshow(basemap_display, extent=extent, origin='upper', interpolation='bilinear')
    
    # Set axis limits to match basemap bounds
    ax.set_xlim(basemap_bounds.left, basemap_bounds.right)
    ax.set_ylim(basemap_bounds.bottom, basemap_bounds.top)
    
    # Extract GCP coordinates
    gcp_lons = [gcp['lon'] for gcp in gcps]
    gcp_lats = [gcp['lat'] for gcp in gcps]
    gcp_ids = [gcp.get('id', f'GCP_{i}') for i, gcp in enumerate(gcps)]
    
    # Plot GCPs
    ax.scatter(gcp_lons, gcp_lats, 
              s=point_size, c=point_color, 
              marker='o', edgecolors='white', linewidths=2,
              alpha=0.8, zorder=10)
    
    # Add labels if requested
    if show_labels:
        for lon, lat, gcp_id in zip(gcp_lons, gcp_lats, gcp_ids):
            ax.annotate(gcp_id, (lon, lat), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, color='white',
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='black', alpha=0.7),
                       zorder=11)
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if output_path:
        # Save with high resolution
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Visualization saved to {output_path}")
        print(f"  Basemap bounds: {basemap_bounds}")
        print(f"  Image size: {basemap_display.shape[1]}x{basemap_display.shape[0]} pixels")
    else:
        plt.show()
    
    plt.close()


def export_basemap_as_png(
    basemap_path: str,
    output_path: str,
    dpi: int = 200
) -> str:
    """
    Export basemap GeoTIFF as PNG image.
    
    Args:
        basemap_path: Path to basemap GeoTIFF file
        output_path: Path to save PNG file
        dpi: Resolution for PNG export
        
    Returns:
        Path to saved PNG file
    """
    # Load basemap
    with rasterio.open(basemap_path) as src:
        basemap_data = src.read()
        basemap_bounds = src.bounds
        
        # Convert to RGB if needed
        if basemap_data.shape[0] == 1:
            # Grayscale, convert to RGB
            basemap_rgb = np.stack([basemap_data[0]] * 3, axis=0)
        elif basemap_data.shape[0] == 3:
            basemap_rgb = basemap_data
        else:
            # Take first 3 bands
            basemap_rgb = basemap_data[:3]
        
        # Normalize to 0-255 range
        basemap_rgb = np.clip(basemap_rgb, 0, 255).astype(np.uint8)
        
        # Transpose for PIL (height, width, channels)
        basemap_display = basemap_rgb.transpose(1, 2, 0)
    
    # Convert to PIL Image and save
    basemap_image = Image.fromarray(basemap_display)
    basemap_image.save(output_path, 'PNG', dpi=(dpi, dpi))
    
    print(f"Basemap exported to PNG: {output_path}")
    print(f"  Image size: {basemap_display.shape[1]}x{basemap_display.shape[0]} pixels")
    print(f"  Bounds: {basemap_bounds}")
    
    return output_path


def calculate_gcp_bbox(gcps: List[Dict], padding: float = 0.01) -> Tuple[float, float, float, float]:
    """
    Calculate bounding box from GCPs with optional padding.
    
    Args:
        gcps: List of GCP dictionaries with 'lat' and 'lon' keys
        padding: Padding in degrees to add to bbox
        
    Returns:
        Tuple of (min_lat, min_lon, max_lat, max_lon)
    """
    if not gcps:
        raise ValueError("Cannot calculate bbox from empty GCP list")
    
    lats = [gcp['lat'] for gcp in gcps]
    lons = [gcp['lon'] for gcp in gcps]
    
    min_lat = min(lats) - padding
    max_lat = max(lats) + padding
    min_lon = min(lons) - padding
    max_lon = max(lons) + padding
    
    # Ensure minimum size (avoid zero-size bbox)
    if max_lat - min_lat < 0.0001:
        center_lat = (min_lat + max_lat) / 2
        min_lat = center_lat - 0.0001
        max_lat = center_lat + 0.0001
    
    if max_lon - min_lon < 0.0001:
        center_lon = (min_lon + max_lon) / 2
        min_lon = center_lon - 0.0001
        max_lon = center_lon + 0.0001
    
    return (min_lat, min_lon, max_lat, max_lon)


def bbox_to_h3_cells(
    bbox: Tuple[float, float, float, float],
    resolution: int = 12
) -> List[str]:
    """
    Calculate H3 cells that overlap a bounding box.
    
    Args:
        bbox: Bounding box as (min_lat, min_lon, max_lat, max_lon)
        resolution: H3 resolution level (0-15, default 12)
        
    Returns:
        List of H3 cell identifiers that overlap the bounding box
    """
    if not H3_AVAILABLE:
        raise ImportError("h3 library is not installed. Install it with: pip install h3")
    
    min_lat, min_lon, max_lat, max_lon = bbox
    
    # For H3 v4, try using H3Shape for polygon_to_cells
    try:
        from h3 import H3Shape
        
        # Create GeoJSON polygon from bounding box
        # GeoJSON uses [lon, lat] format
        geojson_polygon = {
            "type": "Polygon",
            "coordinates": [[
                [min_lon, max_lat],  # Top-left
                [max_lon, max_lat],  # Top-right
                [max_lon, min_lat],  # Bottom-right
                [min_lon, min_lat],  # Bottom-left
                [min_lon, max_lat],  # Close polygon
            ]]
        }
        
        # Create H3Shape from GeoJSON
        shape = H3Shape.from_geojson(geojson_polygon)
        
        # Get all H3 cells that intersect with the polygon
        h3_cells = h3.polygon_to_cells(shape, resolution)
        
        # Convert to sorted list
        return sorted(list(h3_cells))
        
    except (ImportError, AttributeError, Exception) as e:
        # Fallback method: sample points in the bbox and get their cells
        # This is less precise but works without H3Shape
        cells_set = set()
        
        # Sample points across the bbox
        # Use adaptive sampling based on bbox size
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon
        
        # Calculate number of samples needed (aim for ~10-15 samples per degree)
        num_lat_samples = max(5, int(lat_range * 10))
        num_lon_samples = max(5, int(lon_range * 10))
        
        lat_step = lat_range / num_lat_samples if lat_range > 0 else 0.001
        lon_step = lon_range / num_lon_samples if lon_range > 0 else 0.001
        
        for i in range(num_lat_samples + 1):
            for j in range(num_lon_samples + 1):
                lat = min_lat + i * lat_step
                lon = min_lon + j * lon_step
                cell = h3.latlng_to_cell(lat, lon, resolution)
                cells_set.add(cell)
        
        return sorted(list(cells_set))

