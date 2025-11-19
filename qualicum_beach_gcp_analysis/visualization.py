"""Visualization utilities for GCPs on basemaps."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import rasterio
from rasterio.plot import show


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
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Display basemap
    extent = [basemap_bounds.left, basemap_bounds.right, 
              basemap_bounds.bottom, basemap_bounds.top]
    ax.imshow(basemap_display, extent=extent, origin='upper')
    
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
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


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
    
    return (min_lat, min_lon, max_lat, max_lon)

