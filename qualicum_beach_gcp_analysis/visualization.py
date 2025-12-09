"""
Visualization functions for orthomosaic analysis.

Includes functions for visualizing GCPs, feature matches, and comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional
import rasterio
import logging

logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def visualize_feature_matches(
    ortho_array: np.ndarray,
    basemap_array: np.ndarray,
    match_pairs: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    output_path: Path,
    title: str = "Feature Matches",
    max_dimension: int = 2000
) -> Path:
    """
    Visualize feature matches between orthomosaic and basemap.
    
    Shows orthomosaic on left, basemap on right, with matched points
    connected by lines after RANSAC filtering.
    
    Args:
        ortho_array: Orthomosaic array (grayscale or RGB)
        basemap_array: Basemap array (grayscale or RGB)
        match_pairs: List of (src_point, dst_point) tuples in pixels
        output_path: Path to save visualization
        title: Title for the plot
        max_dimension: Maximum dimension for output (default: 2000 pixels)
        
    Returns:
        Path to saved visualization
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to grayscale if needed and normalize
    def normalize_to_uint8(arr):
        if len(arr.shape) == 3:
            arr = np.mean(arr, axis=0) if arr.shape[0] < arr.shape[2] else np.mean(arr, axis=2)
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            normalized = ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(arr, dtype=np.uint8)
        return normalized
    
    ortho_norm = normalize_to_uint8(ortho_array)
    basemap_norm = normalize_to_uint8(basemap_array)
    
    # Downsample if too large
    max_size = max(ortho_norm.shape[0], ortho_norm.shape[1], 
                   basemap_norm.shape[0], basemap_norm.shape[1])
    if max_size > max_dimension:
        scale_factor = max_dimension / max_size
        from scipy.ndimage import zoom
        ortho_h, ortho_w = ortho_norm.shape
        basemap_h, basemap_w = basemap_norm.shape
        new_ortho_h, new_ortho_w = int(ortho_h * scale_factor), int(ortho_w * scale_factor)
        new_basemap_h, new_basemap_w = int(basemap_h * scale_factor), int(basemap_w * scale_factor)
        
        ortho_norm = zoom(ortho_norm, (new_ortho_h / ortho_h, new_ortho_w / ortho_w), order=1)
        basemap_norm = zoom(basemap_norm, (new_basemap_h / basemap_h, new_basemap_w / basemap_w), order=1)
        
        # Scale match pairs
        match_pairs = [
            ((src[0] * scale_factor, src[1] * scale_factor),
             (dst[0] * scale_factor, dst[1] * scale_factor))
            for src, dst in match_pairs
        ]
    
    # Create side-by-side visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Display images
    ax1.imshow(ortho_norm, cmap='gray')
    ax1.set_title('Orthomosaic', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    ax2.imshow(basemap_norm, cmap='gray')
    ax2.set_title('Basemap', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Draw match lines
    # Calculate offset for basemap (it's on the right)
    ortho_width = ortho_norm.shape[1]
    basemap_width = basemap_norm.shape[1]
    
    # Draw matches
    for src_pt, dst_pt in match_pairs:
        # Source point in orthomosaic (left image)
        x1, y1 = src_pt[0], src_pt[1]
        # Destination point in basemap (right image, offset by ortho width)
        x2, y2 = ortho_width + dst_pt[0], dst_pt[1]
        
        # Draw line connecting matches
        ax1.plot([x1, x2], [y1, y2], 'b-', linewidth=0.5, alpha=0.6)
        
        # Draw points
        ax1.plot(x1, y1, 'ro', markersize=3, alpha=0.8)
        ax2.plot(dst_pt[0], dst_pt[1], 'go', markersize=3, alpha=0.8)
    
    # Add overall title
    fig.suptitle(f"{title}\n{len(match_pairs)} matches after RANSAC filtering", 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Feature match visualization saved to: {output_path}")
    return output_path
