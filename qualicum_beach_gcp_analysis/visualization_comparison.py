"""
Visualization functions for orthomosaic comparison reports.

Creates visualizations showing differences, seamlines, and quality metrics
for inclusion in PDF reports.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Dict, Optional, Tuple
import rasterio
import logging

try:
    from scipy.ndimage import zoom
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available. Install with: pip install scipy")

logger = logging.getLogger(__name__)


def create_error_visualization(
    ortho_array: np.ndarray,
    reference_array: np.ndarray,
    output_path: Path,
    title: str = "Error Visualization"
) -> Path:
    """
    Create a visualization showing the difference between orthomosaic and reference.
    
    Args:
        ortho_array: Orthomosaic array (grayscale)
        reference_array: Reference basemap array (grayscale)
        output_path: Path to save the visualization
        title: Title for the plot
        
    Returns:
        Path to saved image
    """
    # Resize ortho to match reference if shapes differ
    if ortho_array.shape != reference_array.shape:
        logger.info(f"Resizing ortho_array from {ortho_array.shape} to {reference_array.shape}")
        ortho_array = _resize_to_match(reference_array.shape, ortho_array)
    
    # Calculate difference
    diff = np.abs(ortho_array.astype(np.float32) - reference_array.astype(np.float32))
    
    # Normalize for visualization
    diff_norm = diff / (diff.max() + 1e-10) * 255
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original orthomosaic
    axes[0].imshow(ortho_array, cmap='gray')
    axes[0].set_title('Orthomosaic')
    axes[0].axis('off')
    
    # Reference basemap
    axes[1].imshow(reference_array, cmap='gray')
    axes[1].set_title('Reference Basemap')
    axes[1].axis('off')
    
    # Difference (error map)
    im = axes[2].imshow(diff_norm, cmap='hot', vmin=0, vmax=255)
    axes[2].set_title('Absolute Difference (Error Map)')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], label='Error (normalized)')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved error visualization to: {output_path}")
    return output_path


def create_seamline_visualization(
    ortho_array: np.ndarray,
    output_path: Path,
    threshold: float = 0.1,
    title: str = "Seamline Detection"
) -> Path:
    """
    Create a visualization highlighting detected seamlines.
    
    Args:
        ortho_array: Orthomosaic array (grayscale)
        output_path: Path to save the visualization
        threshold: Gradient threshold for seamline detection
        title: Title for the plot
        
    Returns:
        Path to saved image
    """
    # Calculate gradients
    grad_x = np.abs(np.gradient(ortho_array.astype(np.float32), axis=1))
    grad_y = np.abs(np.gradient(ortho_array.astype(np.float32), axis=0))
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize
    if grad_mag.max() > 0:
        grad_mag_norm = grad_mag / grad_mag.max()
    else:
        grad_mag_norm = grad_mag
    
    # Create seamline mask
    seamline_mask = grad_mag_norm > threshold
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original orthomosaic
    axes[0].imshow(ortho_array, cmap='gray')
    axes[0].set_title('Orthomosaic')
    axes[0].axis('off')
    
    # Seamlines overlaid
    axes[1].imshow(ortho_array, cmap='gray', alpha=0.7)
    axes[1].imshow(seamline_mask, cmap='Reds', alpha=0.5, interpolation='nearest')
    axes[1].set_title(f'Detected Seamlines (threshold={threshold})')
    axes[1].axis('off')
    
    # Add legend
    red_patch = mpatches.Patch(color='red', alpha=0.5, label='Seamline')
    axes[1].legend(handles=[red_patch], loc='upper right')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved seamline visualization to: {output_path}")
    return output_path


def _resize_to_match(target_shape: Tuple[int, int], source_array: np.ndarray) -> np.ndarray:
    """
    Resize source array to match target shape using interpolation.
    
    Args:
        target_shape: Target (height, width) shape
        source_array: Source array to resize
        
    Returns:
        Resized array
    """
    if source_array.shape == target_shape:
        return source_array
    
    if SCIPY_AVAILABLE:
        zoom_factors = (target_shape[0] / source_array.shape[0], 
                       target_shape[1] / source_array.shape[1])
        resized = zoom(source_array, zoom_factors, order=1)  # Bilinear interpolation
    else:
        # Fallback: use simple cropping/padding
        # Crop or pad to match target shape
        h, w = target_shape
        sh, sw = source_array.shape
        
        if sh > h or sw > w:
            # Crop to center
            start_h = (sh - h) // 2
            start_w = (sw - w) // 2
            resized = source_array[start_h:start_h+h, start_w:start_w+w]
        else:
            # Pad with zeros
            resized = np.zeros(target_shape, dtype=source_array.dtype)
            start_h = (h - sh) // 2
            start_w = (w - sw) // 2
            resized[start_h:start_h+sh, start_w:start_w+sw] = source_array
    
    return resized


def create_comparison_side_by_side(
    ortho_no_gcps: np.ndarray,
    ortho_with_gcps: np.ndarray,
    reference: np.ndarray,
    output_path: Path,
    title: str = "Orthomosaic Comparison"
) -> Path:
    """
    Create a side-by-side comparison of orthomosaics with and without GCPs.
    
    Args:
        ortho_no_gcps: Orthomosaic without GCPs (grayscale)
        ortho_with_gcps: Orthomosaic with GCPs (grayscale)
        reference: Reference basemap (grayscale)
        output_path: Path to save the visualization
        title: Title for the plot
        
    Returns:
        Path to saved image
    """
    # Handle shape mismatches by resizing to smallest common size or reference size
    # Use reference as the target size since it's typically smaller
    target_shape = reference.shape
    
    # Resize orthomosaics to match reference if needed
    if ortho_no_gcps.shape != target_shape:
        logger.info(f"Resizing ortho_no_gcps from {ortho_no_gcps.shape} to {target_shape}")
        ortho_no_gcps = _resize_to_match(target_shape, ortho_no_gcps)
    
    if ortho_with_gcps.shape != target_shape:
        logger.info(f"Resizing ortho_with_gcps from {ortho_with_gcps.shape} to {target_shape}")
        ortho_with_gcps = _resize_to_match(target_shape, ortho_with_gcps)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Reference (top left)
    axes[0, 0].imshow(reference, cmap='gray')
    axes[0, 0].set_title('Reference Basemap', fontweight='bold')
    axes[0, 0].axis('off')
    
    # Without GCPs (top right)
    axes[0, 1].imshow(ortho_no_gcps, cmap='gray')
    axes[0, 1].set_title('Orthomosaic (Without GCPs)', fontweight='bold')
    axes[0, 1].axis('off')
    
    # With GCPs (bottom left)
    axes[1, 0].imshow(ortho_with_gcps, cmap='gray')
    axes[1, 0].set_title('Orthomosaic (With GCPs)', fontweight='bold')
    axes[1, 0].axis('off')
    
    # Difference comparison (bottom right)
    diff_no_gcps = np.abs(ortho_no_gcps.astype(np.float32) - reference.astype(np.float32))
    diff_with_gcps = np.abs(ortho_with_gcps.astype(np.float32) - reference.astype(np.float32))
    
    # Normalize differences
    max_diff = max(diff_no_gcps.max(), diff_with_gcps.max())
    if max_diff > 0:
        diff_no_gcps_norm = diff_no_gcps / max_diff
        diff_with_gcps_norm = diff_with_gcps / max_diff
    else:
        diff_no_gcps_norm = diff_no_gcps
        diff_with_gcps_norm = diff_with_gcps
    
    # Show improvement (difference reduction)
    improvement = diff_no_gcps_norm - diff_with_gcps_norm
    
    im = axes[1, 1].imshow(improvement, cmap='RdYlGn', vmin=-1, vmax=1)
    axes[1, 1].set_title('Improvement Map\n(Red=Worse, Green=Better)', fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], label='Error Reduction')
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved comparison visualization to: {output_path}")
    return output_path


def create_metrics_summary_plot(
    metrics_no_gcps: Dict,
    metrics_with_gcps: Dict,
    output_path: Path,
    title: str = "Quality Metrics Comparison"
) -> Path:
    """
    Create a bar chart comparing quality metrics.
    
    Args:
        metrics_no_gcps: Metrics dictionary for orthomosaic without GCPs
        metrics_with_gcps: Metrics dictionary for orthomosaic with GCPs
        output_path: Path to save the visualization
        title: Title for the plot
        
    Returns:
        Path to saved image
    """
    overall_no = metrics_no_gcps.get('overall', {})
    overall_with = metrics_with_gcps.get('overall', {})
    
    # Prepare data
    metrics_data = {
        'RMSE': {
            'no_gcps': overall_no.get('rmse'),
            'with_gcps': overall_with.get('rmse'),
            'lower_is_better': True
        },
        'MAE': {
            'no_gcps': overall_no.get('mae'),
            'with_gcps': overall_with.get('mae'),
            'lower_is_better': True
        },
        'Similarity': {
            'no_gcps': overall_no.get('similarity'),
            'with_gcps': overall_with.get('similarity'),
            'lower_is_better': False
        },
        'Seamlines (%)': {
            'no_gcps': overall_no.get('seamline_percentage'),
            'with_gcps': overall_with.get('seamline_percentage'),
            'lower_is_better': True
        }
    }
    
    # Filter out None values
    metrics_data = {k: v for k, v in metrics_data.items() 
                   if v['no_gcps'] is not None and v['with_gcps'] is not None}
    
    if not metrics_data:
        logger.warning("No valid metrics data for plotting")
        return output_path
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(metrics_data))
    width = 0.35
    
    no_gcps_values = [metrics_data[k]['no_gcps'] for k in metrics_data.keys()]
    with_gcps_values = [metrics_data[k]['with_gcps'] for k in metrics_data.keys()]
    
    bars1 = ax.bar(x - width/2, no_gcps_values, width, label='Without GCPs', color='#ff7f7f')
    bars2 = ax.bar(x + width/2, with_gcps_values, width, label='With GCPs', color='#7fbf7f')
    
    ax.set_xlabel('Metric', fontweight='bold')
    ax.set_ylabel('Value', fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_data.keys())
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved metrics summary plot to: {output_path}")
    return output_path

