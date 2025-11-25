"""
Quality Metrics for Orthomosaic Comparison.

Compares orthomosaics against reference basemaps (ESRI, OpenStreetMap)
to evaluate quality, accuracy, and identify issues like seamlines.
"""

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import Resampling as RasterioResampling
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def reproject_to_match(
    source_path: Path,
    reference_path: Path,
    output_path: Optional[Path] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Reproject source raster to match reference raster's CRS and bounds.
    
    Args:
        source_path: Path to source GeoTIFF
        reference_path: Path to reference GeoTIFF
        output_path: Optional path to save reprojected raster
        
    Returns:
        Tuple of (reprojected array, transform metadata)
    """
    with rasterio.open(reference_path) as ref:
        ref_crs = ref.crs
        ref_bounds = ref.bounds
        ref_width = ref.width
        ref_height = ref.height
    
    with rasterio.open(source_path) as src:
        src_crs = src.crs
        
        # Calculate transform
        transform, width, height = calculate_default_transform(
            src_crs,
            ref_crs,
            ref_width,
            ref_height,
            *ref_bounds
        )
        
        # Reproject
        reprojected = np.zeros((ref.count, height, width), dtype=src.dtypes[0])
        
        reproject(
            source=rasterio.band(src, range(1, src.count + 1)),
            destination=reprojected,
            src_transform=src.transform,
            src_crs=src_crs,
            dst_transform=transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear
        )
        
        metadata = {
            'transform': transform,
            'crs': ref_crs,
            'width': width,
            'height': height,
            'bounds': ref_bounds
        }
        
        # Save if output path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=ref.count,
                dtype=reprojected.dtype,
                crs=ref_crs,
                transform=transform,
                compress='lzw'
            ) as dst:
                dst.write(reprojected)
            logger.info(f"Saved reprojected raster to: {output_path}")
    
    return reprojected, metadata


def calculate_rmse(
    ortho_array: np.ndarray,
    reference_array: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> float:
    """
    Calculate Root Mean Square Error between orthomosaic and reference.
    
    Args:
        ortho_array: Orthomosaic array
        reference_array: Reference basemap array
        mask: Optional mask to exclude certain pixels
        
    Returns:
        RMSE value
    """
    if mask is not None:
        ortho_masked = ortho_array[mask]
        ref_masked = reference_array[mask]
    else:
        # Flatten arrays for comparison
        ortho_flat = ortho_array.flatten()
        ref_flat = reference_array.flatten()
        
        # Only compare valid pixels (non-zero, non-NaN)
        valid = (ortho_flat > 0) & (ref_flat > 0) & \
                np.isfinite(ortho_flat) & np.isfinite(ref_flat)
        
        ortho_masked = ortho_flat[valid]
        ref_masked = ref_flat[valid]
    
    if len(ortho_masked) == 0:
        return np.nan
    
    mse = np.mean((ortho_masked - ref_masked) ** 2)
    rmse = np.sqrt(mse)
    
    return rmse


def calculate_mae(
    ortho_array: np.ndarray,
    reference_array: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> float:
    """
    Calculate Mean Absolute Error between orthomosaic and reference.
    
    Args:
        ortho_array: Orthomosaic array
        reference_array: Reference basemap array
        mask: Optional mask to exclude certain pixels
        
    Returns:
        MAE value
    """
    if mask is not None:
        ortho_masked = ortho_array[mask]
        ref_masked = reference_array[mask]
    else:
        ortho_flat = ortho_array.flatten()
        ref_flat = reference_array.flatten()
        
        valid = (ortho_flat > 0) & (ref_flat > 0) & \
                np.isfinite(ortho_flat) & np.isfinite(ref_flat)
        
        ortho_masked = ortho_flat[valid]
        ref_masked = ref_flat[valid]
    
    if len(ortho_masked) == 0:
        return np.nan
    
    mae = np.mean(np.abs(ortho_masked - ref_masked))
    
    return mae


def detect_seamlines(
    ortho_array: np.ndarray,
    threshold: float = 0.1
) -> Dict:
    """
    Detect potential seamlines in orthomosaic by analyzing edge gradients.
    
    Args:
        ortho_array: Orthomosaic array (grayscale or first band)
        threshold: Gradient threshold for seamline detection
        
    Returns:
        Dictionary with seamline statistics
    """
    if len(ortho_array.shape) == 3:
        # Use first band for grayscale analysis
        gray = ortho_array[0] if ortho_array.shape[0] < ortho_array.shape[2] else ortho_array[:, :, 0]
    else:
        gray = ortho_array
    
    # Convert to float for gradient calculation
    gray_float = gray.astype(np.float32)
    
    # Calculate gradients
    grad_x = np.abs(np.gradient(gray_float, axis=1))
    grad_y = np.abs(np.gradient(gray_float, axis=0))
    
    # Combined gradient magnitude
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize gradient (0-1 scale)
    if grad_mag.max() > 0:
        grad_mag_norm = grad_mag / grad_mag.max()
    else:
        grad_mag_norm = grad_mag
    
    # Detect high-gradient regions (potential seamlines)
    seamline_mask = grad_mag_norm > threshold
    
    stats = {
        'total_pixels': gray.size,
        'seamline_pixels': np.sum(seamline_mask),
        'seamline_percentage': 100.0 * np.sum(seamline_mask) / gray.size,
        'max_gradient': float(grad_mag.max()),
        'mean_gradient': float(grad_mag.mean()),
        'gradient_std': float(grad_mag.std())
    }
    
    return stats


def calculate_structural_similarity(
    ortho_array: np.ndarray,
    reference_array: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> float:
    """
    Calculate Structural Similarity Index (SSIM) between orthomosaic and reference.
    
    This is a simplified version. For full SSIM, consider using scikit-image.
    
    Args:
        ortho_array: Orthomosaic array
        reference_array: Reference basemap array
        mask: Optional mask
        
    Returns:
        SSIM-like similarity score (0-1, higher is better)
    """
    if len(ortho_array.shape) == 3:
        ortho_gray = np.mean(ortho_array, axis=0) if ortho_array.shape[0] < ortho_array.shape[2] else np.mean(ortho_array, axis=2)
    else:
        ortho_gray = ortho_array
    
    if len(reference_array.shape) == 3:
        ref_gray = np.mean(reference_array, axis=0) if reference_array.shape[0] < reference_array.shape[2] else np.mean(reference_array, axis=2)
    else:
        ref_gray = reference_array
    
    # Normalize to 0-1 range
    ortho_norm = (ortho_gray - ortho_gray.min()) / (ortho_gray.max() - ortho_gray.min() + 1e-10)
    ref_norm = (ref_gray - ref_gray.min()) / (ref_gray.max() - ref_gray.min() + 1e-10)
    
    if mask is not None:
        ortho_masked = ortho_norm[mask]
        ref_masked = ref_norm[mask]
    else:
        valid = (ortho_norm > 0) & (ref_norm > 0) & \
                np.isfinite(ortho_norm) & np.isfinite(ref_norm)
        ortho_masked = ortho_norm[valid]
        ref_masked = ref_norm[valid]
    
    if len(ortho_masked) == 0:
        return 0.0
    
    # Simplified correlation-based similarity
    correlation = np.corrcoef(ortho_masked.flatten(), ref_masked.flatten())[0, 1]
    
    # Convert correlation to 0-1 scale (assuming correlation is -1 to 1)
    similarity = (correlation + 1) / 2.0
    
    return float(similarity)


def compare_orthomosaic_to_basemap(
    ortho_path: Path,
    basemap_path: Path,
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Comprehensive comparison of orthomosaic against reference basemap.
    
    Args:
        ortho_path: Path to orthomosaic GeoTIFF
        basemap_path: Path to reference basemap GeoTIFF
        output_dir: Optional directory for intermediate outputs
        
    Returns:
        Dictionary with comprehensive quality metrics
    """
    logger.info(f"Comparing orthomosaic {ortho_path.name} to basemap {basemap_path.name}")
    
    # Reproject orthomosaic to match basemap
    reprojected_path = None
    if output_dir:
        reprojected_path = output_dir / f"reprojected_{ortho_path.stem}.tif"
    
    ortho_reproj, metadata = reproject_to_match(ortho_path, basemap_path, reprojected_path)
    
    # Load reference basemap
    with rasterio.open(basemap_path) as ref:
        reference_array = ref.read()
    
    # Ensure same number of bands
    if ortho_reproj.shape[0] != reference_array.shape[0]:
        # Use first band of each
        ortho_reproj = ortho_reproj[0:1]
        reference_array = reference_array[0:1]
    
    # Calculate metrics for each band
    metrics = {
        'ortho_path': str(ortho_path),
        'basemap_path': str(basemap_path),
        'num_bands': ortho_reproj.shape[0],
        'bands': {}
    }
    
    for band_idx in range(ortho_reproj.shape[0]):
        ortho_band = ortho_reproj[band_idx]
        ref_band = reference_array[band_idx]
        
        # Calculate error metrics
        rmse = calculate_rmse(ortho_band, ref_band)
        mae = calculate_mae(ortho_band, ref_band)
        
        # Detect seamlines
        seamline_stats = detect_seamlines(ortho_band)
        
        # Calculate similarity
        similarity = calculate_structural_similarity(ortho_band, ref_band)
        
        metrics['bands'][f'band_{band_idx+1}'] = {
            'rmse': float(rmse) if not np.isnan(rmse) else None,
            'mae': float(mae) if not np.isnan(mae) else None,
            'similarity': float(similarity),
            'seamlines': seamline_stats
        }
    
    # Overall metrics (average across bands)
    if metrics['bands']:
        avg_rmse = np.mean([b['rmse'] for b in metrics['bands'].values() if b['rmse'] is not None])
        avg_mae = np.mean([b['mae'] for b in metrics['bands'].values() if b['mae'] is not None])
        avg_similarity = np.mean([b['similarity'] for b in metrics['bands'].values()])
        avg_seamline_pct = np.mean([b['seamlines']['seamline_percentage'] for b in metrics['bands'].values()])
        
        metrics['overall'] = {
            'rmse': float(avg_rmse) if not np.isnan(avg_rmse) else None,
            'mae': float(avg_mae) if not np.isnan(avg_mae) else None,
            'similarity': float(avg_similarity),
            'seamline_percentage': float(avg_seamline_pct)
        }
    
    logger.info(f"Comparison complete. Overall RMSE: {metrics.get('overall', {}).get('rmse', 'N/A')}")
    
    return metrics

