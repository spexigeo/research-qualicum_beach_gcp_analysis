"""
Quality Metrics for Orthomosaic Comparison.

Compares orthomosaics against reference basemaps (ESRI, OpenStreetMap)
to evaluate quality, accuracy, and identify issues like seamlines.
"""

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import Resampling as RasterioResampling
from rasterio import Affine
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

# Try to import feature matching libraries
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available. Feature matching will be disabled. Install with: pip install opencv-python")

try:
    from skimage.feature import match_template
    from skimage.registration import phase_cross_correlation
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    logger.warning("scikit-image not available. Some feature matching methods will be disabled. Install with: pip install scikit-image")


def reproject_to_match_disk_only(
    source_path: Path,
    reference_path: Path,
    output_path: Path,
    tile_size: int = 2048
) -> Dict:
    """
    Memory-efficient reprojection that writes directly to disk without loading into memory.
    
    Args:
        source_path: Path to source GeoTIFF
        reference_path: Path to reference GeoTIFF
        output_path: Path to save reprojected raster (required)
        tile_size: Size of processing tiles (default: 2048 pixels)
        
    Returns:
        Dictionary with transform metadata
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with rasterio.open(reference_path) as ref:
        ref_crs = ref.crs
        ref_bounds = ref.bounds
        ref_width = ref.width
        ref_height = ref.height
        ref_count = ref.count
        ref_transform = ref.transform
        ref_dtype = ref.dtypes[0]
    
    with rasterio.open(source_path) as src:
        src_crs = src.crs
        src_count = src.count
        src_bounds = src.bounds
        src_dtype = src.dtypes[0]
        
        # Calculate transform (same logic as original function)
        try:
            transform, width, height = calculate_default_transform(
                src_crs,
                ref_crs,
                ref_width,
                ref_height,
                ref_bounds.left,
                ref_bounds.bottom,
                ref_bounds.right,
                ref_bounds.top
            )
        except Exception as e:
            logger.warning(f"Failed to calculate transform with reference bounds: {e}")
            logger.info("Attempting resolution-based approach...")
            
            from rasterio.warp import transform_bounds
            src_bounds_ref_crs = transform_bounds(
                src_crs, ref_crs,
                src_bounds.left, src_bounds.bottom,
                src_bounds.right, src_bounds.top
            )
            
            ref_pixel_size_x = abs(ref_transform[0])
            ref_pixel_size_y = abs(ref_transform[4])
            
            output_left = max(src_bounds_ref_crs[0], ref_bounds.left)
            output_bottom = max(src_bounds_ref_crs[1], ref_bounds.bottom)
            output_right = min(src_bounds_ref_crs[2], ref_bounds.right)
            output_top = min(src_bounds_ref_crs[3], ref_bounds.top)
            
            if output_right <= output_left or output_top <= output_bottom:
                raise ValueError(f"Invalid output bounds after intersection: left={output_left}, bottom={output_bottom}, right={output_right}, top={output_top}")
            
            width = int((output_right - output_left) / ref_pixel_size_x)
            height = int((output_top - output_bottom) / ref_pixel_size_y)
            
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid dimensions after bounds transformation: width={width}, height={height}")
            
            transform = Affine.translation(output_left, output_top) * Affine.scale(ref_pixel_size_x, -ref_pixel_size_y)
        
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid dimensions after transform calculation: width={width}, height={height}")
        
        output_count = min(src_count, ref_count)
        
        # Calculate file size to determine if BIGTIFF is needed
        bytes_per_pixel = np.dtype(ref_dtype).itemsize
        estimated_size_gb = (width * height * output_count * bytes_per_pixel) / (1024 ** 3)
        use_bigtiff = estimated_size_gb > 3.5
        
        # Create output file with compression
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=output_count,
            dtype=ref_dtype,
            crs=ref_crs,
            transform=transform,
            compress='lzw',
            BIGTIFF='YES' if use_bigtiff else 'NO',
            tiled=True,
            blockxsize=512,
            blockysize=512
        ) as dst:
            # Process in tiles to avoid memory issues
            for i in range(0, height, tile_size):
                for j in range(0, width, tile_size):
                    # Calculate tile window
                    win_height = min(tile_size, height - i)
                    win_width = min(tile_size, width - j)
                    
                    # Read corresponding tile from source
                    # Calculate source window bounds in destination CRS
                    tile_left = transform[2] + j * transform[0]
                    tile_top = transform[5] + i * transform[4]
                    tile_right = transform[2] + (j + win_width) * transform[0]
                    tile_bottom = transform[5] + (i + win_height) * transform[4]
                    
                    # Transform back to source CRS to get source window
                    from rasterio.warp import transform_bounds
                    src_tile_bounds = transform_bounds(
                        ref_crs, src_crs,
                        tile_left, tile_bottom,
                        tile_right, tile_top
                    )
                    
                    # Read source tile
                    src_window = src.window(*src_tile_bounds)
                    
                    # Reproject tile
                    tile_data = np.zeros((output_count, win_height, win_width), dtype=ref_dtype)
                    reproject(
                        source=rasterio.band(src, list(range(1, output_count + 1))),
                        destination=tile_data,
                        src_transform=src.transform,
                        src_crs=src_crs,
                        dst_transform=transform,
                        dst_crs=ref_crs,
                        resampling=Resampling.bilinear,
                        src_nodata=src.nodata,
                        dst_nodata=0
                    )
                    
                    # Write tile to output
                    dst.write(tile_data, window=rasterio.windows.Window(j, i, win_width, win_height))
        
        logger.info(f"Saved reprojected raster to: {output_path} (BIGTIFF={'YES' if use_bigtiff else 'NO'}, estimated size: {estimated_size_gb:.2f} GB)")
    
    metadata = {
        'transform': transform,
        'crs': ref_crs,
        'width': width,
        'height': height,
        'bounds': ref_bounds,
        'count': output_count
    }
    
    return metadata


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
        ref_count = ref.count
        ref_transform = ref.transform
    
    with rasterio.open(source_path) as src:
        src_crs = src.crs
        src_count = src.count
        src_bounds = src.bounds
        
        # Calculate transform - when dst_width and dst_height are provided,
        # the bounds should be in the destination (reference) CRS
        # However, if transformation fails, try using source bounds instead
        try:
            transform, width, height = calculate_default_transform(
                src_crs,
                ref_crs,
                ref_width,
                ref_height,
                ref_bounds.left,
                ref_bounds.bottom,
                ref_bounds.right,
                ref_bounds.top
            )
        except Exception as e:
            # If transformation fails with reference bounds, use resolution-based approach
            # This avoids the issue where calculate_default_transform tries to transform
            # points from source CRS when width/height are provided
            logger.warning(f"Failed to calculate transform with reference bounds: {e}")
            logger.info("Attempting resolution-based approach...")
            
            # Transform source bounds to reference CRS first
            from rasterio.warp import transform_bounds
            src_bounds_ref_crs = transform_bounds(
                src_crs, ref_crs,
                src_bounds.left, src_bounds.bottom,
                src_bounds.right, src_bounds.top
            )
            
            # Get reference pixel size
            ref_pixel_size_x = abs(ref_transform[0])
            ref_pixel_size_y = abs(ref_transform[4])
            
            # Use the intersection of source bounds (in ref CRS) and reference bounds
            # to determine the output extent
            output_left = max(src_bounds_ref_crs[0], ref_bounds.left)
            output_bottom = max(src_bounds_ref_crs[1], ref_bounds.bottom)
            output_right = min(src_bounds_ref_crs[2], ref_bounds.right)
            output_top = min(src_bounds_ref_crs[3], ref_bounds.top)
            
            # Ensure we have valid bounds
            if output_right <= output_left or output_top <= output_bottom:
                raise ValueError(f"Invalid output bounds after intersection: left={output_left}, bottom={output_bottom}, right={output_right}, top={output_top}")
            
            # Calculate dimensions based on intersection and reference pixel size
            width = int((output_right - output_left) / ref_pixel_size_x)
            height = int((output_top - output_bottom) / ref_pixel_size_y)
            
            # Ensure dimensions are valid
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid dimensions after bounds transformation: width={width}, height={height}")
            
            # Manually construct the transform to avoid transformation validation
            # that causes the "too many points failed to transform" error
            # We already have the bounds in the reference CRS, so we can construct
            # the transform directly using Affine transformation
            # Calculate dimensions from bounds and pixel size
            width = int((output_right - output_left) / ref_pixel_size_x)
            height = int((output_top - output_bottom) / ref_pixel_size_y)
            
            # Create transform: (left, pixel_width, 0, top, 0, -pixel_height)
            # Note: pixel_height is negative because Y increases downward in image coordinates
            # Affine is already imported from rasterio at the top of the file
            transform = Affine.translation(output_left, output_top) * Affine.scale(ref_pixel_size_x, -ref_pixel_size_y)
        
        # Verify dimensions are valid
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid dimensions after transform calculation: width={width}, height={height}")
        
        # Reproject - use source count but match reference dimensions
        # If source and reference have different band counts, use the minimum
        output_count = min(src_count, ref_count)
        reprojected = np.zeros((output_count, height, width), dtype=src.dtypes[0])
        
        # Reproject only the bands we need
        source_bands = list(range(1, output_count + 1))
        reproject(
            source=rasterio.band(src, source_bands),
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
            'bounds': ref_bounds,
            'count': output_count
        }
        
        # Save if output path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Use BIGTIFF=YES for large files (>4GB)
            # Calculate approximate file size to decide if BIGTIFF is needed
            bytes_per_pixel = np.dtype(reprojected.dtype).itemsize
            estimated_size_gb = (width * height * output_count * bytes_per_pixel) / (1024 ** 3)
            use_bigtiff = estimated_size_gb > 3.5  # Use BIGTIFF if >3.5GB to be safe
            
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=output_count,
                dtype=reprojected.dtype,
                crs=ref_crs,
                transform=transform,
                compress='lzw',
                BIGTIFF='YES' if use_bigtiff else 'NO',
                tiled=True,  # Use tiled format for better performance with large files
                blockxsize=512,
                blockysize=512
            ) as dst:
                dst.write(reprojected)
            logger.info(f"Saved reprojected raster to: {output_path} (BIGTIFF={'YES' if use_bigtiff else 'NO'}, estimated size: {estimated_size_gb:.2f} GB)")
    
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


def compute_feature_matching_2d_error(
    ortho_array: np.ndarray,
    reference_array: np.ndarray,
    method: str = 'sift',
    pixel_resolution: Optional[float] = None,
    log_file_path: Optional[Path] = None,
    max_spatial_error_meters: float = 10.0,
    use_tiles: bool = True,
    tile_size: int = 2048,
    use_gpu: bool = True
) -> Dict:
    """
    Compute 2D error measures using feature matching between orthomosaic and reference.
    
    This provides spatial error information (X, Y offsets) in addition to pixel-level errors.
    Optimized for large images with spatial constraints and optional GPU acceleration.
    
    Args:
        ortho_array: Orthomosaic array (grayscale or first band)
        reference_array: Reference basemap array (grayscale or first band)
        method: Feature matching method ('sift', 'orb', 'template', or 'phase')
        pixel_resolution: Pixel resolution in meters (for spatial constraints and meter conversion)
        log_file_path: Optional path to save detailed matching log
        max_spatial_error_meters: Maximum expected spatial error in meters (default: 10m)
        use_tiles: Whether to process large images in tiles (default: True)
        tile_size: Size of tiles for processing (default: 2048 pixels)
        use_gpu: Whether to use GPU acceleration if available (default: True)
        
    Returns:
        Dictionary with 2D error metrics including:
        - mean_offset_x, mean_offset_y: Average pixel offset
        - rmse_2d: 2D RMSE in pixels
        - num_matches: Number of matched features
        - match_confidence: Confidence score
    """
    # Convert to grayscale if needed
    if len(ortho_array.shape) == 3:
        ortho_gray = np.mean(ortho_array, axis=0) if ortho_array.shape[0] < ortho_array.shape[2] else np.mean(ortho_array, axis=2)
    else:
        ortho_gray = ortho_array
    
    if len(reference_array.shape) == 3:
        ref_gray = np.mean(reference_array, axis=0) if reference_array.shape[0] < reference_array.shape[2] else np.mean(reference_array, axis=2)
    else:
        ref_gray = reference_array
    
    # Normalize to uint8 for feature matching
    def normalize_to_uint8(arr):
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            normalized = ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(arr, dtype=np.uint8)
        return normalized
    
    ortho_norm = normalize_to_uint8(ortho_gray)
    ref_norm = normalize_to_uint8(ref_gray)
    
    # Calculate search window size in pixels based on max spatial error
    search_window_pixels = None
    if pixel_resolution and max_spatial_error_meters:
        search_window_pixels = int(np.ceil(max_spatial_error_meters / pixel_resolution))
        logger.info(f"Spatial constraint: searching within {search_window_pixels}x{search_window_pixels} pixel window "
                   f"(max error: {max_spatial_error_meters}m at {pixel_resolution:.4f}m/pixel)")
    
    # Check if images are large enough to benefit from tiling
    max_dimension = max(ortho_norm.shape[0], ortho_norm.shape[1], ref_norm.shape[0], ref_norm.shape[1])
    should_tile = use_tiles and max_dimension > tile_size * 2
    
    errors_2d = {
        'method': method,
        'mean_offset_x': None,
        'mean_offset_y': None,
        'rmse_2d': None,
        'num_matches': 0,
        'match_confidence': 0.0,
        'offsets': [],
        'match_pairs': [],  # List of (src_point, dst_point) tuples in pixels
        'mean_offset_x_meters': None,
        'mean_offset_y_meters': None,
        'rmse_2d_meters': None
    }
    
    # Setup log file if provided
    log_file = None
    if log_file_path:
        log_file_path = Path(log_file_path)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_file_path, 'w', encoding='utf-8')
        log_file.write(f"Feature Matching Results ({method.upper()})\n")
        log_file.write("=" * 60 + "\n\n")
    
    if method in ['sift', 'orb'] and CV2_AVAILABLE:
        # Use OpenCV feature matching
        try:
            # Check for GPU availability
            gpu_available = False
            if use_gpu:
                try:
                    # Check if OpenCV was built with CUDA support
                    if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                        gpu_available = True
                        logger.info("GPU acceleration available and enabled")
                except:
                    pass
            
            if method == 'sift':
                detector = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.01, edgeThreshold=20)
            else:  # orb
                detector = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=10)
            
            # Detect keypoints and descriptors
            # For large images, process in tiles
            if should_tile:
                logger.info(f"Processing large images ({ortho_norm.shape[1]}x{ortho_norm.shape[0]}) in tiles of {tile_size}x{tile_size}")
                kp1_list, des1_list = [], []
                kp2_list, des2_list = [], []
                
                # Process ortho in tiles
                for y in range(0, ortho_norm.shape[0], tile_size):
                    for x in range(0, ortho_norm.shape[1], tile_size):
                        y_end = min(y + tile_size, ortho_norm.shape[0])
                        x_end = min(x + tile_size, ortho_norm.shape[1])
                        tile = ortho_norm[y:y_end, x:x_end]
                        kp_tile, des_tile = detector.detectAndCompute(tile, None)
                        if kp_tile and des_tile is not None:
                            # Adjust keypoint coordinates to global image coordinates
                            for kp in kp_tile:
                                kp.pt = (kp.pt[0] + x, kp.pt[1] + y)
                            kp1_list.extend(kp_tile)
                            if des1_list:
                                des1_list = np.vstack([des1_list, des_tile])
                            else:
                                des1_list = des_tile
                
                # Process reference in tiles
                for y in range(0, ref_norm.shape[0], tile_size):
                    for x in range(0, ref_norm.shape[1], tile_size):
                        y_end = min(y + tile_size, ref_norm.shape[0])
                        x_end = min(x + tile_size, ref_norm.shape[1])
                        tile = ref_norm[y:y_end, x:x_end]
                        kp_tile, des_tile = detector.detectAndCompute(tile, None)
                        if kp_tile and des_tile is not None:
                            # Adjust keypoint coordinates to global image coordinates
                            for kp in kp_tile:
                                kp.pt = (kp.pt[0] + x, kp.pt[1] + y)
                            kp2_list.extend(kp_tile)
                            if des2_list:
                                des2_list = np.vstack([des2_list, des_tile])
                            else:
                                des2_list = des_tile
                
                kp1, des1 = kp1_list, des1_list
                kp2, des2 = kp2_list, des2_list
                logger.info(f"Detected {len(kp1)} keypoints in ortho, {len(kp2)} in reference (tiled processing)")
            else:
                # Process full images
                kp1, des1 = detector.detectAndCompute(ortho_norm, None)
                kp2, des2 = detector.detectAndCompute(ref_norm, None)
            
            if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
                # Match features with spatial constraints
                if search_window_pixels and search_window_pixels > 0:
                    # Use spatial constraint: for each ortho keypoint, only search nearby reference keypoints
                    logger.info(f"Using spatial constraint: {search_window_pixels}x{search_window_pixels} pixel search window")
                    good_matches = []
                    
                    # Convert keypoints to numpy arrays for efficient distance calculation
                    kp1_pts = np.array([kp.pt for kp in kp1])
                    kp2_pts = np.array([kp.pt for kp in kp2])
                    
                    if method == 'sift':
                        matcher = cv2.BFMatcher()
                        # For each ortho keypoint, find nearby reference keypoints
                        for i, kp1_pt in enumerate(kp1_pts):
                            # Find reference keypoints within search window
                            distances = np.sqrt(np.sum((kp2_pts - kp1_pt)**2, axis=1))
                            nearby_indices = np.where(distances <= search_window_pixels)[0]
                            
                            if len(nearby_indices) > 0:
                                # Match only against nearby descriptors
                                nearby_des2 = des2[nearby_indices]
                                matches = matcher.knnMatch(des1[i:i+1], nearby_des2, k=2)
                                for match_pair in matches:
                                    if len(match_pair) == 2:
                                        m, n = match_pair
                                        if m.distance < 0.85 * n.distance:
                                            # Adjust trainIdx to global index
                                            m.trainIdx = nearby_indices[m.trainIdx]
                                            good_matches.append(m)
                    else:  # orb
                        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                        # For each ortho keypoint, find nearby reference keypoints
                        for i, kp1_pt in enumerate(kp1_pts):
                            # Find reference keypoints within search window
                            distances = np.sqrt(np.sum((kp2_pts - kp1_pt)**2, axis=1))
                            nearby_indices = np.where(distances <= search_window_pixels)[0]
                            
                            if len(nearby_indices) > 0:
                                # Match only against nearby descriptors
                                nearby_des2 = des2[nearby_indices]
                                matches = matcher.knnMatch(des1[i:i+1], nearby_des2, k=2)
                                for match_pair in matches:
                                    if len(match_pair) == 2:
                                        m, n = match_pair
                                        if m.distance < 0.75 * n.distance:  # Lowe's ratio test
                                            # Adjust trainIdx to global index
                                            m.trainIdx = nearby_indices[m.trainIdx]
                                            good_matches.append(m)
                    
                    # Sort by distance and take best matches
                    good_matches = sorted(good_matches, key=lambda x: x.distance)[:500]
                    logger.info(f"Spatial-constrained matching found {len(good_matches)} candidate matches")
                else:
                    # Standard matching without spatial constraints
                    if method == 'sift':
                        matcher = cv2.BFMatcher()
                        matches = matcher.knnMatch(des1, des2, k=2)
                        # Apply Lowe's ratio test with more lenient threshold for different imagery
                        good_matches = []
                        for match_pair in matches:
                            if len(match_pair) == 2:
                                m, n = match_pair
                                if m.distance < 0.85 * n.distance:  # More lenient for different imagery
                                    good_matches.append(m)
                    else:  # orb
                        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                        good_matches = matcher.match(des1, des2)
                        good_matches = sorted(good_matches, key=lambda x: x.distance)[:200]  # More matches
                
                # Filter matches by geometric consistency (RANSAC)
                if len(good_matches) > 4:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    
                    # Use RANSAC to find inliers
                    try:
                        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        if mask is not None:
                            inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i]]
                            if len(inlier_matches) > 4:
                                good_matches = inlier_matches
                                logger.debug(f"RANSAC filtered to {len(good_matches)} inlier matches")
                    except:
                        pass  # If RANSAC fails, use all matches
                
                if len(good_matches) > 4:  # Need at least 4 matches
                    # Extract matched points
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    
                    # Store match pairs for visualization
                    match_pairs = [(tuple(src_pts[i, 0, :]), tuple(dst_pts[i, 0, :])) for i in range(len(good_matches))]
                    
                    # Calculate offsets
                    offsets = dst_pts - src_pts
                    offsets_flat = offsets.reshape(-1, 2)
                    
                    mean_offset_x = np.mean(offsets_flat[:, 0])
                    mean_offset_y = np.mean(offsets_flat[:, 1])
                    
                    # Calculate 2D RMSE
                    distances = np.sqrt(offsets_flat[:, 0]**2 + offsets_flat[:, 1]**2)
                    rmse_2d = np.sqrt(np.mean(distances**2))
                    
                    # Convert to meters if pixel resolution provided
                    mean_offset_x_meters = mean_offset_x * pixel_resolution if pixel_resolution else None
                    mean_offset_y_meters = mean_offset_y * pixel_resolution if pixel_resolution else None
                    rmse_2d_meters = rmse_2d * pixel_resolution if pixel_resolution else None
                    
                    # Confidence based on number of matches and consistency
                    match_confidence = min(1.0, len(good_matches) / 50.0)  # Normalize to 0-1
                    std_offset = np.std(distances)
                    if std_offset > 0:
                        match_confidence *= (1.0 / (1.0 + std_offset / 10.0))  # Penalize high variance
                    
                    # Validate shift is reasonable (not > 10% of image size)
                    max_reasonable_shift = max(ortho_norm.shape) * 0.1
                    if abs(mean_offset_x) < max_reasonable_shift and abs(mean_offset_y) < max_reasonable_shift:
                        errors_2d.update({
                            'mean_offset_x': float(mean_offset_x),
                            'mean_offset_y': float(mean_offset_y),
                            'rmse_2d': float(rmse_2d),
                            'num_matches': len(good_matches),
                            'match_confidence': float(match_confidence),
                            'offsets': offsets_flat.tolist(),
                            'match_pairs': match_pairs,
                            'mean_offset_x_meters': mean_offset_x_meters,
                            'mean_offset_y_meters': mean_offset_y_meters,
                            'rmse_2d_meters': rmse_2d_meters
                        })
                        
                        logger.info(f"Feature matching ({method}): {len(good_matches)} matches, "
                                  f"offset=({mean_offset_x:.2f}, {mean_offset_y:.2f}) px, "
                                  f"RMSE_2D={rmse_2d:.2f} px")
                        if pixel_resolution:
                            logger.info(f"  In meters: offset=({mean_offset_x_meters:.4f}, {mean_offset_y_meters:.4f}) m, "
                                      f"RMSE_2D={rmse_2d_meters:.4f} m")
                        
                        # Write detailed results to log file
                        if log_file:
                            log_file.write(f"Method: {method.upper()}\n")
                            log_file.write(f"Number of matches: {len(good_matches)}\n")
                            log_file.write(f"Match confidence: {match_confidence:.4f}\n\n")
                            log_file.write(f"Offset (pixels): X={mean_offset_x:.4f}, Y={mean_offset_y:.4f}\n")
                            log_file.write(f"RMSE 2D (pixels): {rmse_2d:.4f}\n")
                            if pixel_resolution:
                                log_file.write(f"Pixel resolution: {pixel_resolution:.4f} m/pixel\n")
                                log_file.write(f"Offset (meters): X={mean_offset_x_meters:.4f}, Y={mean_offset_y_meters:.4f}\n")
                                log_file.write(f"RMSE 2D (meters): {rmse_2d_meters:.4f}\n")
                            log_file.write(f"\nMatch Pairs (after RANSAC):\n")
                            log_file.write(f"{'Match':<8} {'Ortho Pixel (x,y)':<25} {'Basemap Pixel (x,y)':<25} {'Offset (px)':<20} {'Distance (px)':<15}\n")
                            log_file.write("-" * 100 + "\n")
                            for i, (src_pt, dst_pt) in enumerate(match_pairs):
                                offset_x = dst_pt[0] - src_pt[0]
                                offset_y = dst_pt[1] - src_pt[1]
                                dist = np.sqrt(offset_x**2 + offset_y**2)
                                log_file.write(f"{i+1:<8} ({src_pt[0]:>8.2f}, {src_pt[1]:>8.2f})  ({dst_pt[0]:>8.2f}, {dst_pt[1]:>8.2f})  "
                                            f"({offset_x:>7.2f}, {offset_y:>7.2f})  {dist:>10.2f}\n")
                            log_file.write("\n")
                    else:
                        logger.warning(f"Feature matching ({method}) shift ({mean_offset_x:.1f}, {mean_offset_y:.1f}) too large, rejecting")
                        if log_file:
                            log_file.write(f"WARNING: Shift too large, rejected\n")
                            log_file.write(f"  Shift: ({mean_offset_x:.2f}, {mean_offset_y:.2f}) pixels\n")
                            log_file.write(f"  Max reasonable: {max_reasonable_shift:.2f} pixels\n")
        except Exception as e:
            logger.warning(f"Feature matching ({method}) failed: {e}")
    
    elif method == 'phase' and SKIMAGE_AVAILABLE:
        # Use phase correlation (good for global shifts)
        try:
            shift, error, diffphase = phase_cross_correlation(ref_norm, ortho_norm, upsample_factor=10)
            mean_offset_x = float(shift[1])  # Note: phase_cross_correlation returns (row, col)
            mean_offset_y = float(shift[0])
            
            # Convert to meters if pixel resolution provided
            mean_offset_x_meters = mean_offset_x * pixel_resolution if pixel_resolution else None
            mean_offset_y_meters = mean_offset_y * pixel_resolution if pixel_resolution else None
            rmse_2d_meters = np.sqrt(mean_offset_x**2 + mean_offset_y**2) * pixel_resolution if pixel_resolution else None
            
            # Convert to meters if pixel resolution provided
            mean_offset_x_meters = mean_offset_x * pixel_resolution if pixel_resolution else None
            mean_offset_y_meters = mean_offset_y * pixel_resolution if pixel_resolution else None
            rmse_2d_meters = np.sqrt(mean_offset_x**2 + mean_offset_y**2) * pixel_resolution if pixel_resolution else None
            
            # Validate shift is reasonable
            max_reasonable_shift = max(ortho_norm.shape) * 0.1
            if abs(mean_offset_x) < max_reasonable_shift and abs(mean_offset_y) < max_reasonable_shift:
                errors_2d.update({
                    'mean_offset_x': mean_offset_x,
                    'mean_offset_y': mean_offset_y,
                    'rmse_2d': float(np.sqrt(mean_offset_x**2 + mean_offset_y**2)),
                    'num_matches': 1,
                    'match_confidence': float(1.0 - min(1.0, error)),  # Lower error = higher confidence
                    'mean_offset_x_meters': mean_offset_x_meters,
                    'mean_offset_y_meters': mean_offset_y_meters,
                    'rmse_2d_meters': rmse_2d_meters
                })
                logger.info(f"Phase correlation: shift=({mean_offset_x:.2f}, {mean_offset_y:.2f}) px, error={error:.4f}")
                if pixel_resolution:
                    logger.info(f"  In meters: shift=({mean_offset_x_meters:.4f}, {mean_offset_y_meters:.4f}) m")
                if log_file:
                    log_file.write(f"Method: PHASE_CORRELATION\n")
                    log_file.write(f"Offset (pixels): X={mean_offset_x:.4f}, Y={mean_offset_y:.4f}\n")
                    log_file.write(f"RMSE 2D (pixels): {np.sqrt(mean_offset_x**2 + mean_offset_y**2):.4f}\n")
                    if pixel_resolution:
                        log_file.write(f"Offset (meters): X={mean_offset_x_meters:.4f}, Y={mean_offset_y_meters:.4f}\n")
                        log_file.write(f"RMSE 2D (meters): {rmse_2d_meters:.4f}\n")
            else:
                logger.warning(f"Phase correlation shift ({mean_offset_x:.1f}, {mean_offset_y:.1f}) too large, rejecting")
        except Exception as e:
            logger.warning(f"Phase correlation failed: {e}")
    
    elif method == 'template' and SKIMAGE_AVAILABLE:
        # Template matching (slower but can find local matches)
        try:
            # Sample a template from the center of reference
            h, w = ref_norm.shape
            template_size = min(100, h//4, w//4)
            template = ref_norm[h//2:h//2+template_size, w//2:w//2+template_size]
            
            # Find template in orthomosaic
            result = match_template(ortho_norm, template)
            ij = np.unravel_index(np.argmax(result), result.shape)
            
            # Calculate offset
            offset_x = ij[1] - w//2
            offset_y = ij[0] - h//2
            
            # Validate shift is reasonable
            max_reasonable_shift = max(h, w) * 0.1
            if abs(offset_x) < max_reasonable_shift and abs(offset_y) < max_reasonable_shift:
                errors_2d.update({
                    'mean_offset_x': float(offset_x),
                    'mean_offset_y': float(offset_y),
                    'rmse_2d': float(np.sqrt(offset_x**2 + offset_y**2)),
                    'num_matches': 1,
                    'match_confidence': float(result[ij]),
                })
                logger.info(f"Template matching: offset=({offset_x:.2f}, {offset_y:.2f}) px")
            else:
                logger.warning(f"Template matching shift ({offset_x:.1f}, {offset_y:.1f}) too large, rejecting")
        except Exception as e:
            logger.warning(f"Template matching failed: {e}")
    
    elif method == 'lines' and CV2_AVAILABLE:
        # Use line/edge-based matching for different imagery types
        try:
            # Detect edges using Canny
            edges1 = cv2.Canny(ortho_norm, 50, 150)
            edges2 = cv2.Canny(ref_norm, 50, 150)
            
            # Use phase correlation on edge images for shift estimate
            if SKIMAGE_AVAILABLE:
                try:
                    from skimage.registration import phase_cross_correlation
                    shift, error, diffphase = phase_cross_correlation(edges2, edges1, upsample_factor=10)
                    mean_offset_x = float(shift[1])
                    mean_offset_y = float(shift[0])
                    
                    # Validate shift is reasonable (not > 10% of image size)
            max_reasonable_shift = max(ortho_norm.shape) * 0.1
            if abs(mean_offset_x) < max_reasonable_shift and abs(mean_offset_y) < max_reasonable_shift:
                rmse_2d = float(np.sqrt(mean_offset_x**2 + mean_offset_y**2))
                mean_offset_x_meters = mean_offset_x * pixel_resolution if pixel_resolution else None
                mean_offset_y_meters = mean_offset_y * pixel_resolution if pixel_resolution else None
                rmse_2d_meters = rmse_2d * pixel_resolution if pixel_resolution else None
                
                errors_2d.update({
                    'mean_offset_x': mean_offset_x,
                    'mean_offset_y': mean_offset_y,
                    'rmse_2d': rmse_2d,
                    'num_matches': int(np.sum(edges1 > 0) + np.sum(edges2 > 0)),  # Edge pixel count
                    'match_confidence': float(1.0 - min(1.0, error)),
                    'mean_offset_x_meters': mean_offset_x_meters,
                    'mean_offset_y_meters': mean_offset_y_meters,
                    'rmse_2d_meters': rmse_2d_meters
                })
                logger.info(f"Line/edge matching: offset=({mean_offset_x:.2f}, {mean_offset_y:.2f}) px, error={error:.4f}")
                if pixel_resolution:
                    logger.info(f"  In meters: offset=({mean_offset_x_meters:.4f}, {mean_offset_y_meters:.4f}) m")
                if log_file:
                    log_file.write(f"Method: LINES/EDGES\n")
                    log_file.write(f"Offset (pixels): X={mean_offset_x:.4f}, Y={mean_offset_y:.4f}\n")
                    log_file.write(f"RMSE 2D (pixels): {rmse_2d:.4f}\n")
                    if pixel_resolution:
                        log_file.write(f"Offset (meters): X={mean_offset_x_meters:.4f}, Y={mean_offset_y_meters:.4f}\n")
                        log_file.write(f"RMSE 2D (meters): {rmse_2d_meters:.4f}\n")
            else:
                logger.warning(f"Line/edge matching shift ({mean_offset_x:.1f}, {mean_offset_y:.1f}) too large, rejecting")
                if log_file:
                    log_file.write(f"WARNING: Shift too large, rejected\n")
                    log_file.write(f"  Shift: ({mean_offset_x:.2f}, {mean_offset_y:.2f}) pixels\n")
                except Exception as e:
                    logger.debug(f"Phase correlation on edges failed: {e}")
        except Exception as e:
            logger.warning(f"Line/edge matching failed: {e}")
    
    # Close log file if opened
    if log_file:
        log_file.close()
        logger.info(f"Feature matching log saved to: {log_file_path}")
    
    return errors_2d


def downsample_ortho_to_basemap_resolution(
    ortho_path: Path,
    basemap_path: Path,
    output_path: Path
) -> Path:
    """
    Downsample orthomosaic to match basemap resolution for feature matching.
    
    Args:
        ortho_path: Path to high-resolution orthomosaic
        basemap_path: Path to reference basemap (target resolution)
        output_path: Path to save downsampled orthomosaic
        
    Returns:
        Path to downsampled orthomosaic
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with rasterio.open(basemap_path) as basemap:
        target_resolution = abs(basemap.transform[0])  # Pixel size in meters
        target_crs = basemap.crs
        target_bounds = basemap.bounds
    
    with rasterio.open(ortho_path) as ortho:
        ortho_resolution = abs(ortho.transform[0])  # Pixel size in meters
        ortho_crs = ortho.crs
        
        # Calculate downsampling factor
        downsample_factor = ortho_resolution / target_resolution
        
        logger.info(f"Ortho resolution: {ortho_resolution:.4f}m/pixel")
        logger.info(f"Basemap resolution: {target_resolution:.4f}m/pixel")
        logger.info(f"Downsampling factor: {downsample_factor:.2f}x")
        
        # Calculate transform and dimensions for downsampled/reprojected image
        from rasterio.warp import calculate_default_transform, reproject, Resampling
        
        if downsample_factor < 1.0:
            logger.warning(f"Ortho resolution ({ortho_resolution:.4f}m) is lower than basemap ({target_resolution:.4f}m). Reprojecting to match CRS and bounds.")
            # Just reproject to match CRS and bounds (no downsampling)
            transform, width, height = calculate_default_transform(
                ortho_crs, target_crs,
                ortho.width, ortho.height,
                target_bounds.left, target_bounds.bottom,
                target_bounds.right, target_bounds.top
            )
        else:
            # Calculate new dimensions for downsampling
            new_width = int(ortho.width / downsample_factor)
            new_height = int(ortho.height / downsample_factor)
            
            # Calculate transform for downsampled image matching basemap resolution
            # First transform bounds to target CRS
            from rasterio.warp import transform_bounds
            ortho_bounds_target_crs = transform_bounds(
                ortho_crs, target_crs,
                ortho.bounds.left, ortho.bounds.bottom,
                ortho.bounds.right, ortho.bounds.top
            )
            
            # Calculate transform with target resolution
            transform, width, height = calculate_default_transform(
                ortho_crs, target_crs,
                new_width, new_height,
                ortho_bounds_target_crs[0], ortho_bounds_target_crs[1],
                ortho_bounds_target_crs[2], ortho_bounds_target_crs[3]
            )
        
        # Create output file
        profile = ortho.profile.copy()
        profile.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'compress': 'lzw',
            'tiled': True
        })
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            for i in range(1, ortho.count + 1):
                reproject(
                    source=rasterio.band(ortho, i),
                    destination=rasterio.band(dst, i),
                    src_transform=ortho.transform,
                    src_crs=ortho_crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear
                )
    
    logger.info(f"Downsampled orthomosaic saved to: {output_path}")
    return output_path


def compare_orthomosaic_to_basemap(
    ortho_path: Path,
    basemap_path: Path,
    output_dir: Optional[Path] = None,
    feature_matching_method: str = 'orb'
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
    min_bands = min(ortho_reproj.shape[0], reference_array.shape[0])
    if ortho_reproj.shape[0] != reference_array.shape[0]:
        logger.info(f"Band count mismatch: ortho has {ortho_reproj.shape[0]} bands, reference has {reference_array.shape[0]} bands")
        logger.info(f"Using first {min_bands} band(s) for comparison")
        # Use matching number of bands
        ortho_reproj = ortho_reproj[:min_bands]
        reference_array = reference_array[:min_bands]
    
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
        
        # Compute 2D error using feature matching (only for first band to avoid redundancy)
        errors_2d = {}
        if band_idx == 0:
            # Get pixel resolution from reference basemap for meter conversion
            with rasterio.open(basemap_path) as ref:
                pixel_resolution = abs(ref.transform[0])  # Pixel size in meters
            
            # Setup log file path for detailed matching results
            log_file_path = None
            if output_dir:
                log_dir = Path(output_dir) / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file_path = log_dir / f"feature_matching_{Path(ortho_path).stem}_{Path(basemap_path).stem}.log"
            
            # Use specified method (default: ORB only)
            if feature_matching_method.lower() == 'orb' and CV2_AVAILABLE:
                try:
                    errors_2d = compute_feature_matching_2d_error(
                        ortho_band, ref_band, 
                        method='orb',
                        pixel_resolution=pixel_resolution,
                        log_file_path=log_file_path
                    )
                except Exception as e:
                    logger.warning(f"ORB feature matching failed: {e}")
            elif feature_matching_method.lower() == 'sift' and CV2_AVAILABLE:
                try:
                    errors_2d = compute_feature_matching_2d_error(
                        ortho_band, ref_band, 
                        method='sift',
                        pixel_resolution=pixel_resolution,
                        log_file_path=log_file_path
                    )
                except Exception as e:
                    logger.warning(f"SIFT feature matching failed: {e}")
            else:
                # Fallback: try available methods
                methods_to_try = []
                if CV2_AVAILABLE:
                    methods_to_try.extend(['orb', 'sift'])
                if SKIMAGE_AVAILABLE:
                    methods_to_try.extend(['phase', 'template'])
                
                best_errors_2d = None
                best_confidence = 0.0
                
                for method in methods_to_try:
                    try:
                        errors = compute_feature_matching_2d_error(
                            ortho_band, ref_band, 
                            method=method,
                            pixel_resolution=pixel_resolution,
                            log_file_path=log_file_path
                        )
                        if errors['match_confidence'] > best_confidence:
                            best_confidence = errors['match_confidence']
                            best_errors_2d = errors
                    except Exception as e:
                        logger.debug(f"Feature matching method {method} failed: {e}")
                
                if best_errors_2d:
                    errors_2d = best_errors_2d
            
            # Create visualization if matches found
            if errors_2d and errors_2d.get('match_pairs') and len(errors_2d['match_pairs']) > 0:
                try:
                    from .visualization import visualize_feature_matches
                    vis_dir = Path(output_dir) / "visualizations" if output_dir else Path("outputs/visualizations")
                    vis_dir.mkdir(parents=True, exist_ok=True)
                    vis_path = vis_dir / f"feature_matches_{Path(ortho_path).stem}_{Path(basemap_path).stem}.png"
                    visualize_feature_matches(
                        ortho_band, ref_band,
                        errors_2d['match_pairs'],
                        vis_path,
                        title=f"Feature Matches: {Path(ortho_path).stem} vs {Path(basemap_path).stem}"
                    )
                    logger.info(f"Feature match visualization saved to: {vis_path}")
                except Exception as e:
                    logger.warning(f"Could not create feature match visualization: {e}")
        
        metrics['bands'][f'band_{band_idx+1}'] = {
            'rmse': float(rmse) if not np.isnan(rmse) else None,
            'mae': float(mae) if not np.isnan(mae) else None,
            'similarity': float(similarity),
            'seamlines': seamline_stats,
            'errors_2d': errors_2d if errors_2d else {}
        }
    
    # Overall metrics (average across bands)
    if metrics['bands']:
        avg_rmse = np.mean([b['rmse'] for b in metrics['bands'].values() if b['rmse'] is not None])
        avg_mae = np.mean([b['mae'] for b in metrics['bands'].values() if b['mae'] is not None])
        avg_similarity = np.mean([b['similarity'] for b in metrics['bands'].values()])
        avg_seamline_pct = np.mean([b['seamlines']['seamline_percentage'] for b in metrics['bands'].values()])
        
        # Get 2D error metrics from first band (if available)
        errors_2d_overall = {}
        first_band = list(metrics['bands'].values())[0]
        if first_band.get('errors_2d') and first_band['errors_2d'].get('rmse_2d') is not None:
            errors_2d_overall = {
                'mean_offset_x_pixels': first_band['errors_2d'].get('mean_offset_x'),
                'mean_offset_y_pixels': first_band['errors_2d'].get('mean_offset_y'),
                'rmse_2d_pixels': first_band['errors_2d'].get('rmse_2d'),
                'num_matches': first_band['errors_2d'].get('num_matches', 0),
                'match_confidence': first_band['errors_2d'].get('match_confidence', 0.0),
                'method': first_band['errors_2d'].get('method', 'unknown')
            }
        
        metrics['overall'] = {
            'rmse': float(avg_rmse) if not np.isnan(avg_rmse) else None,
            'mae': float(avg_mae) if not np.isnan(avg_mae) else None,
            'similarity': float(avg_similarity),
            'seamline_percentage': float(avg_seamline_pct),
            'errors_2d': errors_2d_overall
        }
    
    logger.info(f"Comparison complete. Overall RMSE: {metrics.get('overall', {}).get('rmse', 'N/A')}")
    
    return metrics


def compare_orthomosaic_to_basemap_memory_efficient(
    ortho_path: Path,
    basemap_path: Path,
    output_dir: Optional[Path] = None,
    tile_size: int = 2048,
    max_downsample_for_matching: int = 2000
) -> Dict:
    """
    Memory-efficient comparison that processes in tiles without loading entire arrays.
    
    Args:
        ortho_path: Path to orthomosaic GeoTIFF
        basemap_path: Path to reference basemap GeoTIFF
        output_dir: Optional directory for intermediate outputs
        tile_size: Size of processing tiles (default: 2048 pixels)
        max_downsample_for_matching: Maximum dimension for feature matching (default: 2000)
        
    Returns:
        Dictionary with comprehensive quality metrics
    """
    logger.info(f"Comparing orthomosaic {ortho_path.name} to basemap {basemap_path.name} (memory-efficient mode)")
    
    # Reproject orthomosaic to match basemap (disk-only, no memory load)
    reprojected_path = None
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        reprojected_path = output_dir / f"reprojected_{ortho_path.stem}.tif"
    
    if not reprojected_path:
        raise ValueError("output_dir must be provided for memory-efficient mode")
    
    # Check if reprojected file already exists
    if reprojected_path.exists():
        logger.info(f"Using existing reprojected file: {reprojected_path}")
    else:
        logger.info("Reprojecting orthomosaic to match basemap (disk-only mode)...")
        metadata = reproject_to_match_disk_only(ortho_path, basemap_path, reprojected_path, tile_size=tile_size)
    
    # Open both files for tile-based processing
    with rasterio.open(reprojected_path) as ortho_src, rasterio.open(basemap_path) as ref_src:
        # Get metadata
        ortho_count = ortho_src.count
        ref_count = ref_src.count
        min_bands = min(ortho_count, ref_count)
        width = ortho_src.width
        height = ortho_src.height
        
        # Initialize accumulators for metrics
        total_rmse_sq = 0.0
        total_mae = 0.0
        total_pixels = 0
        total_seamline_pixels = 0
        
        # Process in tiles
        logger.info(f"Processing comparison in tiles of {tile_size}x{tile_size} pixels...")
        for i in range(0, height, tile_size):
            for j in range(0, width, tile_size):
                win_height = min(tile_size, height - i)
                win_width = min(tile_size, width - j)
                window = rasterio.windows.Window(j, i, win_width, win_height)
                
                # Read tiles
                ortho_tile = ortho_src.read(window=window)
                ref_tile = ref_src.read(window=window)
                
                # Process each band
                for band_idx in range(min_bands):
                    ortho_band = ortho_tile[band_idx]
                    ref_band = ref_tile[band_idx]
                    
                    # Create valid mask (non-zero, non-NaN)
                    valid = (ortho_band > 0) & (ref_band > 0) & \
                            np.isfinite(ortho_band) & np.isfinite(ref_band)
                    
                    if np.any(valid):
                        ortho_valid = ortho_band[valid]
                        ref_valid = ref_band[valid]
                        
                        # Accumulate RMSE components
                        diff = ortho_valid - ref_valid
                        total_rmse_sq += np.sum(diff ** 2)
                        total_mae += np.sum(np.abs(diff))
                        total_pixels += len(ortho_valid)
                        
                        # Simple seamline detection (gradient-based, only for first band)
                        if band_idx == 0:
                            try:
                                from scipy import ndimage
                                # Calculate gradient magnitude
                                grad_x = ndimage.sobel(ortho_band, axis=1)
                                grad_y = ndimage.sobel(ortho_band, axis=0)
                                grad_mag = np.sqrt(grad_x**2 + grad_y**2)
                                # Threshold for seamlines (adjust based on data range)
                                threshold = np.percentile(grad_mag[valid], 95) if np.any(valid) else 0
                                seamline_mask = grad_mag > threshold
                                total_seamline_pixels += np.sum(seamline_mask & valid)
                            except ImportError:
                                # Fallback: simple edge detection using numpy
                                grad_x = np.diff(ortho_band, axis=1)
                                grad_y = np.diff(ortho_band, axis=0)
                                # Pad to match original size
                                grad_x = np.pad(grad_x, ((0, 0), (0, 1)), mode='edge')
                                grad_y = np.pad(grad_y, ((0, 1), (0, 0)), mode='edge')
                                grad_mag = np.sqrt(grad_x**2 + grad_y**2)
                                threshold = np.percentile(grad_mag[valid], 95) if np.any(valid) else 0
                                seamline_mask = grad_mag > threshold
                                total_seamline_pixels += np.sum(seamline_mask & valid)
        
        # Calculate final metrics
        if total_pixels > 0:
            rmse = np.sqrt(total_rmse_sq / total_pixels)
            mae = total_mae / total_pixels
            seamline_pct = (total_seamline_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0
        else:
            rmse = np.nan
            mae = np.nan
            seamline_pct = 0.0
        
        # Feature matching on downsampled version (memory-efficient)
        logger.info("Computing 2D shift using feature matching on downsampled images...")
        errors_2d = {}
        
        # Downsample for feature matching
        downsample_factor = max(1, max(width, height) // max_downsample_for_matching)
        if downsample_factor > 1:
            logger.info(f"Downsampling by factor {downsample_factor} for feature matching...")
            # Read a representative tile from center
            center_i = height // 2
            center_j = width // 2
            tile_size_matching = min(tile_size * 2, width, height)
            start_i = max(0, center_i - tile_size_matching // 2)
            start_j = max(0, center_j - tile_size_matching // 2)
            end_i = min(height, start_i + tile_size_matching)
            end_j = min(width, start_j + tile_size_matching)
            
            window = rasterio.windows.Window(start_j, start_i, end_j - start_j, end_i - start_i)
            ortho_sample = ortho_src.read(1, window=window)
            ref_sample = ref_src.read(1, window=window)
            
            # Downsample
            ortho_sample = ortho_sample[::downsample_factor, ::downsample_factor]
            ref_sample = ref_sample[::downsample_factor, ::downsample_factor]
        else:
            # Read first band only for matching
            ortho_sample = ortho_src.read(1)
            ref_sample = ref_src.read(1)
        
        # Try feature matching methods
        methods_to_try = []
        if CV2_AVAILABLE:
            methods_to_try.extend(['sift', 'orb'])
        if SKIMAGE_AVAILABLE:
            methods_to_try.extend(['phase', 'template'])
        
        best_errors_2d = None
        best_confidence = 0.0
        
        for method in methods_to_try:
            try:
                errors = compute_feature_matching_2d_error(ortho_sample, ref_sample, method=method)
                if errors['match_confidence'] > best_confidence:
                    best_confidence = errors['match_confidence']
                    best_errors_2d = errors
                    # Scale back if downsampled
                    if downsample_factor > 1:
                        best_errors_2d['mean_offset_x'] = best_errors_2d.get('mean_offset_x', 0) * downsample_factor
                        best_errors_2d['mean_offset_y'] = best_errors_2d.get('mean_offset_y', 0) * downsample_factor
            except Exception as e:
                logger.debug(f"Feature matching method {method} failed: {e}")
        
        if best_errors_2d:
            errors_2d = best_errors_2d
    
    # Build metrics dictionary
    metrics = {
        'ortho_path': str(ortho_path),
        'basemap_path': str(basemap_path),
        'num_bands': min_bands,
        'reprojected_path': str(reprojected_path),
        'overall': {
            'rmse': float(rmse) if not np.isnan(rmse) else None,
            'mae': float(mae) if not np.isnan(mae) else None,
            'seamline_percentage': float(seamline_pct),
            'errors_2d': errors_2d if errors_2d else {}
        }
    }
    
    logger.info(f"Comparison complete. Overall RMSE: {metrics['overall'].get('rmse', 'N/A')}")
    
    return metrics


def apply_2d_shift_to_orthomosaic(
    ortho_path: Path,
    reference_path: Path,
    output_path: Path,
    shift_x: Optional[float] = None,
    shift_y: Optional[float] = None
) -> Tuple[Path, Dict]:
    """
    Apply a 2D shift to an orthomosaic to align it with a reference basemap.
    
    If shift_x and shift_y are not provided, they will be computed using feature matching.
    
    Args:
        ortho_path: Path to orthomosaic GeoTIFF
        reference_path: Path to reference basemap GeoTIFF
        output_path: Path to save shifted orthomosaic
        shift_x: Optional X offset in pixels (positive = shift right)
        shift_y: Optional Y offset in pixels (positive = shift down)
        
    Returns:
        Tuple of (output_path, shift_info_dict)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Reproject orthomosaic to match reference first
    logger.info(f"Reprojecting orthomosaic to match reference...")
    ortho_reproj, metadata = reproject_to_match(ortho_path, reference_path)
    
    # Load reference for feature matching if shift not provided
    with rasterio.open(reference_path) as ref:
        reference_array = ref.read()
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_width = ref.width
        ref_height = ref.height
    
    # Compute shift if not provided
    if shift_x is None or shift_y is None:
        logger.info("Computing 2D shift using feature matching...")
        # Use first band for feature matching
        ortho_band = ortho_reproj[0] if len(ortho_reproj.shape) == 3 else ortho_reproj
        ref_band = reference_array[0] if len(reference_array.shape) == 3 else reference_array
        
        # Resize images if they're very large (feature matching works better on smaller images)
        max_size = 2000  # Maximum dimension for feature matching
        scale_factor = 1.0
        if ortho_band.shape[0] > max_size or ortho_band.shape[1] > max_size:
            scale_factor = min(max_size / ortho_band.shape[0], max_size / ortho_band.shape[1])
            new_h = int(ortho_band.shape[0] * scale_factor)
            new_w = int(ortho_band.shape[1] * scale_factor)
            logger.info(f"Resizing images for feature matching: {ortho_band.shape} -> ({new_h}, {new_w})")
            
            try:
                from scipy.ndimage import zoom
                ortho_band = zoom(ortho_band, scale_factor, order=1)
                ref_band = zoom(ref_band, scale_factor, order=1)
            except ImportError:
                # Fallback to simple downsampling
                step = int(1 / scale_factor)
                ortho_band = ortho_band[::step, ::step]
                ref_band = ref_band[::step, ::step]
        
        # Try multiple feature matching methods, use the best one
        # Add line/edge-based matching for different imagery types
        methods_to_try = []
        if CV2_AVAILABLE:
            methods_to_try.extend(['sift', 'orb', 'lines'])
        if SKIMAGE_AVAILABLE:
            methods_to_try.extend(['phase', 'template'])
        
        best_errors_2d = None
        best_confidence = 0.0
        best_method = None
        
        # Get pixel resolution for meter conversion and logging
        pixel_resolution = abs(ref_transform[0])  # Pixel size in meters
        
        # Setup log file path
        log_file_path = None
        if output_path:
            log_dir = output_path.parent / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file_path = log_dir / f"2d_shift_{Path(ortho_path).stem}.log"
        
        for method in methods_to_try:
            try:
                logger.debug(f"Trying feature matching method: {method}")
                errors_2d = compute_feature_matching_2d_error(
                    ortho_band, ref_band, 
                    method=method,
                    pixel_resolution=pixel_resolution,
                    log_file_path=log_file_path
                )
                
                if errors_2d.get('mean_offset_x') is not None and errors_2d.get('mean_offset_y') is not None:
                    confidence = errors_2d.get('match_confidence', 0.0)
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_errors_2d = errors_2d
                        best_method = method
                        logger.info(f"Method {method} found shift: ({errors_2d['mean_offset_x']:.2f}, {errors_2d['mean_offset_y']:.2f}) px, confidence={confidence:.3f}")
                        if errors_2d.get('mean_offset_x_meters'):
                            logger.info(f"  In meters: ({errors_2d['mean_offset_x_meters']:.4f}, {errors_2d['mean_offset_y_meters']:.4f}) m")
            except Exception as e:
                logger.debug(f"Feature matching method {method} failed: {e}")
                continue
        
        if best_errors_2d and best_errors_2d.get('mean_offset_x') is not None and best_errors_2d.get('mean_offset_y') is not None:
            # Scale shift back if we resized
            shift_x = best_errors_2d['mean_offset_x'] / scale_factor
            shift_y = best_errors_2d['mean_offset_y'] / scale_factor
            shift_x_meters = (best_errors_2d.get('mean_offset_x_meters', 0) or 0) / scale_factor
            shift_y_meters = (best_errors_2d.get('mean_offset_y_meters', 0) or 0) / scale_factor
            logger.info(f"Computed shift using {best_method}: X={shift_x:.2f} px, Y={shift_y:.2f} px (confidence={best_confidence:.3f})")
            if shift_x_meters != 0 or shift_y_meters != 0:
                logger.info(f"  In meters: X={shift_x_meters:.4f} m, Y={shift_y_meters:.4f} m")
            
            # Create visualization if matches found
            if best_errors_2d.get('match_pairs') and len(best_errors_2d['match_pairs']) > 0:
                try:
                    from .visualization import visualize_feature_matches
                    vis_dir = output_path.parent / "visualizations"
                    vis_dir.mkdir(parents=True, exist_ok=True)
                    vis_path = vis_dir / f"2d_shift_matches_{Path(ortho_path).stem}.png"
                    visualize_feature_matches(
                        ortho_band, ref_band,
                        best_errors_2d['match_pairs'],
                        vis_path,
                        title=f"2D Shift Feature Matches: {Path(ortho_path).stem}"
                    )
                    logger.info(f"Feature match visualization saved to: {vis_path}")
                except Exception as e:
                    logger.warning(f"Could not create visualization: {e}")
        else:
            logger.warning("Could not compute shift from any feature matching method. Using zero shift.")
            logger.warning(f"Tried methods: {methods_to_try}")
            logger.warning(f"Image shapes: ortho={ortho_band.shape}, ref={ref_band.shape}")
            shift_x = 0.0
            shift_y = 0.0
            shift_x_meters = 0.0
            shift_y_meters = 0.0
    else:
        logger.info(f"Using provided shift: X={shift_x:.2f} px, Y={shift_y:.2f} px")
    
    # Apply shift using scipy.ndimage.shift for sub-pixel accuracy, or numpy for integer shifts
    try:
        from scipy.ndimage import shift as ndimage_shift
        use_scipy = True
    except ImportError:
        use_scipy = False
        logger.warning("scipy not available. Using integer pixel shifts only.")
    
    shift_x_int = int(round(shift_x))
    shift_y_int = int(round(shift_y))
    
    # Apply shift to each band
    shifted_ortho = np.zeros_like(ortho_reproj)
    
    for band_idx in range(ortho_reproj.shape[0]):
        band = ortho_reproj[band_idx]
        
        if use_scipy and (abs(shift_x - shift_x_int) > 0.01 or abs(shift_y - shift_y_int) > 0.01):
            # Use scipy for sub-pixel shifts
            shifted_band = ndimage_shift(band, (shift_y, shift_x), order=1, mode='constant', cval=0.0)
        else:
            # Use numpy for integer shifts
            if shift_x_int != 0 or shift_y_int != 0:
                # Create a new array with zeros
                shifted_band = np.zeros_like(band)
                
                # Calculate valid source and destination regions
                if shift_x_int > 0:
                    src_x = slice(0, band.shape[1] - shift_x_int)
                    dst_x = slice(shift_x_int, band.shape[1])
                elif shift_x_int < 0:
                    src_x = slice(-shift_x_int, band.shape[1])
                    dst_x = slice(0, band.shape[1] + shift_x_int)
                else:
                    src_x = slice(None)
                    dst_x = slice(None)
                
                if shift_y_int > 0:
                    src_y = slice(0, band.shape[0] - shift_y_int)
                    dst_y = slice(shift_y_int, band.shape[0])
                elif shift_y_int < 0:
                    src_y = slice(-shift_y_int, band.shape[0])
                    dst_y = slice(0, band.shape[0] + shift_y_int)
                else:
                    src_y = slice(None)
                    dst_y = slice(None)
                
                # Copy valid region
                if src_x != slice(None) or src_y != slice(None):
                    shifted_band[dst_y, dst_x] = band[src_y, src_x]
                else:
                    shifted_band = band.copy()
            else:
                shifted_band = band.copy()
        
        shifted_ortho[band_idx] = shifted_band
    
    # Update transform to account for shift
    # Shift in pixels needs to be converted to geographic coordinates
    pixel_size_x = abs(ref_transform[0])  # Pixel width
    pixel_size_y = abs(ref_transform[4])  # Pixel height (usually negative)
    
    # Adjust transform origin
    new_transform = Affine(
        ref_transform[0], ref_transform[1],
        ref_transform[2] - shift_x * pixel_size_x,  # Adjust X origin
        ref_transform[3], ref_transform[4],
        ref_transform[5] - shift_y * abs(pixel_size_y)  # Adjust Y origin (note: Y is usually negative)
    )
    
    # Save shifted orthomosaic with JPEG compression (quality 90)
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=ref_height,
        width=ref_width,
        count=ortho_reproj.shape[0],
        dtype=ortho_reproj.dtype,
        crs=ref_crs,
        transform=new_transform,
        compress='jpeg',
        jpeg_quality=90,
        tiled=True,
        blockxsize=512,
        blockysize=512
    ) as dst:
        dst.write(shifted_ortho)
    
    shift_info = {
        'shift_x_pixels': float(shift_x),
        'shift_y_pixels': float(shift_y),
        'shift_x_meters': float(shift_x * pixel_size_x),
        'shift_y_meters': float(shift_y * abs(pixel_size_y)),
        'shift_x_geographic': float(shift_x * pixel_size_x),  # Keep for backward compatibility
        'shift_y_geographic': float(shift_y * abs(pixel_size_y)),  # Keep for backward compatibility
        'output_path': str(output_path)
    }
    
    logger.info(f"Applied 2D shift and saved shifted orthomosaic to: {output_path}")
    logger.info(f"Shift: ({shift_x:.2f}, {shift_y:.2f}) pixels = "
                f"({shift_x * pixel_size_x:.4f}, {shift_y * abs(pixel_size_y):.4f}) meters")
    
    return output_path, shift_info


def align_orthomosaic_to_gcps(
    ortho_path: Path,
    reference_path: Path,
    gcps: List[Dict],
    output_path: Path
) -> Tuple[Path, Dict]:
    """
    Align an orthomosaic to ground control points by computing a transformation
    from GCP positions in the orthomosaic to their known coordinates.
    
    Args:
        ortho_path: Path to orthomosaic GeoTIFF
        reference_path: Path to reference basemap GeoTIFF (for CRS and bounds)
        gcps: List of GCP dictionaries with 'lat', 'lon', and optionally 'z' keys
        output_path: Path to save aligned orthomosaic
        
    Returns:
        Tuple of (output_path, alignment_info_dict)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Aligning orthomosaic to {len(gcps)} GCPs...")
    
    # Reproject orthomosaic to match reference first
    logger.info("Reprojecting orthomosaic to match reference...")
    ortho_reproj, metadata = reproject_to_match(ortho_path, reference_path)
    
    # Load reference for CRS and transform
    with rasterio.open(reference_path) as ref:
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_width = ref.width
        ref_height = ref.height
        reference_array = ref.read()
    
    # Convert GCP lat/lon to pixel coordinates
    # For the reference: GCPs should be at their known coordinates
    # For the orthomosaic: We need to find where GCPs actually appear (they may be misaligned)
    gcp_pairs = []
    
    # Get first band for feature detection
    ortho_band = ortho_reproj[0] if len(ortho_reproj.shape) == 3 else ortho_reproj
    ref_band = reference_array[0] if len(reference_array.shape) == 3 else reference_array
    
    # Normalize images for feature matching
    def normalize_to_uint8(arr):
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            normalized = ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(arr, dtype=np.uint8)
        return normalized
    
    ortho_norm = normalize_to_uint8(ortho_band)
    ref_norm = normalize_to_uint8(ref_band)
    
    for gcp in gcps:
        lat = gcp.get('lat')
        lon = gcp.get('lon')
        
        if lat is None or lon is None:
            logger.warning(f"Skipping GCP {gcp.get('id', 'unknown')}: missing lat/lon")
            continue
        
        # Convert GCP coordinates to pixel coordinates in reference (where they should be)
        try:
            ref_row, ref_col = rasterio.transform.rowcol(ref_transform, lon, lat)
            
            # Check if reference coordinates are within bounds
            if not (0 <= ref_row < ref_height and 0 <= ref_col < ref_width):
                logger.debug(f"GCP {gcp.get('id', 'unknown')} outside reference bounds, skipping")
                continue
            
            # Project GCP directly to pixel coordinates in orthomosaic using its geotransform
            # This is more accurate than template matching since we know the GCP coordinates
            ortho_transform = metadata['transform']
            ortho_row, ortho_col = rasterio.transform.rowcol(ortho_transform, lon, lat)
            
            logger.debug(f"GCP {gcp.get('id', 'unknown')}: reference pixel ({ref_col:.1f}, {ref_row:.1f}), "
                        f"orthomosaic pixel ({ortho_col:.1f}, {ortho_row:.1f})")
            
            # Check if orthomosaic coordinates are within bounds
            if (0 <= ortho_row < ortho_reproj.shape[1] and 0 <= ortho_col < ortho_reproj.shape[2]):
                gcp_pairs.append({
                    'id': gcp.get('id', 'unknown'),
                    'ortho_pixel': (ortho_col, ortho_row),  # (x, y) in pixels - where GCP actually appears
                    'ref_pixel': (ref_col, ref_row),  # (x, y) in pixels - where GCP should be
                    'lat': lat,
                    'lon': lon
                })
            else:
                logger.debug(f"GCP {gcp.get('id', 'unknown')} outside orthomosaic bounds, skipping")
        except Exception as e:
            logger.warning(f"Could not process GCP {gcp.get('id', 'unknown')}: {e}")
            continue
    
    if len(gcp_pairs) < 3:
        logger.error(f"Need at least 3 GCPs for alignment, but only {len(gcp_pairs)} are valid")
        raise ValueError(f"Insufficient GCPs for alignment: {len(gcp_pairs)} < 3")
    
    logger.info(f"Using {len(gcp_pairs)} GCPs for alignment")
    
    # Compute affine transformation from ortho pixels to ref pixels
    # We'll use a least-squares approach to solve for the transformation matrix
    # Affine transformation: [x'] = [a b c] [x]
    #                        [y']   [d e f] [y]
    #                                        [1]
    
    # Build matrices for least squares
    ortho_coords = np.array([[p['ortho_pixel'][0], p['ortho_pixel'][1], 1] for p in gcp_pairs])
    ref_coords_x = np.array([p['ref_pixel'][0] for p in gcp_pairs])
    ref_coords_y = np.array([p['ref_pixel'][1] for p in gcp_pairs])
    
    # Solve for transformation parameters
    try:
        # Solve for X transformation: ref_x = a*ortho_x + b*ortho_y + c
        params_x, residuals_x, rank_x, s_x = np.linalg.lstsq(ortho_coords, ref_coords_x, rcond=None)
        
        # Solve for Y transformation: ref_y = d*ortho_x + e*ortho_y + f
        params_y, residuals_y, rank_y, s_y = np.linalg.lstsq(ortho_coords, ref_coords_y, rcond=None)
        
        # Extract transformation parameters
        a, b, c = params_x
        d, e, f = params_y
        
        # Compute transformation error
        predicted_x = ortho_coords @ params_x
        predicted_y = ortho_coords @ params_y
        errors_x = ref_coords_x - predicted_x
        errors_y = ref_coords_y - predicted_y
        rmse_x = np.sqrt(np.mean(errors_x**2))
        rmse_y = np.sqrt(np.mean(errors_y**2))
        rmse_total = np.sqrt(rmse_x**2 + rmse_y**2)
        
        # Get pixel resolution for meter conversion
        pixel_resolution = abs(ref_transform[0])  # Pixel size in meters
        
        logger.info(f"Affine transformation computed:")
        logger.info(f"  X: {a:.6f}*x + {b:.6f}*y + {c:.6f}")
        logger.info(f"  Y: {d:.6f}*x + {e:.6f}*y + {f:.6f}")
        logger.info(f"  RMSE: X={rmse_x:.2f} px ({rmse_x * pixel_resolution:.4f} m), "
                   f"Y={rmse_y:.2f} px ({rmse_y * pixel_resolution:.4f} m), "
                   f"Total={rmse_total:.2f} px ({rmse_total * pixel_resolution:.4f} m)")
        
    except np.linalg.LinAlgError as e:
        logger.error(f"Failed to compute affine transformation: {e}")
        raise ValueError(f"Could not compute transformation from GCPs: {e}")
    
    # Apply transformation to orthomosaic
    # We computed T: ortho -> ref, but for resampling we need T^-1: ref -> ortho
    # For affine transformation [x'] = [a b c] [x], the inverse is:
    #                        [y']   [d e f] [y]
    #                                        [1]
    # [x] = 1/(ae-bd) * [e -b] [x' - c]
    # [y]              [-d  a] [y' - f]
    
    det = a * e - b * d
    if abs(det) < 1e-10:
        logger.warning("Transformation matrix is singular, using identity transformation")
        a_inv, b_inv, c_inv = 1.0, 0.0, 0.0
        d_inv, e_inv, f_inv = 0.0, 1.0, 0.0
    else:
        # Compute inverse transformation
        a_inv = e / det
        b_inv = -b / det
        c_inv = (b * f - c * e) / det
        d_inv = -d / det
        e_inv = a / det
        f_inv = (c * d - a * f) / det
    
    # Create output array matching reference dimensions
    aligned_ortho = np.zeros((ortho_reproj.shape[0], ref_height, ref_width), dtype=ortho_reproj.dtype)
    
    # Create coordinate grids for output (reference) space
    ref_x, ref_y = np.meshgrid(np.arange(ref_width), np.arange(ref_height))
    
    # Apply inverse transformation to find source coordinates in orthomosaic
    ortho_h, ortho_w = ortho_reproj.shape[1], ortho_reproj.shape[2]
    ortho_x = a_inv * ref_x + b_inv * ref_y + c_inv
    ortho_y = d_inv * ref_x + e_inv * ref_y + f_inv
    
    # Use scipy for interpolation if available
    try:
        from scipy.ndimage import map_coordinates
        use_scipy = True
    except ImportError:
        use_scipy = False
        logger.warning("scipy not available. Using nearest neighbor interpolation.")
    
    for band_idx in range(ortho_reproj.shape[0]):
        band = ortho_reproj[band_idx]
        
        if use_scipy:
            # Use map_coordinates for sub-pixel interpolation
            # Stack coordinates: (2, height, width) where first is y (row), second is x (col)
            coords = np.stack([ortho_y.ravel(), ortho_x.ravel()], axis=0)
            
            # Reshape to (2, height, width)
            coords = coords.reshape(2, ref_height, ref_width)
            
            # Map coordinates (note: map_coordinates uses (row, col) = (y, x))
            aligned_band = map_coordinates(
                band,
                coords,
                order=1,  # Bilinear interpolation
                mode='constant',
                cval=0.0,
                prefilter=False
            )
        else:
            # Nearest neighbor fallback
            ortho_x_int = np.clip(np.round(ortho_x).astype(int), 0, ortho_w - 1)
            ortho_y_int = np.clip(np.round(ortho_y).astype(int), 0, ortho_h - 1)
            
            # Map using advanced indexing
            aligned_band = np.zeros((ref_height, ref_width), dtype=band.dtype)
            valid_mask = (ortho_x >= 0) & (ortho_x < ortho_w) & (ortho_y >= 0) & (ortho_y < ortho_h)
            aligned_band[valid_mask] = band[ortho_y_int[valid_mask], ortho_x_int[valid_mask]]
        
        aligned_ortho[band_idx] = aligned_band
    
    # Get pixel resolution for meter conversion
    pixel_resolution = abs(ref_transform[0])  # Pixel size in meters
    
    # Setup log file path
    log_file_path = None
    if output_path:
        log_dir = output_path.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir / f"gcp_alignment_{Path(ortho_path).stem}.log"
    
    # Write detailed alignment results to log file
    if log_file_path:
        with open(log_file_path, 'w', encoding='utf-8') as log_file:
            log_file.write("GCP Alignment Results\n")
            log_file.write("=" * 60 + "\n\n")
            log_file.write(f"Number of GCPs used: {len(gcp_pairs)}\n")
            log_file.write(f"Pixel resolution: {pixel_resolution:.4f} m/pixel\n\n")
            log_file.write(f"Affine Transformation:\n")
            log_file.write(f"  X: {a:.6f}*x + {b:.6f}*y + {c:.6f}\n")
            log_file.write(f"  Y: {d:.6f}*x + {e:.6f}*y + {f:.6f}\n\n")
            log_file.write(f"RMSE (pixels): X={rmse_x:.4f}, Y={rmse_y:.4f}, Total={rmse_total:.4f}\n")
            log_file.write(f"RMSE (meters): X={rmse_x * pixel_resolution:.4f}, Y={rmse_y * pixel_resolution:.4f}, Total={rmse_total * pixel_resolution:.4f}\n\n")
            log_file.write(f"GCP Pairs:\n")
            log_file.write(f"{'GCP ID':<15} {'Ortho Pixel (x,y)':<25} {'Ref Pixel (x,y)':<25} {'Error (px)':<20} {'Error (m)':<15}\n")
            log_file.write("-" * 100 + "\n")
            for pair in gcp_pairs:
                ortho_px, ortho_py = pair['ortho_pixel']
                ref_px, ref_py = pair['ref_pixel']
                # Compute predicted position
                pred_x = a * ortho_px + b * ortho_py + c
                pred_y = d * ortho_px + e * ortho_py + f
                error_x = ref_px - pred_x
                error_y = ref_py - pred_y
                error_px = np.sqrt(error_x**2 + error_y**2)
                error_m = error_px * pixel_resolution
                log_file.write(f"{pair['id']:<15} ({ortho_px:>8.2f}, {ortho_py:>8.2f})  ({ref_px:>8.2f}, {ref_py:>8.2f})  "
                             f"{error_px:>10.2f}  {error_m:>10.4f}\n")
        logger.info(f"GCP alignment log saved to: {log_file_path}")
    
    # Save aligned orthomosaic with JPEG compression (quality 90)
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=ref_height,
        width=ref_width,
        count=ortho_reproj.shape[0],
        dtype=ortho_reproj.dtype,
        crs=ref_crs,
        transform=ref_transform,
        compress='jpeg',
        jpeg_quality=90,
        tiled=True,
        blockxsize=512,
        blockysize=512
    ) as dst:
        dst.write(aligned_ortho)
    
    # Create visualization of GCP alignment
    try:
        from .visualization import visualize_feature_matches
        vis_dir = output_path.parent / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
        vis_path = vis_dir / f"gcp_alignment_{Path(ortho_path).stem}.png"
        
        # Convert GCP pairs to match_pairs format for visualization
        match_pairs = [(pair['ortho_pixel'], pair['ref_pixel']) for pair in gcp_pairs]
        visualize_feature_matches(
            ortho_band, ref_band,
            match_pairs,
            vis_path,
            title=f"GCP Alignment: {Path(ortho_path).stem} ({len(gcp_pairs)} GCPs)"
        )
        logger.info(f"GCP alignment visualization saved to: {vis_path}")
    except Exception as e:
        logger.warning(f"Could not create GCP alignment visualization: {e}")
    
    alignment_info = {
        'num_gcps_used': len(gcp_pairs),
        'transformation_params': {
            'a': float(a), 'b': float(b), 'c': float(c),
            'd': float(d), 'e': float(e), 'f': float(f)
        },
        'rmse_x_pixels': float(rmse_x),
        'rmse_y_pixels': float(rmse_y),
        'rmse_total_pixels': float(rmse_total),
        'rmse_x_meters': float(rmse_x * pixel_resolution),
        'rmse_y_meters': float(rmse_y * pixel_resolution),
        'rmse_total_meters': float(rmse_total * pixel_resolution),
        'output_path': str(output_path)
    }
    
    logger.info(f"Alignment RMSE: X={rmse_x:.2f} px ({rmse_x * pixel_resolution:.4f} m), "
                f"Y={rmse_y:.2f} px ({rmse_y * pixel_resolution:.4f} m), "
                f"Total={rmse_total:.2f} px ({rmse_total * pixel_resolution:.4f} m)")
    
    logger.info(f"Aligned orthomosaic saved to: {output_path}")
    logger.info(f"Alignment RMSE: {rmse_total:.2f} pixels")
    
    return output_path, alignment_info

