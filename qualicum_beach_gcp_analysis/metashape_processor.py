"""
MetaShape Processing Module for Orthomosaic Generation.

Handles processing drone imagery with MetaShape to create orthomosaics,
with support for both images-only and images+GCPs workflows.
"""

import os
import logging
import shutil
from pathlib import Path
from typing import List, Optional, Dict
from enum import IntEnum
from PIL import Image

# Import Metashape
try:
    import Metashape
    METASHAPE_AVAILABLE = True
except ImportError:
    METASHAPE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Metashape not available. Install Agisoft Metashape Python API.")

logger = logging.getLogger(__name__)

# Unset max image size to avoid decompression bomb errors
Image.MAX_IMAGE_PIXELS = None


class PhotoMatchQuality(IntEnum):
    """Photo matching quality settings"""
    UltraQuality = 0
    HighQuality = 1
    MediumQuality = 2
    LowQuality = 4
    LowestQuality = 8


class DepthMapQuality(IntEnum):
    """Depth map quality settings"""
    UltraQuality = 1
    HighQuality = 2
    MediumQuality = 4
    LowQuality = 8
    LowestQuality = 16


def find_image_files(folder: Path, types: List[str] = [".jpg", ".jpeg"]) -> List[str]:
    """Find image files in a directory.
    
    Args:
        folder: Directory to search
        types: List of file extensions to include
        
    Returns:
        List of image file paths
    """
    files = []
    for entry in os.scandir(folder):
        if entry.is_file() and os.path.splitext(entry.name)[1].lower() in types:
            files.append(entry.path)
        elif entry.is_dir():
            files.extend(find_image_files(Path(entry.path), types))
    return files


def cleanup_previous_run(project_path: Path) -> bool:
    """Clean up files from previous processing run.
    
    Args:
        project_path: Path to the Metashape project file
        
    Returns:
        True if cleanup was successful, False otherwise
    """
    try:
        # Remove the project file
        if project_path.exists():
            project_path.unlink()
            logger.info(f"Removed project file: {project_path}")
        
        # Remove the project files directory
        files_directory = project_path.parent / f"{project_path.stem}.files"
        if files_directory.exists():
            shutil.rmtree(files_directory)
            logger.info(f"Removed project files directory: {files_directory}")
        
        return True
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        return False


def process_orthomosaic(
    photos_dir: Path,
    output_path: Path,
    project_path: Path,
    gcps: Optional[List[Dict]] = None,
    gcp_file: Optional[Path] = None,
    product_id: str = "orthomosaic",
    clean_intermediate_files: bool = True,
    photo_match_quality: int = PhotoMatchQuality.MediumQuality,
    depth_map_quality: int = DepthMapQuality.MediumQuality,
    tiepoint_limit: int = 10000,
    use_gcps: bool = False
) -> Dict:
    """
    Process orthomosaic using MetaShape.
    
    Args:
        photos_dir: Directory containing input images
        output_path: Directory for output files
        project_path: Path to save MetaShape project file
        gcps: List of GCP dictionaries (optional, if gcp_file not provided)
        gcp_file: Path to GCP file (CSV or XML) (optional, if gcps not provided)
        product_id: Identifier for this product
        clean_intermediate_files: Whether to clean up previous runs
        photo_match_quality: PhotoMatchQuality enum value
        depth_map_quality: DepthMapQuality enum value
        tiepoint_limit: Maximum number of tie points
        use_gcps: Whether to use GCPs in processing
        
    Returns:
        Dictionary with processing results and statistics
    """
    if not METASHAPE_AVAILABLE:
        raise ImportError("Metashape is not available. Please install Agisoft Metashape Python API.")
    
    # Configure GPU
    Metashape.app.gpu_mask = ~0
    
    # Prevent license errors in multi-instance environments
    os.environ["AGISOFT_CHECKOUT_RETRIES"] = "10"
    
    # Setup paths
    output_path.mkdir(parents=True, exist_ok=True)
    project_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Cleanup if requested
    if clean_intermediate_files:
        logger.info("🧹 Cleaning up previous processing files...")
        cleanup_previous_run(project_path)
    
    # Compression settings (from main.py)
    compression = Metashape.ImageCompression()
    compression.tiff_compression = Metashape.ImageCompression.TiffCompressionNone
    compression.tiff_big = True
    compression.tiff_overviews = True
    compression.tiff_tiled = True
    
    # Initialize project
    logger.info("🚀 Initializing Metashape project...")
    doc = Metashape.Document()
    doc.save(str(project_path))
    chunk = doc.addChunk()
    
    # Add photos
    logger.info(f"Adding photos from: {photos_dir}")
    photos = find_image_files(photos_dir)
    if not photos:
        raise ValueError(f"No images found in {photos_dir}")
    
    logger.info(f"Found {len(photos)} images")
    chunk.addPhotos(photos)
    doc.save()
    
    # Add GCPs if requested
    if use_gcps:
        if gcp_file and gcp_file.exists():
            logger.info(f"Loading GCPs from file: {gcp_file}")
            chunk.importMarkers(str(gcp_file))
            doc.save()
        elif gcps:
            logger.info(f"Using {len(gcps)} GCPs from provided list")
            # Add markers manually
            for gcp in gcps:
                marker = chunk.addMarker()
                marker.label = gcp.get('id', f"GCP_{gcps.index(gcp)+1}")
                marker.reference.location = Metashape.Vector((
                    gcp.get('lon', 0.0),
                    gcp.get('lat', 0.0),
                    gcp.get('z', 0.0)
                ))
                marker.reference.enabled = True
                marker.reference.accuracy = Metashape.Vector((
                    gcp.get('accuracy', 1.0),
                    gcp.get('accuracy', 1.0),
                    gcp.get('accuracy', 1.0)
                ))
            doc.save()
        else:
            logger.warning("use_gcps=True but no GCPs provided. Processing without GCPs.")
    
    # Match photos
    logger.info("Matching photos...")
    chunk.matchPhotos(
        downscale=photo_match_quality,
        tiepoint_limit=tiepoint_limit,
    )
    doc.save()
    
    # Align cameras
    logger.info("Aligning cameras...")
    chunk.alignCameras()
    doc.save()
    
    # Build depth maps
    logger.info("Building depth maps...")
    chunk.buildDepthMaps(
        downscale=depth_map_quality,
        filter_mode=Metashape.MildFiltering
    )
    doc.save()
    
    # Build 3D model
    logger.info("Building 3D model...")
    chunk.buildModel()
    doc.save()
    
    # Build orthomosaic
    logger.info("Building orthomosaic...")
    chunk.buildOrthomosaic()
    doc.save()
    
    # Export GeoTIFF
    ortho_path = output_path / f"{product_id}.tif"
    logger.info(f"Exporting GeoTIFF to: {ortho_path}")
    chunk.exportRaster(
        str(ortho_path),
        image_compression=compression,
        description=f"Orthomosaic generated by Qualicum Beach GCP Analysis ({'with GCPs' if use_gcps else 'without GCPs'})",
    )
    doc.save()
    
    # Get statistics
    stats = {
        'product_id': product_id,
        'use_gcps': use_gcps,
        'num_photos': len(photos),
        'num_markers': len(chunk.markers) if use_gcps else 0,
        'ortho_path': str(ortho_path),
        'project_path': str(project_path)
    }
    
    # Get camera alignment statistics if available
    if chunk.cameras:
        aligned = sum(1 for cam in chunk.cameras if cam.transform)
        stats['aligned_cameras'] = aligned
        stats['total_cameras'] = len(chunk.cameras)
    
    # Get marker statistics if GCPs were used
    if use_gcps and chunk.markers:
        enabled_markers = sum(1 for m in chunk.markers if m.reference.enabled)
        stats['enabled_markers'] = enabled_markers
        stats['total_markers'] = len(chunk.markers)
    
    logger.info("✅ MetaShape processing completed successfully")
    
    return stats

