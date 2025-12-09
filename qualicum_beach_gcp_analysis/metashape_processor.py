"""
MetaShape Processing Module for Orthomosaic Generation.

Handles processing drone imagery with MetaShape to create orthomosaics,
with support for both images-only and images+GCPs workflows.
"""

import os
import sys
import logging
import shutil
from pathlib import Path
from typing import List, Optional, Dict
from enum import IntEnum
from contextlib import contextmanager
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


class TeeOutput:
    """
    A file-like object that writes to both a file and optionally shows progress messages.
    Filters MetaShape verbose output and only shows key progress messages.
    """
    def __init__(self, log_file, original_stdout=None, show_progress=True):
        self.log_file = log_file
        self.original_stdout = original_stdout
        self.show_progress = show_progress
        self.buffer = ""
        
        # Keywords that indicate important progress messages
        self.progress_keywords = [
            "AddPhotos",
            "MatchPhotos",
            "AlignCameras",
            "BuildDepthMaps",
            "BuildModel",
            "BuildOrthomosaic",
            "ExportRaster",
            "LoadProject",
            "SaveProject",
            "ImportMarkers",
            "OptimizeCameras",
            "Filtering done",
            "depth maps filtered",
            "saved depth maps",
            "loaded depth map",
            "Processing",
            "Progress:",
            "completed",
            "finished",
            "error",
            "Error",
            "warning",
            "Warning"
        ]
    
    def write(self, text):
        """Write to both log file and optionally show progress in notebook."""
        # Always write to log file FIRST (before any filtering)
        try:
            self.log_file.write(text)
            self.log_file.flush()  # Ensure it's written immediately
        except Exception as e:
            # If log file write fails, try to write to original stdout as fallback
            if self.original_stdout:
                print(f"[LOG ERROR: {e}]", file=self.original_stdout, flush=True)
        
        # Buffer text to check for complete lines
        self.buffer += text
        
        # Process complete lines
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            line = line.strip()
            
            if not line:
                continue
            
            # Check if this line contains progress information
            if self.show_progress and self.original_stdout:
                # Show only lines with progress keywords or important status
                if any(keyword.lower() in line.lower() for keyword in self.progress_keywords):
                    # Clean up the line for display
                    display_line = line
                    # Remove timestamps if present
                    if ':' in display_line and len(display_line) > 20:
                        parts = display_line.split(':', 2)
                        if len(parts) >= 3 and parts[0].replace(' ', '').replace('-', '').isdigit():
                            display_line = ':'.join(parts[2:]).strip()
                    
                    # Show progress message
                    print(display_line, file=self.original_stdout, flush=True)
    
    def flush(self):
        """Flush both outputs."""
        self.log_file.flush()
        if self.original_stdout:
            self.original_stdout.flush()
    
    def close(self):
        """Close the log file (but not original stdout)."""
        if self.log_file:
            self.log_file.close()


@contextmanager
def redirect_metashape_output(log_file_path: Path, show_progress: bool = True):
    """
    Context manager to redirect MetaShape's stdout/stderr to a log file
    while keeping logger output and key progress messages visible in the notebook.
    
    Args:
        log_file_path: Path to the log file
        show_progress: Whether to show progress messages in notebook (default: True)
    """
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Open log file in write mode (not append) to start fresh each time
    log_file = open(log_file_path, 'w', encoding='utf-8')
    
    # Save original stdout and stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Get the root logger and configure it to use original stderr
    # This ensures logger output still shows in notebook even after redirecting stderr
    root_logger = logging.getLogger()
    
    # Remove any existing StreamHandlers that write to stderr (they'll write to file after redirect)
    handlers_to_remove = []
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream in (sys.stderr, sys.stdout):
            handlers_to_remove.append(handler)
    
    for handler in handlers_to_remove:
        root_logger.removeHandler(handler)
    
    # Create a handler that writes to the original stderr (saved before redirect)
    console_handler = logging.StreamHandler(original_stderr)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    try:
        # Create TeeOutput that writes to file and shows progress
        tee_stdout = TeeOutput(log_file, original_stdout, show_progress=show_progress)
        tee_stderr = TeeOutput(log_file, original_stdout, show_progress=show_progress)
        
        # Redirect stdout and stderr to TeeOutput (MetaShape output goes here)
        sys.stdout = tee_stdout
        sys.stderr = tee_stderr
        
        yield log_file
        
    finally:
        # Flush before restoring
        if hasattr(sys.stdout, 'flush'):
            sys.stdout.flush()
        if hasattr(sys.stderr, 'flush'):
            sys.stderr.flush()
        
        # Restore original stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        
        # Close TeeOutput (which closes log file)
        if hasattr(tee_stdout, 'close'):
            tee_stdout.close()
        if hasattr(tee_stderr, 'close'):
            tee_stderr.close()
        
        # Remove the console handler we added
        root_logger.removeHandler(console_handler)
        
        # Restore removed handlers if any
        for handler in handlers_to_remove:
            root_logger.addHandler(handler)


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


def check_processing_status(chunk) -> Dict[str, bool]:
    """
    Check which processing steps have been completed in a MetaShape chunk.
    
    Args:
        chunk: MetaShape chunk object
        
    Returns:
        Dictionary indicating which steps are complete
    """
    # Check tie points - MetaShape.TiePoints object doesn't support len()
    # Check if tie_points exists and has points by accessing the points property
    photos_matched = False
    if chunk.tie_points is not None:
        try:
            # Try to access tie points - if it has points, this will work
            points = chunk.tie_points.points
            photos_matched = points is not None and len(points) > 0
        except (AttributeError, TypeError):
            # If tie_points exists but has no points property or is empty
            photos_matched = False
    
    # Check depth maps - MetaShape.DepthMaps is a collection object
    depth_maps_built = False
    if chunk.depth_maps is not None:
        try:
            # Try to access depth maps as a collection
            # DepthMaps might be iterable or have a count
            if hasattr(chunk.depth_maps, '__len__'):
                depth_maps_built = len(chunk.depth_maps) > 0
            elif hasattr(chunk.depth_maps, '__iter__'):
                # Try to iterate and count
                try:
                    count = sum(1 for _ in chunk.depth_maps)
                    depth_maps_built = count > 0
                except:
                    depth_maps_built = chunk.depth_maps is not None
            else:
                depth_maps_built = chunk.depth_maps is not None
        except (TypeError, AttributeError):
            # If depth_maps exists but doesn't support len()
            depth_maps_built = chunk.depth_maps is not None
    
    status = {
        'photos_added': len(chunk.cameras) > 0,
        'photos_matched': photos_matched,
        'cameras_aligned': len(chunk.cameras) > 0 and any(cam.transform for cam in chunk.cameras),
        'depth_maps_built': depth_maps_built,
        'model_built': chunk.model is not None,
        'orthomosaic_built': chunk.orthomosaic is not None
    }
    return status


def process_orthomosaic(
    photos_dir: Path,
    output_path: Path,
    project_path: Path,
    gcps: Optional[List[Dict]] = None,
    gcp_file: Optional[Path] = None,
    product_id: str = "orthomosaic",
    clean_intermediate_files: bool = False,
    photo_match_quality: int = PhotoMatchQuality.MediumQuality,
    depth_map_quality: int = DepthMapQuality.MediumQuality,
    tiepoint_limit: int = 10000,
    use_gcps: bool = False,
    gcp_accuracy: float = 0.05
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
        gcp_accuracy: Accuracy of GCPs in meters. Lower values = higher weight in bundle adjustment.
                     Default 0.05m (5cm) gives high weight. Use 0.01m (1cm) for very high accuracy GCPs,
                     or 0.10m (10cm) for lower accuracy GCPs.
        
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
    
    # Setup log file for MetaShape verbose output
    log_dir = project_path.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / f"{product_id}_metashape.log"
    
    logger.info(f"📝 MetaShape verbose output will be saved to: {log_file_path}")
    
    # Compression settings - use JPEG compression with quality factor 90
    compression = Metashape.ImageCompression()
    # Use JPEG compression with quality 90 for good balance of file size and quality
    compression.tiff_compression = Metashape.ImageCompression.TiffCompressionJPEG
    compression.jpeg_quality = 90
    compression.tiff_big = True
    compression.tiff_overviews = True
    compression.tiff_tiled = True
    
    # Check if project exists and load it, or create new one
    project_exists = project_path.exists()
    
    # Track if document is in read-only mode (needs to be accessible throughout function)
    read_only_mode = False
    
    def safe_save_document():
        """Safely save the document, handling read-only mode errors."""
        nonlocal read_only_mode, doc
        if read_only_mode:
            return False
        
        try:
            doc.save(str(project_path))
            return True
        except (OSError, RuntimeError) as save_error:
            error_msg = str(save_error).lower()
            if "read-only" in error_msg or "editing is disabled" in error_msg:
                logger.warning(f"⚠️  Document is in read-only mode, skipping save")
                read_only_mode = True
                return False
            else:
                # Re-raise if it's a different error
                raise
    
    # Use context manager to redirect MetaShape output to log file
    with redirect_metashape_output(log_file_path):
        if project_exists and not clean_intermediate_files:
            logger.info(f"📂 Loading existing project: {project_path}")
            
            # Close any existing documents first to prevent read-only mode
            try:
                # Close the main document if it exists
                if hasattr(Metashape.app, 'document') and Metashape.app.document:
                    try:
                        Metashape.app.document.close()
                    except:
                        pass
                # Close all documents in the app
                if hasattr(Metashape.app, 'documents'):
                    for existing_doc in list(Metashape.app.documents):
                        try:
                            existing_doc.close()
                        except:
                            pass
            except:
                pass
            
            # Wait a moment for files to be released
            import time
            time.sleep(0.5)
            
            doc = Metashape.Document()
            max_retries = 3
            retry_count = 0
            opened_successfully = False
            
            while retry_count < max_retries and not opened_successfully:
                try:
                    doc.open(str(project_path))
                    
                    # Test if document is actually writable by attempting a test save
                    # If it fails, we know it's in read-only mode
                    try:
                        doc.save(str(project_path))
                        opened_successfully = True
                        read_only_mode = False  # Explicitly set to False when writable
                        logger.info("  ✓ Project opened in writable mode")
                    except (OSError, RuntimeError) as save_error:
                        error_msg = str(save_error).lower()
                        if "read-only" in error_msg or "editing is disabled" in error_msg:
                            logger.warning(f"⚠️  Project opened in read-only mode (attempt {retry_count + 1}/{max_retries})")
                            doc.close()
                            # Close all documents again
                            try:
                                if hasattr(Metashape.app, 'document') and Metashape.app.document:
                                    Metashape.app.document.close()
                                if hasattr(Metashape.app, 'documents'):
                                    for existing_doc in list(Metashape.app.documents):
                                        try:
                                            existing_doc.close()
                                        except:
                                            pass
                            except:
                                pass
                            # Wait longer before retrying
                            time.sleep(1.0 * (retry_count + 1))
                            retry_count += 1
                            doc = Metashape.Document()
                            if retry_count >= max_retries:
                                # After max retries, accept read-only mode and continue
                                logger.warning(f"⚠️  Could not open project in writable mode after {max_retries} attempts.")
                                logger.warning(f"⚠️  Continuing in read-only mode (saves will be skipped)")
                                read_only_mode = True
                                # Try to open one more time (will be read-only)
                                try:
                                    doc.open(str(project_path))
                                    opened_successfully = True
                                except:
                                    # If even read-only open fails, recreate
                                    logger.warning(f"⚠️  Recreating project file (existing data will be lost)...")
                                    doc = Metashape.Document()
                                    doc.save(str(project_path))
                                    opened_successfully = True
                                    read_only_mode = False
                                    logger.info("  ✓ Recreated project file in writable mode")
                        else:
                            raise
                            
                except RuntimeError as e:
                    error_msg = str(e).lower()
                    if "already in use" in error_msg or "read-only" in error_msg:
                        logger.warning(f"Project file is already open (attempt {retry_count + 1}/{max_retries}), attempting to close and reopen...")
                        # Try to close any existing document again
                        try:
                            if hasattr(Metashape.app, 'document') and Metashape.app.document:
                                Metashape.app.document.close()
                            if hasattr(Metashape.app, 'documents'):
                                for existing_doc in list(Metashape.app.documents):
                                    try:
                                        existing_doc.close()
                                    except:
                                        pass
                        except:
                            pass
                        # Wait longer before retrying
                        time.sleep(1.0 * (retry_count + 1))
                        retry_count += 1
                        doc = Metashape.Document()
                        if retry_count >= max_retries:
                            # After max retries, try to open in read-only mode
                            logger.warning(f"⚠️  Could not open project after {max_retries} attempts.")
                            logger.warning(f"⚠️  Attempting to open in read-only mode...")
                            try:
                                doc.open(str(project_path))
                                read_only_mode = True
                                opened_successfully = True
                                logger.info("  ✓ Project opened in read-only mode (saves will be skipped)")
                            except:
                                # Last resort: recreate document
                                logger.warning(f"⚠️  Recreating project file (existing data will be lost)...")
                                doc = Metashape.Document()
                                doc.save(str(project_path))
                                opened_successfully = True
                                read_only_mode = False
                                logger.info("  ✓ Recreated project file in writable mode")
                    else:
                        raise
            
            # Use the first chunk (or create one if none exists)
            if len(doc.chunks) > 0:
                chunk = doc.chunks[0]
                logger.info(f"  Found existing chunk with {len(chunk.cameras)} cameras")
            else:
                chunk = doc.addChunk()
                logger.info("  No chunks found, created new chunk")
        else:
            # Cleanup if requested
            if clean_intermediate_files and project_exists:
                logger.info("🧹 Cleaning up previous processing files...")
                cleanup_previous_run(project_path)
            
            # Initialize new project
            logger.info("🚀 Creating new Metashape project...")
            doc = Metashape.Document()
            doc.save(str(project_path))
            chunk = doc.addChunk()
        
        # Check processing status
        status = check_processing_status(chunk)
        logger.info("Processing status:")
        logger.info(f"  Photos added: {status['photos_added']}")
        logger.info(f"  Photos matched: {status['photos_matched']}")
        logger.info(f"  Cameras aligned: {status['cameras_aligned']}")
        logger.info(f"  Depth maps built: {status['depth_maps_built']}")
        logger.info(f"  Model built: {status['model_built']}")
        logger.info(f"  Orthomosaic built: {status['orthomosaic_built']}")
        
        # Add photos (if not already added)
        photos = []
        if not status['photos_added']:
            logger.info(f"Adding photos from: {photos_dir}")
            photos = find_image_files(photos_dir)
            if not photos:
                raise ValueError(f"No images found in {photos_dir}")
            
            logger.info(f"Found {len(photos)} images")
            chunk.addPhotos(photos)
            safe_save_document()
        else:
            logger.info(f"✓ Photos already added ({len(chunk.cameras)} cameras)")
            # Get camera paths - MetaShape Camera objects use 'label' or 'photo' property
            photos = []
            for cam in chunk.cameras:
                try:
                    # Try different ways to get the path
                    if hasattr(cam, 'photo') and cam.photo:
                        photo_path = cam.photo.path if hasattr(cam.photo, 'path') else str(cam.photo)
                    elif hasattr(cam, 'label') and cam.label:
                        photo_path = str(cam.label)
                    elif hasattr(cam, 'path'):
                        photo_path = str(cam.path)
                    else:
                        continue
                    if photo_path:
                        photos.append(photo_path)
                except (AttributeError, TypeError):
                    continue
        
        # Add GCPs if requested (only if not already added)
        if use_gcps:
            existing_markers = len(chunk.markers)
            logger.info("=" * 60)
            logger.info("GROUND CONTROL POINTS (GCPs) CONFIGURATION")
            logger.info("=" * 60)
            if existing_markers == 0:
                if gcp_file and gcp_file.exists():
                    logger.info(f"Loading GCPs from file: {gcp_file}")
                    
                    # Check file extension to determine format
                    file_ext = gcp_file.suffix.lower()
                    
                    if file_ext == '.xml':
                        # MetaShape XML format - try importMarkers first
                        logger.info("  Detected XML format, attempting importMarkers")
                        try:
                            chunk.importMarkers(str(gcp_file))
                            markers_added = len(chunk.markers) - existing_markers
                            logger.info(f"  ✓ Added {markers_added} markers from XML via importMarkers")
                            safe_save_document()
                        except (RuntimeError, Exception) as e:
                            # XML import failed, fall back to CSV parsing (which works more reliably)
                            logger.warning(f"  ⚠️  XML importMarkers failed: {e}")
                            logger.info("  Falling back to CSV-style parsing of XML file...")
                            # Parse XML manually and add markers
                            import xml.etree.ElementTree as ET
                            markers_added = 0
                            try:
                                tree = ET.parse(str(gcp_file))
                                root = tree.getroot()
                                # Find all markers in the XML
                                for marker_elem in root.findall('.//marker'):
                                    try:
                                        label = marker_elem.get('label', f'GCP_{markers_added+1:03d}')
                                        position = marker_elem.find('position')
                                        accuracy_elem = marker_elem.find('accuracy')
                                        
                                        if position is not None:
                                            lon = float(position.get('x', 0.0))
                                            lat = float(position.get('y', 0.0))
                                            z = float(position.get('z', 0.0))
                                            
                                            # Get accuracy (default to gcp_accuracy parameter)
                                            if accuracy_elem is not None:
                                                acc_x = float(accuracy_elem.get('x', gcp_accuracy))
                                                acc_y = float(accuracy_elem.get('y', gcp_accuracy))
                                                acc_z = float(accuracy_elem.get('z', gcp_accuracy))
                                                final_accuracy = min(acc_x, acc_y, acc_z)  # Use minimum
                                            else:
                                                final_accuracy = gcp_accuracy
                                            
                                            # Add marker
                                            marker = chunk.addMarker()
                                            marker.label = label
                                            marker.reference.location = Metashape.Vector([lon, lat, z])
                                            marker.reference.accuracy = Metashape.Vector([final_accuracy, final_accuracy, final_accuracy])
                                            marker.reference.enabled = True
                                            markers_added += 1
                                            logger.debug(f"  Added marker {label}: ({lon:.6f}, {lat:.6f}, {z:.2f}), accuracy={final_accuracy}m")
                                    except (ValueError, AttributeError, TypeError) as parse_error:
                                        logger.warning(f"  Skipping invalid marker in XML: {parse_error}")
                                        continue
                                
                                logger.info(f"  ✓ Added {markers_added} markers from XML (parsed manually)")
                                safe_save_document()
                            except ET.ParseError as parse_err:
                                logger.error(f"  ✗ Failed to parse XML file: {parse_err}")
                                logger.error("  Please check the XML file format or use CSV format instead")
                                raise
                    elif file_ext == '.csv' or file_ext == '.txt':
                        # CSV format - read and add markers manually
                        logger.info("  Detected CSV format, reading and adding markers manually")
                        import csv
                        markers_added = 0
                        with open(gcp_file, 'r', encoding='utf-8') as f:
                            # Try tab delimiter first (MetaShape CSV format), then comma
                            sample = f.read(1024)
                            f.seek(0)
                            tab_char = '\t'
                            delimiter = tab_char if tab_char in sample else ','
                            delimiter_name = 'tab' if delimiter == tab_char else 'comma'
                            logger.info(f"  Using delimiter: {delimiter_name}")
                            reader = csv.DictReader(f, delimiter=delimiter)
                            for row in reader:
                                try:
                                    marker = chunk.addMarker()
                                    marker.label = row.get('Label', f"GCP_{markers_added+1}")
                                    
                                    # Parse coordinates
                                    x = float(row.get('X', row.get('x', 0.0)))
                                    y = float(row.get('Y', row.get('y', 0.0)))
                                    z = float(row.get('Z', row.get('z', 0.0)))
                                    
                                    marker.reference.location = Metashape.Vector((x, y, z))
                                    marker.reference.enabled = True
                                    
                                    # Parse accuracy - use provided gcp_accuracy parameter for high weight
                                    # Lower accuracy values = higher weight in bundle adjustment
                                    # If CSV has accuracy, use the minimum of CSV value and gcp_accuracy
                                    csv_accuracy = float(row.get('Accuracy', row.get('accuracy', gcp_accuracy)))
                                    final_accuracy = min(csv_accuracy, gcp_accuracy) if csv_accuracy > 0 else gcp_accuracy
                                    marker.reference.accuracy = Metashape.Vector((final_accuracy, final_accuracy, final_accuracy))
                                    logger.debug(f"  Marker {marker.label}: accuracy = {final_accuracy}m (weight = 1/{final_accuracy:.3f})")
                                    
                                    markers_added += 1
                                except (ValueError, KeyError) as e:
                                    logger.warning(f"  Skipping invalid marker row: {e}")
                        
                        logger.info(f"  ✓ Added {markers_added} markers from CSV file")
                        safe_save_document()
                    else:
                        # Try to use importMarkers (might work for other formats)
                        logger.info(f"  Unknown format ({file_ext}), attempting importMarkers")
                        try:
                            chunk.importMarkers(str(gcp_file))
                            markers_added = len(chunk.markers) - existing_markers
                            logger.info(f"  ✓ Added {markers_added} markers via importMarkers")
                            safe_save_document()
                        except Exception as e:
                            logger.error(f"  ✗ Failed to import markers: {e}")
                            logger.error("  Please use XML or CSV format, or provide GCPs as a list")
                            raise
                elif gcps:
                    logger.info(f"Using {len(gcps)} GCPs from provided list")
                    logger.info(f"  Setting GCP accuracy to {gcp_accuracy}m for high weight in bundle adjustment")
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
                        # Use gcp_accuracy parameter for high weight (lower = higher weight)
                        # If GCP dict has accuracy, use the minimum
                        gcp_dict_accuracy = gcp.get('accuracy', gcp_accuracy)
                        final_accuracy = min(gcp_dict_accuracy, gcp_accuracy) if gcp_dict_accuracy > 0 else gcp_accuracy
                        marker.reference.accuracy = Metashape.Vector((
                            final_accuracy,
                            final_accuracy,
                            final_accuracy
                        ))
                        logger.debug(f"  Marker {marker.label}: accuracy = {final_accuracy}m")
                    logger.info(f"  ✓ Added {len(gcps)} markers from provided list")
                    safe_save_document()
                else:
                    logger.error("  ✗ use_gcps=True but no GCPs provided (neither gcp_file nor gcps list)!")
                    raise ValueError("use_gcps=True requires either gcp_file or gcps parameter")
            else:
                logger.info(f"✓ GCPs already loaded ({existing_markers} markers)")
            
            # Final GCP summary
            final_marker_count = len(chunk.markers)
            enabled_markers = sum(1 for m in chunk.markers if m.reference.enabled)
            logger.info(f"✓ Total markers in project: {final_marker_count}")
            logger.info(f"✓ Enabled markers (will be used): {enabled_markers}")
            logger.info(f"✓ GCP accuracy setting: {gcp_accuracy}m ({gcp_accuracy*1000:.1f}mm)")
            logger.info("=" * 60)
        elif len(chunk.markers) > 0:
            logger.info(f"Note: {len(chunk.markers)} markers found in project but use_gcps=False")
        
        # Match photos (if not already matched)
        if not status['photos_matched']:
            logger.info("🔍 Matching photos (this may take a while)...")
            chunk.matchPhotos(
                downscale=photo_match_quality,
                tiepoint_limit=tiepoint_limit,
            )
            safe_save_document()
            logger.info("  ✓ Photo matching complete")
        else:
            # Get tie points count safely
            tie_points_count = 0
            if chunk.tie_points is not None:
                try:
                    points = chunk.tie_points.points
                    tie_points_count = len(points) if points is not None else 0
                except (AttributeError, TypeError):
                    tie_points_count = 0
            logger.info(f"✓ Photos already matched ({tie_points_count} tie points) - REUSING existing results")
        
        # Set coordinate accuracy bounds on photos and markers
        # CRITICAL: This MUST be done BEFORE alignCameras() so GCPs have proper weight
        logger.info("Setting coordinate accuracy bounds on photos and markers")
        
        # Set Camera Reference Accuracy to 10m (assumes photos have initial coordinate data from drone GPS)
        chunk.camera_location_accuracy = (10, 10, 10)  # (x, y, z) in meters
        logger.info("  Camera location accuracy set to 10m (x, y, z)")
        
        # Set Marker (GCP) Accuracy - use gcp_accuracy parameter (default 0.05m = 5cm)
        # Lower values = higher weight in bundle adjustment
        # 0.05m (5cm) gives very high weight compared to 10m camera accuracy
        marker_accuracy = gcp_accuracy if use_gcps else 0.005  # Default 5mm if not using GCPs
        chunk.marker_location_accuracy = (marker_accuracy, marker_accuracy, marker_accuracy)  # (x, y, z) in meters
        if use_gcps:
            logger.info(f"  ✓ Marker (GCP) location accuracy set to {marker_accuracy}m ({marker_accuracy*1000:.1f}mm)")
            logger.info(f"  ✓ GCP weight in bundle adjustment: 1/{marker_accuracy:.3f} (vs camera weight: 1/10.000)")
            logger.info(f"  ✓ GCPs will have {10.0/marker_accuracy:.0f}x higher weight than camera poses")
        else:
            logger.info(f"  Marker location accuracy set to 0.005m (5mm) - not using GCPs")
        
        # Align cameras (if not already aligned)
        if not status['cameras_aligned']:
            logger.info("📐 Aligning cameras...")
            if use_gcps and len(chunk.markers) > 0:
                enabled_markers = sum(1 for m in chunk.markers if m.reference.enabled)
                logger.info(f"  ✓ CONFIRMED: Using {enabled_markers} GCPs in bundle adjustment")
                logger.info(f"  ✓ GCP accuracy: {marker_accuracy}m (weight = 1/{marker_accuracy:.3f})")
                logger.info(f"  ✓ Camera accuracy: 10m (weight = 1/10.000)")
                logger.info(f"  ✓ GCPs have {10.0/marker_accuracy:.0f}x higher weight than camera metadata")
                
                # Verify markers are enabled and have correct accuracy
                for i, marker in enumerate(chunk.markers):
                    if marker.reference.enabled:
                        acc = marker.reference.accuracy
                        logger.info(f"    GCP {i+1} ({marker.label}): enabled=True, accuracy=({acc.x:.4f}, {acc.y:.4f}, {acc.z:.4f})m")
            else:
                if use_gcps:
                    logger.warning("  ⚠️  WARNING: use_gcps=True but no markers found! Processing without GCPs.")
                else:
                    logger.info("  Processing without GCPs (use_gcps=False)")
            chunk.alignCameras()
            safe_save_document()
            logger.info("  ✓ Camera alignment complete")
            
            # After alignment, verify GCP usage
            if use_gcps and len(chunk.markers) > 0:
                enabled_markers = [m for m in chunk.markers if m.reference.enabled]
                logger.info(f"  ✓ Post-alignment: {len(enabled_markers)} GCPs remain enabled")
                # Check if markers have projections (indicating they were used)
                markers_with_projections = sum(1 for m in enabled_markers if len(m.projections) > 0)
                logger.info(f"  ✓ {markers_with_projections}/{len(enabled_markers)} GCPs have projections (used in alignment)")
        else:
            aligned_count = sum(1 for cam in chunk.cameras if cam.transform)
            logger.info(f"✓ Cameras already aligned ({aligned_count}/{len(chunk.cameras)} cameras)")
            if use_gcps and len(chunk.markers) > 0:
                enabled_markers = sum(1 for m in chunk.markers if m.reference.enabled)
                markers_with_projections = sum(1 for m in chunk.markers if m.reference.enabled and len(m.projections) > 0)
                logger.info(f"  ✓ {enabled_markers} GCPs enabled, {markers_with_projections} have projections")
        
        # Optimize Cameras to refine the initial alignment using the set reference accuracies
        # This should be done after alignment and accuracy settings, but before building depth maps
        logger.info("Optimizing cameras for improved accuracy")
        logger.info("  Optimizing: f, cx, cy, k1, k2, k3, p1, p2, and coordinates")
        if use_gcps and len(chunk.markers) > 0:
            enabled_markers = sum(1 for m in chunk.markers if m.reference.enabled)
            logger.info(f"  ✓ Using {enabled_markers} GCPs with accuracy {marker_accuracy}m in optimization")
        chunk.optimizeCameras()
        safe_save_document()
        logger.info("  ✓ Camera optimization complete")
        
        # Build depth maps (if not already built)
        if not status['depth_maps_built']:
            logger.info("Building depth maps...")
            chunk.buildDepthMaps(
                downscale=depth_map_quality,
                filter_mode=Metashape.MildFiltering
            )
            safe_save_document()
            logger.info("  ✓ Depth maps built")
        else:
            # Get depth maps count safely
            depth_maps_count = 0
            if chunk.depth_maps is not None:
                try:
                    if hasattr(chunk.depth_maps, '__len__'):
                        depth_maps_count = len(chunk.depth_maps)
                    elif hasattr(chunk.depth_maps, '__iter__'):
                        depth_maps_count = sum(1 for _ in chunk.depth_maps)
                except (TypeError, AttributeError):
                    depth_maps_count = 0
            logger.info(f"✓ Depth maps already built ({depth_maps_count} depth maps) - REUSING existing results")
        
        # Build 3D model (if not already built)
        if not status['model_built']:
            logger.info("Building 3D model...")
            # Verify all camera images are accessible before building model
            logger.info("  Verifying image file accessibility...")
            inaccessible_images = []
            for cam in chunk.cameras:
                try:
                    if hasattr(cam, 'photo') and cam.photo:
                        photo_path = cam.photo.path if hasattr(cam.photo, 'path') else str(cam.photo)
                    elif hasattr(cam, 'label') and cam.label:
                        photo_path = str(cam.label)
                    elif hasattr(cam, 'path'):
                        photo_path = str(cam.path)
                    else:
                        continue
                    
                    if photo_path and not Path(photo_path).exists():
                        inaccessible_images.append(photo_path)
                except (AttributeError, TypeError):
                    continue
            
            if inaccessible_images:
                logger.warning(f"  ⚠️  Found {len(inaccessible_images)} inaccessible image files:")
                for img in inaccessible_images[:5]:  # Show first 5
                    logger.warning(f"    - {img}")
                if len(inaccessible_images) > 5:
                    logger.warning(f"    ... and {len(inaccessible_images) - 5} more")
                logger.warning("  Attempting to build model anyway (MetaShape may handle missing files)...")
            
            # Save project before building model (in case of failure)
            safe_save_document()
            
            # Build model with retry logic for file access errors
            max_retries = 3
            retry_count = 0
            model_built = False
            
            while retry_count < max_retries and not model_built:
                try:
                    # Build model using depth maps as source
                    # Metashape API varies by version - try different approaches
                    build_success = False
                    
                    # Try different API variations
                    # Option 1: Try with DepthMapsData (correct enum name)
                    if not build_success:
                        try:
                            if hasattr(Metashape, 'DataSource') and hasattr(Metashape.DataSource, 'DepthMapsData'):
                                chunk.buildModel(
                                    source=Metashape.DataSource.DepthMapsData,
                                    surface=Metashape.SurfaceType.Arbitrary,
                                    quality=Metashape.Quality.MediumQuality,
                                    face_count=Metashape.FaceCount.HighFaceCount
                                )
                                build_success = True
                                logger.info("  Using DepthMapsData as source")
                        except (AttributeError, TypeError) as e:
                            logger.debug(f"  DepthMapsData approach failed: {e}")
                    
                    # Option 2: Try with quality and face_count only (source defaults to depth maps)
                    if not build_success:
                        try:
                            chunk.buildModel(
                                surface=Metashape.SurfaceType.Arbitrary,
                                quality=Metashape.Quality.MediumQuality,
                                face_count=Metashape.FaceCount.HighFaceCount
                            )
                            build_success = True
                            logger.info("  Using default source (depth maps) with quality/face_count")
                        except (AttributeError, TypeError) as e:
                            logger.debug(f"  Quality/face_count approach failed: {e}")
                    
                    # Option 3: Try minimal parameters (all defaults)
                    if not build_success:
                        try:
                            chunk.buildModel()
                            build_success = True
                            logger.info("  Using default parameters")
                        except Exception as e:
                            logger.debug(f"  Default parameters failed: {e}")
                    
                    if not build_success:
                        raise RuntimeError("Could not build model with any parameter combination")
                    
                    model_built = True
                    safe_save_document()
                    logger.info("  ✓ 3D model built successfully")
                except (OSError, IOError) as e:
                    error_msg = str(e).lower()
                    if "interrupted system call" in error_msg or "can't open file" in error_msg:
                        retry_count += 1
                        if retry_count < max_retries:
                            logger.warning(f"  ⚠️  File access error during model building (attempt {retry_count}/{max_retries})")
                            logger.warning(f"  Error: {e}")
                            logger.info("  Retrying after 5 seconds...")
                            import time
                            time.sleep(5)
                            # Reload project to refresh file handles
                            try:
                                doc.save(str(project_path))
                                doc.open(str(project_path))
                                chunk = doc.chunks[0]
                            except:
                                pass
                        else:
                            logger.error(f"  ✗ Failed to build model after {max_retries} attempts")
                            logger.error(f"  Last error: {e}")
                            logger.error("  Model building failed. You can re-run this cell to resume from this point.")
                            raise
                    else:
                        # Different error, re-raise
                        raise
        else:
            logger.info("✓ 3D model already built - REUSING existing results")
        
        # Check if exported GeoTIFF already exists (before building orthomosaic)
        ortho_path = output_path / f"{product_id}.tif"
        ortho_file_exists = ortho_path.exists() and not clean_intermediate_files
        
        # Build orthomosaic (if not already built AND exported file doesn't exist)
        if not status['orthomosaic_built'] and not ortho_file_exists:
            logger.info("Building orthomosaic...")
            # Save project before building orthomosaic (in case of failure)
            safe_save_document()
            
            # Build orthomosaic with retry logic for file access errors
            max_retries = 3
            retry_count = 0
            ortho_built = False
            
            while retry_count < max_retries and not ortho_built:
                try:
                    chunk.buildOrthomosaic()
                    ortho_built = True
                    safe_save_document()
                    logger.info("  ✓ Orthomosaic built successfully")
                except (OSError, IOError) as e:
                    error_msg = str(e).lower()
                    if "interrupted system call" in error_msg or "can't open file" in error_msg:
                        retry_count += 1
                        if retry_count < max_retries:
                            logger.warning(f"  ⚠️  File access error during orthomosaic building (attempt {retry_count}/{max_retries})")
                            logger.warning(f"  Error: {e}")
                            logger.info("  Retrying after 5 seconds...")
                            import time
                            time.sleep(5)
                            # Reload project to refresh file handles
                            try:
                                doc.save(str(project_path))
                                doc.open(str(project_path))
                                chunk = doc.chunks[0]
                            except:
                                pass
                        else:
                            logger.error(f"  ✗ Failed to build orthomosaic after {max_retries} attempts")
                            logger.error(f"  Last error: {e}")
                            logger.error("  Orthomosaic building failed. You can re-run this cell to resume from this point.")
                            raise
                    else:
                        # Different error, re-raise
                        raise
        elif status['orthomosaic_built']:
            logger.info("✓ Orthomosaic already built in project - REUSING existing results")
        elif ortho_file_exists:
            logger.info(f"✓ Orthomosaic GeoTIFF already exists - REUSING existing file")
            logger.info(f"  File: {ortho_path}")
            existing_size_mb = ortho_path.stat().st_size / (1024 * 1024)
            logger.info(f"  Size: {existing_size_mb:.2f} MB")
        
        # Export GeoTIFF
        # Check if orthomosaic is built in project
        if chunk.orthomosaic is None and not ortho_file_exists:
            logger.warning("⚠️  Orthomosaic not built yet. Cannot export GeoTIFF.")
            logger.warning("  Make sure 'Build orthomosaic' step completed successfully.")
        else:
            # Check if we should skip export
            if ortho_file_exists:
                # Already logged above, just skip export
                pass
            elif ortho_path.exists() and not clean_intermediate_files:
                existing_size_mb = ortho_path.stat().st_size / (1024 * 1024)
                logger.info(f"✓ Orthomosaic GeoTIFF already exists: {ortho_path}")
                logger.info(f"  File size: {existing_size_mb:.2f} MB")
                logger.info("  Skipping export (use clean_intermediate_files=True to force re-export)")
            else:
                logger.info(f"Exporting GeoTIFF to: {ortho_path}")
                logger.info(f"  Full path: {ortho_path.absolute()}")
                logger.info(f"  Output directory exists: {output_path.exists()}")
                
                try:
                    # Ensure output directory exists
                    output_path.mkdir(parents=True, exist_ok=True)
                    
                    chunk.exportRaster(
                        str(ortho_path),
                        image_compression=compression,
                        description=f"Orthomosaic generated by Qualicum Beach GCP Analysis ({'with GCPs' if use_gcps else 'without GCPs'})",
                    )
                    # Save after export (if not read-only)
                    safe_save_document()
                    
                    # Verify the file was created
                    if ortho_path.exists():
                        file_size_mb = ortho_path.stat().st_size / (1024 * 1024)
                        logger.info(f"✓ GeoTIFF exported successfully!")
                        logger.info(f"  File: {ortho_path.absolute()}")
                        logger.info(f"  Size: {file_size_mb:.2f} MB")
                    else:
                        logger.error(f"✗ ERROR: GeoTIFF export completed but file not found!")
                        logger.error(f"  Expected path: {ortho_path.absolute()}")
                        logger.error(f"  Output directory: {output_path.absolute()}")
                        logger.error(f"  Directory exists: {output_path.exists()}")
                        logger.error(f"  Directory contents: {list(output_path.iterdir()) if output_path.exists() else 'N/A'}")
                except Exception as e:
                    logger.error(f"✗ ERROR exporting GeoTIFF: {e}")
                    logger.error(f"  Attempted path: {ortho_path.absolute()}")
                    logger.error(f"  Output directory: {output_path.absolute()}")
                    import traceback
                    logger.error(f"  Traceback: {traceback.format_exc()}")
                    raise
        
        # Get statistics
        stats = {
            'product_id': product_id,
            'use_gcps': use_gcps,
            'num_photos': len(photos) if photos else len(chunk.cameras),
            'num_markers': len(chunk.markers),
            'ortho_path': str(ortho_path),
            'project_path': str(project_path),
            'log_file_path': str(log_file_path),
            'processing_status': status,
            'project_loaded': project_exists and not clean_intermediate_files
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
        logger.info(f"📄 Full verbose log saved to: {log_file_path}")
        
        return stats

