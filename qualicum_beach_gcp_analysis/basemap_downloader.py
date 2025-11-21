"""Basemap downloader utilities for downloading map tiles."""

import math
import time
import requests
from typing import Tuple, Optional
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import io
from pathlib import Path


def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
    """Convert lat/lon to tile coordinates."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)


def num2deg(xtile: int, ytile: int, zoom: int) -> Tuple[float, float]:
    """Convert tile coordinates to lat/lon of top-left corner."""
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)


def get_tile_url(xtile: int, ytile: int, zoom: int, source: str = "openstreetmap") -> str:
    """Get URL for a tile."""
    if source == "openstreetmap":
        return f"https://tile.openstreetmap.org/{zoom}/{xtile}/{ytile}.png"
    elif source == "esri_world_imagery":
        return f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{ytile}/{xtile}"
    else:
        raise ValueError(f"Unknown tile source: {source}")


def download_tile(xtile: int, ytile: int, zoom: int, source: str = "openstreetmap", 
                  verbose: bool = False, retries: int = 2) -> Optional[Image.Image]:
    """Download a single tile with retry logic."""
    url = get_tile_url(xtile, ytile, zoom, source)
    
    headers = {
        'User-Agent': 'qualicum-beach-gcp-analysis/0.1.0'
    }
    
    for attempt in range(retries + 1):
        try:
            response = requests.get(url, timeout=10, headers=headers)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
            return img
        except requests.exceptions.HTTPError as e:
            if attempt < retries:
                time.sleep(0.1 * (attempt + 1))
                continue
            if verbose:
                status_code = e.response.status_code if hasattr(e, 'response') and e.response is not None else 'unknown'
                print(f"Warning: HTTP error downloading tile {zoom}/{xtile}/{ytile}: {e} (Status: {status_code})")
            return None
        except requests.exceptions.RequestException as e:
            if attempt < retries:
                time.sleep(0.1 * (attempt + 1))
                continue
            if verbose:
                print(f"Warning: Request error downloading tile {zoom}/{xtile}/{ytile}: {e}")
            return None
        except Exception as e:
            if attempt < retries:
                time.sleep(0.1 * (attempt + 1))
                continue
            if verbose:
                print(f"Warning: Failed to download tile {zoom}/{xtile}/{ytile}: {e}")
            return None
    
    return None


def calculate_zoom_level(bbox: Tuple[float, float, float, float], 
                         max_tiles: int = 64, 
                         target_resolution: Optional[float] = None) -> int:
    """
    Calculate appropriate zoom level based on bounding box size or target resolution.
    
    Args:
        bbox: Bounding box as (min_lat, min_lon, max_lat, max_lon)
        max_tiles: Maximum number of tiles to download (default 64 for better quality)
        target_resolution: Target resolution in meters per pixel (optional)
        
    Returns:
        Zoom level
    """
    min_lat, min_lon, max_lat, max_lon = bbox
    
    if target_resolution:
        # Calculate zoom based on target resolution
        # Approximate: at equator, 1 tile at zoom z covers ~156543 meters / 2^z
        center_lat = (min_lat + max_lat) / 2
        meters_per_pixel_at_equator = 156543.03392
        meters_per_pixel = meters_per_pixel_at_equator * math.cos(math.radians(center_lat))
        
        for zoom in range(1, 20):
            tile_size_meters = meters_per_pixel * 256 / (2 ** zoom)
            if tile_size_meters <= target_resolution:
                return zoom
        return 18
    
    # Calculate based on bounding box size
    lat_range = max_lat - min_lat
    lon_range = max_lon - min_lon
    
    # Calculate approximate area in square degrees
    area_deg2 = lat_range * lon_range
    
    # For small areas, use higher zoom levels
    # Area thresholds (rough estimates):
    # - Very small (< 0.001 deg²): zoom 15-17
    # - Small (0.001-0.01 deg²): zoom 13-15
    # - Medium (0.01-0.1 deg²): zoom 11-13
    # - Large (> 0.1 deg²): zoom 9-11
    
    if area_deg2 < 0.0001:
        # Very small area - use high zoom
        base_zoom = 16
    elif area_deg2 < 0.001:
        base_zoom = 15
    elif area_deg2 < 0.01:
        base_zoom = 13
    elif area_deg2 < 0.1:
        base_zoom = 11
    else:
        base_zoom = 9
    
    # Now check tile count and adjust if needed
    for zoom in range(base_zoom, base_zoom - 5, -1):
        if zoom < 1:
            break
        xtile_min, ytile_min = deg2num(min_lat, min_lon, zoom)
        xtile_max, ytile_max = deg2num(max_lat, max_lon, zoom)
        
        num_tiles = (xtile_max - xtile_min + 1) * (ytile_max - ytile_min + 1)
        if num_tiles <= max_tiles:
            return zoom
    
    # Fallback: use base zoom even if it exceeds max_tiles
    return max(1, base_zoom)


def download_basemap(
    bbox: Tuple[float, float, float, float],
    output_path: str,
    source: str = "openstreetmap",
    zoom: Optional[int] = None,
    target_resolution: Optional[float] = None
) -> str:
    """
    Download basemap tiles and create a GeoTIFF.
    
    Args:
        bbox: Bounding box as (min_lat, min_lon, max_lat, max_lon)
        output_path: Path to save GeoTIFF
        source: Tile source ('openstreetmap' or 'esri_world_imagery')
        zoom: Zoom level (auto-calculated if None)
        target_resolution: Target resolution in meters per pixel
        
    Returns:
        Path to saved GeoTIFF
    """
    min_lat, min_lon, max_lat, max_lon = bbox
    
    # Validate bounding box
    if min_lat >= max_lat:
        raise ValueError(f"Invalid bounding box: min_lat ({min_lat}) must be less than max_lat ({max_lat})")
    if min_lon >= max_lon:
        raise ValueError(f"Invalid bounding box: min_lon ({min_lon}) must be less than max_lon ({max_lon})")
    
    if zoom is None:
        zoom = calculate_zoom_level(bbox, target_resolution=target_resolution)
    
    print(f"Downloading basemap at zoom level {zoom}...")
    
    # Calculate tile range
    # Note: Y tiles increase southward, so min_lat gives max Y tile and vice versa
    xtile_min, ytile_max = deg2num(min_lat, min_lon, zoom)  # min_lat, min_lon -> top-left
    xtile_max, ytile_min = deg2num(max_lat, max_lon, zoom)  # max_lat, max_lon -> bottom-right
    
    # Ensure correct ordering
    if xtile_min > xtile_max:
        xtile_min, xtile_max = xtile_max, xtile_min
    if ytile_min > ytile_max:
        ytile_min, ytile_max = ytile_max, ytile_min
    
    print(f"Tile range: X [{xtile_min}, {xtile_max}], Y [{ytile_min}, {ytile_max}]")
    
    # Download tiles
    tiles = []
    for y in range(ytile_min, ytile_max + 1):
        row = []
        for x in range(xtile_min, xtile_max + 1):
            tile = download_tile(x, y, zoom, source, verbose=True)
            if tile is None:
                # Create blank tile
                tile = Image.new('RGB', (256, 256), color=(128, 128, 128))
            row.append(tile)
        tiles.append(row)
    
    # Stitch tiles together
    tile_height = tiles[0][0].height
    tile_width = tiles[0][0].width
    
    stitched = Image.new('RGB', 
                        ((xtile_max - xtile_min + 1) * tile_width,
                         (ytile_max - ytile_min + 1) * tile_height))
    
    for y_idx, row in enumerate(tiles):
        for x_idx, tile in enumerate(row):
            x_pos = (x_idx) * tile_width
            y_pos = (y_idx) * tile_height
            stitched.paste(tile, (x_pos, y_pos))
    
    # Get bounds of stitched image
    top_left_lat, top_left_lon = num2deg(xtile_min, ytile_min, zoom)
    bottom_right_lat, bottom_right_lon = num2deg(xtile_max + 1, ytile_max + 1, zoom)
    
    # Crop to requested bounds
    # Calculate pixel positions
    pixels_per_degree_lon = stitched.width / (bottom_right_lon - top_left_lon)
    pixels_per_degree_lat = stitched.height / (top_left_lat - bottom_right_lat)
    
    left_pixel = int((min_lon - top_left_lon) * pixels_per_degree_lon)
    top_pixel = int((top_left_lat - max_lat) * pixels_per_degree_lat)
    right_pixel = int((max_lon - top_left_lon) * pixels_per_degree_lon)
    bottom_pixel = int((top_left_lat - min_lat) * pixels_per_degree_lat)
    
    left_pixel = max(0, left_pixel)
    top_pixel = max(0, top_pixel)
    right_pixel = min(stitched.width, right_pixel)
    bottom_pixel = min(stitched.height, bottom_pixel)
    
    # Ensure valid crop rectangle
    if right_pixel <= left_pixel:
        right_pixel = left_pixel + 1
        if right_pixel > stitched.width:
            left_pixel = stitched.width - 1
            right_pixel = stitched.width
    if bottom_pixel <= top_pixel:
        bottom_pixel = top_pixel + 1
        if bottom_pixel > stitched.height:
            top_pixel = stitched.height - 1
            bottom_pixel = stitched.height
    
    cropped = stitched.crop((left_pixel, top_pixel, right_pixel, bottom_pixel))
    
    # Save as GeoTIFF
    width, height = cropped.size
    
    # Validate dimensions
    if width == 0 or height == 0:
        raise ValueError(f"Invalid cropped image dimensions: {width}x{height}. "
                        f"This may indicate an invalid bounding box or tile calculation issue. "
                        f"Bbox: ({min_lat}, {min_lon}, {max_lat}, {max_lon}), "
                        f"Crop coords: ({left_pixel}, {top_pixel}, {right_pixel}, {bottom_pixel}), "
                        f"Stitched size: {stitched.width}x{stitched.height}")
    
    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)
    
    array = np.array(cropped)
    
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=3,
        dtype=array.dtype,
        crs=CRS.from_epsg(4326),  # WGS84
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(array.transpose(2, 0, 1))
    
    print(f"Basemap saved to {output_path}")
    return output_path


import math
import time
import requests
from typing import Tuple, Optional
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import io
from pathlib import Path


def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
    """Convert lat/lon to tile coordinates."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)


def num2deg(xtile: int, ytile: int, zoom: int) -> Tuple[float, float]:
    """Convert tile coordinates to lat/lon of top-left corner."""
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)


def get_tile_url(xtile: int, ytile: int, zoom: int, source: str = "openstreetmap") -> str:
    """Get URL for a tile."""
    if source == "openstreetmap":
        return f"https://tile.openstreetmap.org/{zoom}/{xtile}/{ytile}.png"
    elif source == "esri_world_imagery":
        return f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{ytile}/{xtile}"
    else:
        raise ValueError(f"Unknown tile source: {source}")


def download_tile(xtile: int, ytile: int, zoom: int, source: str = "openstreetmap", 
                  verbose: bool = False, retries: int = 2) -> Optional[Image.Image]:
    """Download a single tile with retry logic."""
    url = get_tile_url(xtile, ytile, zoom, source)
    
    headers = {
        'User-Agent': 'qualicum-beach-gcp-analysis/0.1.0'
    }
    
    for attempt in range(retries + 1):
        try:
            response = requests.get(url, timeout=10, headers=headers)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
            return img
        except requests.exceptions.HTTPError as e:
            if attempt < retries:
                time.sleep(0.1 * (attempt + 1))
                continue
            if verbose:
                status_code = e.response.status_code if hasattr(e, 'response') and e.response is not None else 'unknown'
                print(f"Warning: HTTP error downloading tile {zoom}/{xtile}/{ytile}: {e} (Status: {status_code})")
            return None
        except requests.exceptions.RequestException as e:
            if attempt < retries:
                time.sleep(0.1 * (attempt + 1))
                continue
            if verbose:
                print(f"Warning: Request error downloading tile {zoom}/{xtile}/{ytile}: {e}")
            return None
        except Exception as e:
            if attempt < retries:
                time.sleep(0.1 * (attempt + 1))
                continue
            if verbose:
                print(f"Warning: Failed to download tile {zoom}/{xtile}/{ytile}: {e}")
            return None
    
    return None


def calculate_zoom_level(bbox: Tuple[float, float, float, float], 
                         max_tiles: int = 64, 
                         target_resolution: Optional[float] = None) -> int:
    """
    Calculate appropriate zoom level based on bounding box size or target resolution.
    
    Args:
        bbox: Bounding box as (min_lat, min_lon, max_lat, max_lon)
        max_tiles: Maximum number of tiles to download (default 64 for better quality)
        target_resolution: Target resolution in meters per pixel (optional)
        
    Returns:
        Zoom level
    """
    min_lat, min_lon, max_lat, max_lon = bbox
    
    if target_resolution:
        # Calculate zoom based on target resolution
        # Approximate: at equator, 1 tile at zoom z covers ~156543 meters / 2^z
        center_lat = (min_lat + max_lat) / 2
        meters_per_pixel_at_equator = 156543.03392
        meters_per_pixel = meters_per_pixel_at_equator * math.cos(math.radians(center_lat))
        
        for zoom in range(1, 20):
            tile_size_meters = meters_per_pixel * 256 / (2 ** zoom)
            if tile_size_meters <= target_resolution:
                return zoom
        return 18
    
    # Calculate based on bounding box size
    lat_range = max_lat - min_lat
    lon_range = max_lon - min_lon
    
    # Calculate approximate area in square degrees
    area_deg2 = lat_range * lon_range
    
    # For small areas, use higher zoom levels
    # Area thresholds (rough estimates):
    # - Very small (< 0.001 deg²): zoom 15-17
    # - Small (0.001-0.01 deg²): zoom 13-15
    # - Medium (0.01-0.1 deg²): zoom 11-13
    # - Large (> 0.1 deg²): zoom 9-11
    
    if area_deg2 < 0.0001:
        # Very small area - use high zoom
        base_zoom = 16
    elif area_deg2 < 0.001:
        base_zoom = 15
    elif area_deg2 < 0.01:
        base_zoom = 13
    elif area_deg2 < 0.1:
        base_zoom = 11
    else:
        base_zoom = 9
    
    # Now check tile count and adjust if needed
    for zoom in range(base_zoom, base_zoom - 5, -1):
        if zoom < 1:
            break
        xtile_min, ytile_min = deg2num(min_lat, min_lon, zoom)
        xtile_max, ytile_max = deg2num(max_lat, max_lon, zoom)
        
        num_tiles = (xtile_max - xtile_min + 1) * (ytile_max - ytile_min + 1)
        if num_tiles <= max_tiles:
            return zoom
    
    # Fallback: use base zoom even if it exceeds max_tiles
    return max(1, base_zoom)


def download_basemap(
    bbox: Tuple[float, float, float, float],
    output_path: str,
    source: str = "openstreetmap",
    zoom: Optional[int] = None,
    target_resolution: Optional[float] = None
) -> str:
    """
    Download basemap tiles and create a GeoTIFF.
    
    Args:
        bbox: Bounding box as (min_lat, min_lon, max_lat, max_lon)
        output_path: Path to save GeoTIFF
        source: Tile source ('openstreetmap' or 'esri_world_imagery')
        zoom: Zoom level (auto-calculated if None)
        target_resolution: Target resolution in meters per pixel
        
    Returns:
        Path to saved GeoTIFF
    """
    min_lat, min_lon, max_lat, max_lon = bbox
    
    # Validate bounding box
    if min_lat >= max_lat:
        raise ValueError(f"Invalid bounding box: min_lat ({min_lat}) must be less than max_lat ({max_lat})")
    if min_lon >= max_lon:
        raise ValueError(f"Invalid bounding box: min_lon ({min_lon}) must be less than max_lon ({max_lon})")
    
    if zoom is None:
        zoom = calculate_zoom_level(bbox, target_resolution=target_resolution)
    
    print(f"Downloading basemap at zoom level {zoom}...")
    
    # Calculate tile range
    # Note: Y tiles increase southward, so min_lat gives max Y tile and vice versa
    xtile_min, ytile_max = deg2num(min_lat, min_lon, zoom)  # min_lat, min_lon -> top-left
    xtile_max, ytile_min = deg2num(max_lat, max_lon, zoom)  # max_lat, max_lon -> bottom-right
    
    # Ensure correct ordering
    if xtile_min > xtile_max:
        xtile_min, xtile_max = xtile_max, xtile_min
    if ytile_min > ytile_max:
        ytile_min, ytile_max = ytile_max, ytile_min
    
    print(f"Tile range: X [{xtile_min}, {xtile_max}], Y [{ytile_min}, {ytile_max}]")
    
    # Download tiles
    tiles = []
    for y in range(ytile_min, ytile_max + 1):
        row = []
        for x in range(xtile_min, xtile_max + 1):
            tile = download_tile(x, y, zoom, source, verbose=True)
            if tile is None:
                # Create blank tile
                tile = Image.new('RGB', (256, 256), color=(128, 128, 128))
            row.append(tile)
        tiles.append(row)
    
    # Stitch tiles together
    tile_height = tiles[0][0].height
    tile_width = tiles[0][0].width
    
    stitched = Image.new('RGB', 
                        ((xtile_max - xtile_min + 1) * tile_width,
                         (ytile_max - ytile_min + 1) * tile_height))
    
    for y_idx, row in enumerate(tiles):
        for x_idx, tile in enumerate(row):
            x_pos = (x_idx) * tile_width
            y_pos = (y_idx) * tile_height
            stitched.paste(tile, (x_pos, y_pos))
    
    # Get bounds of stitched image
    top_left_lat, top_left_lon = num2deg(xtile_min, ytile_min, zoom)
    bottom_right_lat, bottom_right_lon = num2deg(xtile_max + 1, ytile_max + 1, zoom)
    
    # Crop to requested bounds
    # Calculate pixel positions
    pixels_per_degree_lon = stitched.width / (bottom_right_lon - top_left_lon)
    pixels_per_degree_lat = stitched.height / (top_left_lat - bottom_right_lat)
    
    left_pixel = int((min_lon - top_left_lon) * pixels_per_degree_lon)
    top_pixel = int((top_left_lat - max_lat) * pixels_per_degree_lat)
    right_pixel = int((max_lon - top_left_lon) * pixels_per_degree_lon)
    bottom_pixel = int((top_left_lat - min_lat) * pixels_per_degree_lat)
    
    left_pixel = max(0, left_pixel)
    top_pixel = max(0, top_pixel)
    right_pixel = min(stitched.width, right_pixel)
    bottom_pixel = min(stitched.height, bottom_pixel)
    
    # Ensure valid crop rectangle
    if right_pixel <= left_pixel:
        right_pixel = left_pixel + 1
        if right_pixel > stitched.width:
            left_pixel = stitched.width - 1
            right_pixel = stitched.width
    if bottom_pixel <= top_pixel:
        bottom_pixel = top_pixel + 1
        if bottom_pixel > stitched.height:
            top_pixel = stitched.height - 1
            bottom_pixel = stitched.height
    
    cropped = stitched.crop((left_pixel, top_pixel, right_pixel, bottom_pixel))
    
    # Save as GeoTIFF
    width, height = cropped.size
    
    # Validate dimensions
    if width == 0 or height == 0:
        raise ValueError(f"Invalid cropped image dimensions: {width}x{height}. "
                        f"This may indicate an invalid bounding box or tile calculation issue. "
                        f"Bbox: ({min_lat}, {min_lon}, {max_lat}, {max_lon}), "
                        f"Crop coords: ({left_pixel}, {top_pixel}, {right_pixel}, {bottom_pixel}), "
                        f"Stitched size: {stitched.width}x{stitched.height}")
    
    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)
    
    array = np.array(cropped)
    
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=3,
        dtype=array.dtype,
        crs=CRS.from_epsg(4326),  # WGS84
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(array.transpose(2, 0, 1))
    
    print(f"Basemap saved to {output_path}")
    return output_path


import math
import time
import requests
from typing import Tuple, Optional
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import io
from pathlib import Path


def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
    """Convert lat/lon to tile coordinates."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)


def num2deg(xtile: int, ytile: int, zoom: int) -> Tuple[float, float]:
    """Convert tile coordinates to lat/lon of top-left corner."""
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)


def get_tile_url(xtile: int, ytile: int, zoom: int, source: str = "openstreetmap") -> str:
    """Get URL for a tile."""
    if source == "openstreetmap":
        return f"https://tile.openstreetmap.org/{zoom}/{xtile}/{ytile}.png"
    elif source == "esri_world_imagery":
        return f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{ytile}/{xtile}"
    else:
        raise ValueError(f"Unknown tile source: {source}")


def download_tile(xtile: int, ytile: int, zoom: int, source: str = "openstreetmap", 
                  verbose: bool = False, retries: int = 2) -> Optional[Image.Image]:
    """Download a single tile with retry logic."""
    url = get_tile_url(xtile, ytile, zoom, source)
    
    headers = {
        'User-Agent': 'qualicum-beach-gcp-analysis/0.1.0'
    }
    
    for attempt in range(retries + 1):
        try:
            response = requests.get(url, timeout=10, headers=headers)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
            return img
        except requests.exceptions.HTTPError as e:
            if attempt < retries:
                time.sleep(0.1 * (attempt + 1))
                continue
            if verbose:
                status_code = e.response.status_code if hasattr(e, 'response') and e.response is not None else 'unknown'
                print(f"Warning: HTTP error downloading tile {zoom}/{xtile}/{ytile}: {e} (Status: {status_code})")
            return None
        except requests.exceptions.RequestException as e:
            if attempt < retries:
                time.sleep(0.1 * (attempt + 1))
                continue
            if verbose:
                print(f"Warning: Request error downloading tile {zoom}/{xtile}/{ytile}: {e}")
            return None
        except Exception as e:
            if attempt < retries:
                time.sleep(0.1 * (attempt + 1))
                continue
            if verbose:
                print(f"Warning: Failed to download tile {zoom}/{xtile}/{ytile}: {e}")
            return None
    
    return None


def calculate_zoom_level(bbox: Tuple[float, float, float, float], 
                         max_tiles: int = 64, 
                         target_resolution: Optional[float] = None) -> int:
    """
    Calculate appropriate zoom level based on bounding box size or target resolution.
    
    Args:
        bbox: Bounding box as (min_lat, min_lon, max_lat, max_lon)
        max_tiles: Maximum number of tiles to download (default 64 for better quality)
        target_resolution: Target resolution in meters per pixel (optional)
        
    Returns:
        Zoom level
    """
    min_lat, min_lon, max_lat, max_lon = bbox
    
    if target_resolution:
        # Calculate zoom based on target resolution
        # Approximate: at equator, 1 tile at zoom z covers ~156543 meters / 2^z
        center_lat = (min_lat + max_lat) / 2
        meters_per_pixel_at_equator = 156543.03392
        meters_per_pixel = meters_per_pixel_at_equator * math.cos(math.radians(center_lat))
        
        for zoom in range(1, 20):
            tile_size_meters = meters_per_pixel * 256 / (2 ** zoom)
            if tile_size_meters <= target_resolution:
                return zoom
        return 18
    
    # Calculate based on bounding box size
    lat_range = max_lat - min_lat
    lon_range = max_lon - min_lon
    
    # Calculate approximate area in square degrees
    area_deg2 = lat_range * lon_range
    
    # For small areas, use higher zoom levels
    # Area thresholds (rough estimates):
    # - Very small (< 0.001 deg²): zoom 15-17
    # - Small (0.001-0.01 deg²): zoom 13-15
    # - Medium (0.01-0.1 deg²): zoom 11-13
    # - Large (> 0.1 deg²): zoom 9-11
    
    if area_deg2 < 0.0001:
        # Very small area - use high zoom
        base_zoom = 16
    elif area_deg2 < 0.001:
        base_zoom = 15
    elif area_deg2 < 0.01:
        base_zoom = 13
    elif area_deg2 < 0.1:
        base_zoom = 11
    else:
        base_zoom = 9
    
    # Now check tile count and adjust if needed
    for zoom in range(base_zoom, base_zoom - 5, -1):
        if zoom < 1:
            break
        xtile_min, ytile_min = deg2num(min_lat, min_lon, zoom)
        xtile_max, ytile_max = deg2num(max_lat, max_lon, zoom)
        
        num_tiles = (xtile_max - xtile_min + 1) * (ytile_max - ytile_min + 1)
        if num_tiles <= max_tiles:
            return zoom
    
    # Fallback: use base zoom even if it exceeds max_tiles
    return max(1, base_zoom)


def download_basemap(
    bbox: Tuple[float, float, float, float],
    output_path: str,
    source: str = "openstreetmap",
    zoom: Optional[int] = None,
    target_resolution: Optional[float] = None
) -> str:
    """
    Download basemap tiles and create a GeoTIFF.
    
    Args:
        bbox: Bounding box as (min_lat, min_lon, max_lat, max_lon)
        output_path: Path to save GeoTIFF
        source: Tile source ('openstreetmap' or 'esri_world_imagery')
        zoom: Zoom level (auto-calculated if None)
        target_resolution: Target resolution in meters per pixel
        
    Returns:
        Path to saved GeoTIFF
    """
    min_lat, min_lon, max_lat, max_lon = bbox
    
    # Validate bounding box
    if min_lat >= max_lat:
        raise ValueError(f"Invalid bounding box: min_lat ({min_lat}) must be less than max_lat ({max_lat})")
    if min_lon >= max_lon:
        raise ValueError(f"Invalid bounding box: min_lon ({min_lon}) must be less than max_lon ({max_lon})")
    
    if zoom is None:
        zoom = calculate_zoom_level(bbox, target_resolution=target_resolution)
    
    print(f"Downloading basemap at zoom level {zoom}...")
    
    # Calculate tile range
    # Note: Y tiles increase southward, so min_lat gives max Y tile and vice versa
    xtile_min, ytile_max = deg2num(min_lat, min_lon, zoom)  # min_lat, min_lon -> top-left
    xtile_max, ytile_min = deg2num(max_lat, max_lon, zoom)  # max_lat, max_lon -> bottom-right
    
    # Ensure correct ordering
    if xtile_min > xtile_max:
        xtile_min, xtile_max = xtile_max, xtile_min
    if ytile_min > ytile_max:
        ytile_min, ytile_max = ytile_max, ytile_min
    
    print(f"Tile range: X [{xtile_min}, {xtile_max}], Y [{ytile_min}, {ytile_max}]")
    
    # Download tiles
    tiles = []
    for y in range(ytile_min, ytile_max + 1):
        row = []
        for x in range(xtile_min, xtile_max + 1):
            tile = download_tile(x, y, zoom, source, verbose=True)
            if tile is None:
                # Create blank tile
                tile = Image.new('RGB', (256, 256), color=(128, 128, 128))
            row.append(tile)
        tiles.append(row)
    
    # Stitch tiles together
    tile_height = tiles[0][0].height
    tile_width = tiles[0][0].width
    
    stitched = Image.new('RGB', 
                        ((xtile_max - xtile_min + 1) * tile_width,
                         (ytile_max - ytile_min + 1) * tile_height))
    
    for y_idx, row in enumerate(tiles):
        for x_idx, tile in enumerate(row):
            x_pos = (x_idx) * tile_width
            y_pos = (y_idx) * tile_height
            stitched.paste(tile, (x_pos, y_pos))
    
    # Get bounds of stitched image
    top_left_lat, top_left_lon = num2deg(xtile_min, ytile_min, zoom)
    bottom_right_lat, bottom_right_lon = num2deg(xtile_max + 1, ytile_max + 1, zoom)
    
    # Crop to requested bounds
    # Calculate pixel positions
    pixels_per_degree_lon = stitched.width / (bottom_right_lon - top_left_lon)
    pixels_per_degree_lat = stitched.height / (top_left_lat - bottom_right_lat)
    
    left_pixel = int((min_lon - top_left_lon) * pixels_per_degree_lon)
    top_pixel = int((top_left_lat - max_lat) * pixels_per_degree_lat)
    right_pixel = int((max_lon - top_left_lon) * pixels_per_degree_lon)
    bottom_pixel = int((top_left_lat - min_lat) * pixels_per_degree_lat)
    
    left_pixel = max(0, left_pixel)
    top_pixel = max(0, top_pixel)
    right_pixel = min(stitched.width, right_pixel)
    bottom_pixel = min(stitched.height, bottom_pixel)
    
    # Ensure valid crop rectangle
    if right_pixel <= left_pixel:
        right_pixel = left_pixel + 1
        if right_pixel > stitched.width:
            left_pixel = stitched.width - 1
            right_pixel = stitched.width
    if bottom_pixel <= top_pixel:
        bottom_pixel = top_pixel + 1
        if bottom_pixel > stitched.height:
            top_pixel = stitched.height - 1
            bottom_pixel = stitched.height
    
    cropped = stitched.crop((left_pixel, top_pixel, right_pixel, bottom_pixel))
    
    # Save as GeoTIFF
    width, height = cropped.size
    
    # Validate dimensions
    if width == 0 or height == 0:
        raise ValueError(f"Invalid cropped image dimensions: {width}x{height}. "
                        f"This may indicate an invalid bounding box or tile calculation issue. "
                        f"Bbox: ({min_lat}, {min_lon}, {max_lat}, {max_lon}), "
                        f"Crop coords: ({left_pixel}, {top_pixel}, {right_pixel}, {bottom_pixel}), "
                        f"Stitched size: {stitched.width}x{stitched.height}")
    
    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)
    
    array = np.array(cropped)
    
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=3,
        dtype=array.dtype,
        crs=CRS.from_epsg(4326),  # WGS84
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(array.transpose(2, 0, 1))
    
    print(f"Basemap saved to {output_path}")
    return output_path

