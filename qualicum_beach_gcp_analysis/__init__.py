"""Qualicum Beach GCP Analysis Package."""

from .kmz_parser import parse_kmz_file, load_gcps_from_kmz, inspect_kmz_structure
from .basemap_downloader import download_basemap
from .visualization import visualize_gcps_on_basemap, calculate_gcp_bbox, export_basemap_as_png, bbox_to_h3_cells
from .gcp_exporter import export_to_metashape, export_to_metashape_csv, export_to_metashape_xml
from .s3_downloader import download_all_images_from_input_dir, download_images_from_manifest, parse_manifest_file
from .metashape_processor import process_orthomosaic, PhotoMatchQuality, DepthMapQuality
from .quality_metrics import compare_orthomosaic_to_basemap, calculate_rmse, calculate_mae, detect_seamlines
from .report_generator import generate_comparison_report, generate_markdown_report

__all__ = [
    'parse_kmz_file',
    'load_gcps_from_kmz',
    'inspect_kmz_structure',
    'download_basemap',
    'visualize_gcps_on_basemap',
    'calculate_gcp_bbox',
    'export_basemap_as_png',
    'bbox_to_h3_cells',
    'export_to_metashape',
    'export_to_metashape_csv',
    'export_to_metashape_xml',
    'download_all_images_from_input_dir',
    'download_images_from_manifest',
    'parse_manifest_file',
    'process_orthomosaic',
    'PhotoMatchQuality',
    'DepthMapQuality',
    'compare_orthomosaic_to_basemap',
    'calculate_rmse',
    'calculate_mae',
    'detect_seamlines',
    'generate_comparison_report',
    'generate_markdown_report',
]
