"""Qualicum Beach GCP Analysis Package."""

from .kmz_parser import parse_kmz_file, load_gcps_from_kmz, inspect_kmz_structure
from .basemap_downloader import download_basemap
from .visualization import visualize_gcps_on_basemap, calculate_gcp_bbox, export_basemap_as_png, bbox_to_h3_cells, visualize_feature_matches
from .gcp_exporter import export_to_metashape, export_to_metashape_csv, export_to_metashape_xml
from .s3_downloader import download_all_images_from_input_dir, download_images_from_manifest, parse_manifest_file
from .metashape_processor import process_orthomosaic, PhotoMatchQuality, DepthMapQuality
from .quality_metrics import compare_orthomosaic_to_basemap, calculate_rmse, calculate_mae, detect_seamlines, compute_feature_matching_2d_error, apply_2d_shift_to_orthomosaic, align_orthomosaic_to_gcps, downsample_ortho_to_basemap_resolution
from .report_generator import generate_comparison_report, generate_markdown_report, convert_to_json_serializable
from .latex_report_generator import generate_latex_report
from .visualization_comparison import create_error_visualization, create_error_visualization_memory_efficient, create_seamline_visualization, create_comparison_side_by_side, create_metrics_summary_plot

__all__ = [
    'parse_kmz_file',
    'load_gcps_from_kmz',
    'inspect_kmz_structure',
    'download_basemap',
    'visualize_gcps_on_basemap',
    'calculate_gcp_bbox',
    'export_basemap_as_png',
    'bbox_to_h3_cells',
    'visualize_feature_matches',
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
    'compute_feature_matching_2d_error',
    'apply_2d_shift_to_orthomosaic',
    'align_orthomosaic_to_gcps',
    'generate_comparison_report',
    'generate_markdown_report',
    'convert_to_json_serializable',
    'generate_latex_report',
    'create_error_visualization',
    'create_error_visualization_memory_efficient',
    'create_seamline_visualization',
    'create_comparison_side_by_side',
    'create_metrics_summary_plot',
]
