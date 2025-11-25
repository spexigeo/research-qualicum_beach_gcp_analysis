"""
Report Generation for Orthomosaic Quality Analysis.

Generates comprehensive reports comparing orthomosaics with and without GCPs
against reference basemaps.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import numpy as np

logger = logging.getLogger(__name__)


def convert_to_json_serializable(obj: Any) -> Any:
    """
    Convert numpy types and other non-JSON-serializable objects to native Python types.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj


def generate_comparison_report(
    metrics_with_gcps: Dict,
    metrics_without_gcps: Dict,
    output_path: Path,
    basemap_source: str = "ESRI"
) -> Path:
    """
    Generate a comprehensive comparison report.
    
    Args:
        metrics_with_gcps: Quality metrics for orthomosaic with GCPs
        metrics_without_gcps: Quality metrics for orthomosaic without GCPs
        output_path: Path to save the report (JSON)
        basemap_source: Source of reference basemap (ESRI or OpenStreetMap)
        
    Returns:
        Path to saved report
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        'report_metadata': {
            'generated_at': datetime.now().isoformat(),
            'basemap_source': basemap_source,
            'comparison_type': 'with_gcps_vs_without_gcps'
        },
        'orthomosaic_with_gcps': metrics_with_gcps,
        'orthomosaic_without_gcps': metrics_without_gcps,
        'comparison': {}
    }
    
    # Compare overall metrics
    if 'overall' in metrics_with_gcps and 'overall' in metrics_without_gcps:
        with_gcps = metrics_with_gcps['overall']
        without_gcps = metrics_without_gcps['overall']
        
        comparison = {
            'rmse_improvement': None,
            'mae_improvement': None,
            'similarity_improvement': None,
            'seamline_reduction': None
        }
        
        # RMSE improvement (lower is better)
        if with_gcps.get('rmse') and without_gcps.get('rmse'):
            rmse_improvement = ((without_gcps['rmse'] - with_gcps['rmse']) / without_gcps['rmse']) * 100
            comparison['rmse_improvement'] = {
                'percentage': float(rmse_improvement),
                'absolute': float(without_gcps['rmse'] - with_gcps['rmse']),
                'with_gcps': with_gcps['rmse'],
                'without_gcps': without_gcps['rmse']
            }
        
        # MAE improvement (lower is better)
        if with_gcps.get('mae') and without_gcps.get('mae'):
            mae_improvement = ((without_gcps['mae'] - with_gcps['mae']) / without_gcps['mae']) * 100
            comparison['mae_improvement'] = {
                'percentage': float(mae_improvement),
                'absolute': float(without_gcps['mae'] - with_gcps['mae']),
                'with_gcps': with_gcps['mae'],
                'without_gcps': without_gcps['mae']
            }
        
        # Similarity improvement (higher is better)
        if with_gcps.get('similarity') and without_gcps.get('similarity'):
            similarity_improvement = ((with_gcps['similarity'] - without_gcps['similarity']) / without_gcps['similarity']) * 100
            comparison['similarity_improvement'] = {
                'percentage': float(similarity_improvement),
                'absolute': float(with_gcps['similarity'] - without_gcps['similarity']),
                'with_gcps': with_gcps['similarity'],
                'without_gcps': without_gcps['similarity']
            }
        
        # Seamline reduction (lower is better)
        if with_gcps.get('seamline_percentage') and without_gcps.get('seamline_percentage'):
            seamline_reduction = ((without_gcps['seamline_percentage'] - with_gcps['seamline_percentage']) / without_gcps['seamline_percentage']) * 100
            comparison['seamline_reduction'] = {
                'percentage': float(seamline_reduction),
                'absolute': float(without_gcps['seamline_percentage'] - with_gcps['seamline_percentage']),
                'with_gcps': with_gcps['seamline_percentage'],
                'without_gcps': without_gcps['seamline_percentage']
            }
        
        report['comparison'] = comparison
    
    # Convert numpy types to JSON-serializable types
    report = convert_to_json_serializable(report)
    
    # Save JSON report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Saved comparison report to: {output_path}")
    
    return output_path


def generate_markdown_report(
    json_report_path: Path,
    output_path: Path
) -> Path:
    """
    Generate a human-readable Markdown report from JSON.
    
    Args:
        json_report_path: Path to JSON report
        output_path: Path to save Markdown report
        
    Returns:
        Path to saved Markdown report
    """
    with open(json_report_path, 'r') as f:
        report = json.load(f)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    md_lines = []
    
    # Header
    md_lines.append("# Orthomosaic Quality Comparison Report")
    md_lines.append("")
    md_lines.append(f"**Generated:** {report['report_metadata']['generated_at']}")
    md_lines.append(f"**Basemap Source:** {report['report_metadata']['basemap_source']}")
    md_lines.append("")
    
    # Overview
    md_lines.append("## Overview")
    md_lines.append("")
    md_lines.append("This report compares orthomosaics generated:")
    md_lines.append("- **With GCPs:** Using ground control points for georeferencing")
    md_lines.append("- **Without GCPs:** Using only image-based alignment")
    md_lines.append("")
    md_lines.append("Both orthomosaics are compared against reference basemaps to evaluate:")
    md_lines.append("- Absolute accuracy (RMSE, MAE)")
    md_lines.append("- Structural similarity")
    md_lines.append("- Seamline artifacts")
    md_lines.append("")
    
    # Overall Metrics
    md_lines.append("## Overall Quality Metrics")
    md_lines.append("")
    
    if 'overall' in report['orthomosaic_with_gcps']:
        with_gcps = report['orthomosaic_with_gcps']['overall']
        without_gcps = report['orthomosaic_without_gcps']['overall']
        
        md_lines.append("| Metric | Without GCPs | With GCPs | Improvement |")
        md_lines.append("|--------|--------------|-----------|-------------|")
        
        # RMSE
        if with_gcps.get('rmse') and without_gcps.get('rmse'):
            improvement = report['comparison'].get('rmse_improvement', {})
            pct = improvement.get('percentage', 0)
            md_lines.append(f"| RMSE | {without_gcps['rmse']:.4f} | {with_gcps['rmse']:.4f} | {pct:+.2f}% |")
        
        # MAE
        if with_gcps.get('mae') and without_gcps.get('mae'):
            improvement = report['comparison'].get('mae_improvement', {})
            pct = improvement.get('percentage', 0)
            md_lines.append(f"| MAE | {without_gcps['mae']:.4f} | {with_gcps['mae']:.4f} | {pct:+.2f}% |")
        
        # Similarity
        if with_gcps.get('similarity') and without_gcps.get('similarity'):
            improvement = report['comparison'].get('similarity_improvement', {})
            pct = improvement.get('percentage', 0)
            md_lines.append(f"| Similarity | {without_gcps['similarity']:.4f} | {with_gcps['similarity']:.4f} | {pct:+.2f}% |")
        
        # Seamlines
        if with_gcps.get('seamline_percentage') and without_gcps.get('seamline_percentage'):
            improvement = report['comparison'].get('seamline_reduction', {})
            pct = improvement.get('percentage', 0)
            md_lines.append(f"| Seamline % | {without_gcps['seamline_percentage']:.2f}% | {with_gcps['seamline_percentage']:.2f}% | {pct:+.2f}% |")
        
        md_lines.append("")
    
    # Detailed Comparison
    md_lines.append("## Detailed Comparison")
    md_lines.append("")
    
    comparison = report.get('comparison', {})
    
    if comparison.get('rmse_improvement'):
        rmse = comparison['rmse_improvement']
        md_lines.append(f"### Root Mean Square Error (RMSE)")
        md_lines.append("")
        md_lines.append(f"- **Without GCPs:** {rmse['without_gcps']:.4f}")
        md_lines.append(f"- **With GCPs:** {rmse['with_gcps']:.4f}")
        md_lines.append(f"- **Improvement:** {rmse['absolute']:.4f} ({rmse['percentage']:+.2f}%)")
        md_lines.append("")
        md_lines.append("*Lower RMSE indicates better accuracy.*")
        md_lines.append("")
    
    if comparison.get('mae_improvement'):
        mae = comparison['mae_improvement']
        md_lines.append(f"### Mean Absolute Error (MAE)")
        md_lines.append("")
        md_lines.append(f"- **Without GCPs:** {mae['without_gcps']:.4f}")
        md_lines.append(f"- **With GCPs:** {mae['with_gcps']:.4f}")
        md_lines.append(f"- **Improvement:** {mae['absolute']:.4f} ({mae['percentage']:+.2f}%)")
        md_lines.append("")
        md_lines.append("*Lower MAE indicates better accuracy.*")
        md_lines.append("")
    
    if comparison.get('similarity_improvement'):
        sim = comparison['similarity_improvement']
        md_lines.append(f"### Structural Similarity")
        md_lines.append("")
        md_lines.append(f"- **Without GCPs:** {sim['without_gcps']:.4f}")
        md_lines.append(f"- **With GCPs:** {sim['with_gcps']:.4f}")
        md_lines.append(f"- **Improvement:** {sim['absolute']:.4f} ({sim['percentage']:+.2f}%)")
        md_lines.append("")
        md_lines.append("*Higher similarity (closer to 1.0) indicates better match to reference.*")
        md_lines.append("")
    
    if comparison.get('seamline_reduction'):
        seam = comparison['seamline_reduction']
        md_lines.append(f"### Seamline Artifacts")
        md_lines.append("")
        md_lines.append(f"- **Without GCPs:** {seam['without_gcps']:.2f}% of pixels")
        md_lines.append(f"- **With GCPs:** {seam['with_gcps']:.2f}% of pixels")
        md_lines.append(f"- **Reduction:** {seam['absolute']:.2f}% ({seam['percentage']:+.2f}%)")
        md_lines.append("")
        md_lines.append("*Lower percentage indicates fewer visible seamlines.*")
        md_lines.append("")
    
    # Issues and Observations
    md_lines.append("## Issues and Observations")
    md_lines.append("")
    
    # Analyze which version is better
    improvements = []
    if comparison.get('rmse_improvement', {}).get('percentage', 0) > 0:
        improvements.append("RMSE reduction")
    if comparison.get('mae_improvement', {}).get('percentage', 0) > 0:
        improvements.append("MAE reduction")
    if comparison.get('similarity_improvement', {}).get('percentage', 0) > 0:
        improvements.append("Improved similarity")
    if comparison.get('seamline_reduction', {}).get('percentage', 0) > 0:
        improvements.append("Reduced seamlines")
    
    if improvements:
        md_lines.append("### GCP Benefits")
        md_lines.append("")
        md_lines.append("Using GCPs resulted in improvements in:")
        for imp in improvements:
            md_lines.append(f"- {imp}")
        md_lines.append("")
    else:
        md_lines.append("### Note")
        md_lines.append("")
        md_lines.append("Limited improvement observed with GCPs. This may indicate:")
        md_lines.append("- High-quality image alignment without GCPs")
        md_lines.append("- GCP accuracy or distribution issues")
        md_lines.append("- Reference basemap limitations")
        md_lines.append("")
    
    # File Paths
    md_lines.append("## File Paths")
    md_lines.append("")
    md_lines.append(f"- **Orthomosaic (with GCPs):** `{report['orthomosaic_with_gcps'].get('ortho_path', 'N/A')}`")
    md_lines.append(f"- **Orthomosaic (without GCPs):** `{report['orthomosaic_without_gcps'].get('ortho_path', 'N/A')}`")
    md_lines.append(f"- **Reference Basemap:** `{report['orthomosaic_with_gcps'].get('basemap_path', 'N/A')}`")
    md_lines.append("")
    
    # Write Markdown file
    with open(output_path, 'w') as f:
        f.write('\n'.join(md_lines))
    
    logger.info(f"Saved Markdown report to: {output_path}")
    
    return output_path

