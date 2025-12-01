"""
LaTeX Report Generator for Orthomosaic Quality Analysis.

Generates professional PDF reports with visualizations comparing orthomosaics
with and without GCPs.
"""

import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import logging
import subprocess
import shutil

logger = logging.getLogger(__name__)


def generate_latex_report(
    json_report_path: Path,
    output_path: Path,
    visualization_dir: Optional[Path] = None,
    json_report_path_osm: Optional[Path] = None,
    visualization_dir_osm: Optional[Path] = None
) -> Path:
    """
    Generate a comprehensive LaTeX report from JSON metrics and visualizations.
    Can include both ESRI and OSM basemap comparisons.
    
    Args:
        json_report_path: Path to primary JSON report (typically ESRI)
        output_path: Path to save LaTeX file (and PDF if pdflatex available)
        visualization_dir: Directory containing visualization images for primary report
        json_report_path_osm: Optional path to OSM JSON report
        visualization_dir_osm: Optional directory containing OSM visualization images
        
    Returns:
        Path to generated LaTeX file (or PDF if compilation successful)
    """
    with open(json_report_path, 'r') as f:
        report_primary = json.load(f)
    
    report_osm = None
    if json_report_path_osm:
        with open(json_report_path_osm, 'r') as f:
            report_osm = json.load(f)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine visualization directories
    if visualization_dir is None:
        visualization_dir = output_path.parent / "visualizations" / "esri"
    visualization_dir = Path(visualization_dir)
    
    if visualization_dir_osm is None and report_osm:
        visualization_dir_osm = output_path.parent / "visualizations" / "osm"
    if visualization_dir_osm:
        visualization_dir_osm = Path(visualization_dir_osm)
    
    # Generate LaTeX content
    latex_path = output_path.with_suffix('.tex')
    latex_content = generate_latex_content(
        report_primary, visualization_dir, latex_path,
        report_osm=report_osm, visualization_dir_osm=visualization_dir_osm
    )
    
    # Write LaTeX file
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    logger.info(f"Generated LaTeX report: {latex_path}")
    
    # Try to compile to PDF
    pdf_path = output_path.with_suffix('.pdf')
    if compile_latex_to_pdf(latex_path, pdf_path):
        logger.info(f"Successfully compiled PDF: {pdf_path}")
        return pdf_path
    else:
        logger.info("PDF compilation not available. LaTeX file saved for manual compilation.")
        return latex_path


def generate_latex_content(
    report: Dict, 
    visualization_dir: Path, 
    latex_path: Path,
    report_osm: Optional[Dict] = None,
    visualization_dir_osm: Optional[Path] = None
) -> str:
    """Generate comprehensive LaTeX document content with ESRI and optionally OSM comparisons."""
    
    metadata = report.get('report_metadata', {})
    basemap_source_primary = metadata.get('basemap_source', 'ESRI World Imagery')
    generated_at = metadata.get('generated_at', datetime.now().isoformat())
    
    metrics_with = report.get('orthomosaic_with_gcps', {})
    metrics_without = report.get('orthomosaic_without_gcps', {})
    comparison = report.get('comparison', {})
    
    overall_with = metrics_with.get('overall', {})
    overall_without = metrics_without.get('overall', {})
    
    # Process OSM report if available
    comparison_osm = None
    overall_with_osm = None
    overall_without_osm = None
    if report_osm:
        metrics_with_osm = report_osm.get('orthomosaic_with_gcps', {})
        metrics_without_osm = report_osm.get('orthomosaic_without_gcps', {})
        comparison_osm = report_osm.get('comparison', {})
        overall_with_osm = metrics_with_osm.get('overall', {})
        overall_without_osm = metrics_without_osm.get('overall', {})
    
    # Build basemap sources text
    basemap_sources_text = basemap_source_primary
    if report_osm:
        basemap_sources_text += " and OpenStreetMap"
    
    # LaTeX document
    latex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{geometry}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{float}
\usepackage{caption}
\usepackage{subcaption}

\geometry{margin=2.5cm}
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    filecolor=blue,
    urlcolor=blue,
    citecolor=blue
}

\title{Orthomosaic Quality Comparison Report}
\author{Qualicum Beach GCP Analysis}
\date{""" + generated_at + r"""}

\begin{document}

\maketitle

\begin{abstract}
This comprehensive report compares orthomosaics generated with and without ground control points (GCPs)
against reference basemaps from """ + basemap_sources_text + r""". The analysis evaluates
absolute accuracy (RMSE, MAE), structural similarity, seamline artifacts, and 2D spatial
errors to assess the impact of GCPs on orthomosaic quality. The report provides detailed comparisons
for each basemap and concludes with recommendations on which method (with or without GCPs) provides
better results based on comprehensive analysis across all metrics.
\end{abstract}

\section{Methodology}

\subsection{Comparison Approach}
The orthomosaics are compared to reference basemaps using the following methodology:

\begin{enumerate}
\item \textbf{Reprojection}: The orthomosaic is reprojected to match the reference basemap's
      coordinate reference system (CRS) and spatial extent using bilinear resampling.
      
\item \textbf{Pixel-level Metrics}: 
\begin{itemize}
    \item \textbf{RMSE} (Root Mean Square Error): Measures overall pixel intensity differences
    \item \textbf{MAE} (Mean Absolute Error): Measures average absolute pixel differences
    \item \textbf{Structural Similarity}: Correlation-based measure of structural similarity
\end{itemize}

\item \textbf{Feature Matching}: Feature-based matching (SIFT, ORB, or phase correlation)
      is used to compute 2D spatial errors, providing X and Y offset measurements in pixels.
      This identifies systematic shifts or misalignments between the orthomosaic and reference.

\item \textbf{Seamline Detection}: Gradient-based analysis detects potential seamline artifacts
      by identifying high-gradient regions that may indicate stitching errors or discontinuities.
\end{enumerate}

\subsection{Reference Basemaps}
The orthomosaics are compared against multiple reference basemaps to ensure robust assessment:
\begin{itemize}
    \item \textbf{""" + basemap_source_primary + r"""}: High-resolution satellite imagery providing a detailed baseline
"""
    if report_osm:
        latex += r"""    \item \textbf{OpenStreetMap}: Community-sourced mapping data providing an alternative reference
"""
    latex += r"""\end{itemize}
Using multiple basemaps helps validate findings and account for potential biases in any single reference source.

"""
    
    # Add metrics tables
    latex += r"""\section{Quality Metrics}

\subsection{Overall Metrics Comparison - """ + basemap_source_primary + r"""}

\begin{table}[H]
\centering
\caption{Overall Quality Metrics Comparison (""" + basemap_source_primary + r""")}
\begin{tabular}{lccc}
\toprule
Metric & Without GCPs & With GCPs & Improvement \\
\midrule
"""
    
    # RMSE
    if overall_without.get('rmse') and overall_with.get('rmse'):
        rmse_imp = comparison.get('rmse_improvement', {})
        pct = rmse_imp.get('percentage', 0)
        latex += f"RMSE & {overall_without['rmse']:.4f} & {overall_with['rmse']:.4f} & {pct:+.2f}\\% \\\\\n"
    
    # MAE
    if overall_without.get('mae') and overall_with.get('mae'):
        mae_imp = comparison.get('mae_improvement', {})
        pct = mae_imp.get('percentage', 0)
        latex += f"MAE & {overall_without['mae']:.4f} & {overall_with['mae']:.4f} & {pct:+.2f}\\% \\\\\n"
    
    # Similarity
    if overall_without.get('similarity') and overall_with.get('similarity'):
        sim_imp = comparison.get('similarity_improvement', {})
        pct = sim_imp.get('percentage', 0)
        latex += f"Similarity & {overall_without['similarity']:.4f} & {overall_with['similarity']:.4f} & {pct:+.2f}\\% \\\\\n"
    
    # Seamlines
    if overall_without.get('seamline_percentage') and overall_with.get('seamline_percentage'):
        seam_imp = comparison.get('seamline_reduction', {})
        pct = seam_imp.get('percentage', 0)
        latex += f"Seamlines (\\%) & {overall_without['seamline_percentage']:.2f} & {overall_with['seamline_percentage']:.2f} & {pct:+.2f}\\% \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}

"""
    
    # Add 2D error metrics if available
    errors_2d_with = overall_with.get('errors_2d', {})
    errors_2d_without = overall_without.get('errors_2d', {})
    
    if errors_2d_with.get('rmse_2d_pixels') or errors_2d_without.get('rmse_2d_pixels'):
        latex += r"""\subsection{2D Spatial Error Metrics}

Feature matching provides spatial error measurements in pixels:

\begin{table}[H]
\centering
\caption{2D Spatial Error from Feature Matching}
\begin{tabular}{lcc}
\toprule
Metric & Without GCPs & With GCPs \\
\midrule
"""
        if errors_2d_without.get('mean_offset_x_pixels') is not None:
            latex += f"Mean X Offset (px) & {errors_2d_without.get('mean_offset_x_pixels', 0):.2f} & {errors_2d_with.get('mean_offset_x_pixels', 0):.2f} \\\\\n"
        if errors_2d_without.get('mean_offset_y_pixels') is not None:
            latex += f"Mean Y Offset (px) & {errors_2d_without.get('mean_offset_y_pixels', 0):.2f} & {errors_2d_with.get('mean_offset_y_pixels', 0):.2f} \\\\\n"
        if errors_2d_without.get('rmse_2d_pixels') is not None:
            latex += f"2D RMSE (px) & {errors_2d_without.get('rmse_2d_pixels', 0):.2f} & {errors_2d_with.get('rmse_2d_pixels', 0):.2f} \\\\\n"
        if errors_2d_without.get('num_matches', 0) > 0:
            latex += f"Feature Matches & {errors_2d_without.get('num_matches', 0)} & {errors_2d_with.get('num_matches', 0)} \\\\\n"
        
        latex += r"""\bottomrule
\end{tabular}
\end{table}

"""
    
    # Add OSM metrics section if available
    if report_osm and overall_with_osm and overall_without_osm:
        latex += r"""\subsection{Overall Metrics Comparison - OpenStreetMap}

\begin{table}[H]
\centering
\caption{Overall Quality Metrics Comparison (OpenStreetMap)}
\begin{tabular}{lccc}
\toprule
Metric & Without GCPs & With GCPs & Improvement \\
\midrule
"""
        # RMSE
        if overall_without_osm.get('rmse') and overall_with_osm.get('rmse'):
            rmse_imp_osm = comparison_osm.get('rmse_improvement', {})
            pct = rmse_imp_osm.get('percentage', 0)
            latex += f"RMSE & {overall_without_osm['rmse']:.4f} & {overall_with_osm['rmse']:.4f} & {pct:+.2f}\\% \\\\\n"
        
        # MAE
        if overall_without_osm.get('mae') and overall_with_osm.get('mae'):
            mae_imp_osm = comparison_osm.get('mae_improvement', {})
            pct = mae_imp_osm.get('percentage', 0)
            latex += f"MAE & {overall_without_osm['mae']:.4f} & {overall_with_osm['mae']:.4f} & {pct:+.2f}\\% \\\\\n"
        
        # Similarity
        if overall_without_osm.get('similarity') and overall_with_osm.get('similarity'):
            sim_imp_osm = comparison_osm.get('similarity_improvement', {})
            pct = sim_imp_osm.get('percentage', 0)
            latex += f"Similarity & {overall_without_osm['similarity']:.4f} & {overall_with_osm['similarity']:.4f} & {pct:+.2f}\\% \\\\\n"
        
        # Seamlines
        if overall_without_osm.get('seamline_percentage') and overall_with_osm.get('seamline_percentage'):
            seam_imp_osm = comparison_osm.get('seamline_reduction', {})
            pct = seam_imp_osm.get('percentage', 0)
            latex += f"Seamlines (\\%) & {overall_without_osm['seamline_percentage']:.2f} & {overall_with_osm['seamline_percentage']:.2f} & {pct:+.2f}\\% \\\\\n"
        
        latex += r"""\bottomrule
\end{tabular}
\end{table}

"""
        
        # OSM 2D error metrics
        errors_2d_with_osm = overall_with_osm.get('errors_2d', {})
        errors_2d_without_osm = overall_without_osm.get('errors_2d', {})
        
        if errors_2d_with_osm.get('rmse_2d_pixels') or errors_2d_without_osm.get('rmse_2d_pixels'):
            latex += r"""\subsection{2D Spatial Error Metrics - OpenStreetMap}

Feature matching provides spatial error measurements in pixels:

\begin{table}[H]
\centering
\caption{2D Spatial Error from Feature Matching (OpenStreetMap)}
\begin{tabular}{lcc}
\toprule
Metric & Without GCPs & With GCPs \\
\midrule
"""
            if errors_2d_without_osm.get('mean_offset_x_pixels') is not None:
                latex += f"Mean X Offset (px) & {errors_2d_without_osm.get('mean_offset_x_pixels', 0):.2f} & {errors_2d_with_osm.get('mean_offset_x_pixels', 0):.2f} \\\\\n"
            if errors_2d_without_osm.get('mean_offset_y_pixels') is not None:
                latex += f"Mean Y Offset (px) & {errors_2d_without_osm.get('mean_offset_y_pixels', 0):.2f} & {errors_2d_with_osm.get('mean_offset_y_pixels', 0):.2f} \\\\\n"
            if errors_2d_without_osm.get('rmse_2d_pixels') is not None:
                latex += f"2D RMSE (px) & {errors_2d_without_osm.get('rmse_2d_pixels', 0):.2f} & {errors_2d_with_osm.get('rmse_2d_pixels', 0):.2f} \\\\\n"
            if errors_2d_without_osm.get('num_matches', 0) > 0:
                latex += f"Feature Matches & {errors_2d_without_osm.get('num_matches', 0)} & {errors_2d_with_osm.get('num_matches', 0)} \\\\\n"
            
            latex += r"""\bottomrule
\end{tabular}
\end{table}

"""
    
    # Add visualizations if available
    latex += r"""\section{Visual Comparisons}

"""
    
    # Check for visualization files
    vis_files = {
        'comparison': visualization_dir / 'comparison_side_by_side.png',
        'metrics': visualization_dir / 'metrics_summary.png',
        'seamlines_no_gcps': visualization_dir / 'seamlines_no_gcps.png',
        'seamlines_with_gcps': visualization_dir / 'seamlines_with_gcps.png',
        'error_no_gcps': visualization_dir / 'error_no_gcps.png',
        'error_with_gcps': visualization_dir / 'error_with_gcps.png',
    }
    
    # Side-by-side comparison
    if vis_files['comparison'].exists():
        rel_path = vis_files['comparison'].relative_to(latex_path.parent)
        latex += r"""\subsection{Side-by-Side Comparison}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{""" + str(rel_path) + r"""}
\caption{Comparison of orthomosaics with and without GCPs against the reference basemap.
The improvement map (bottom right) shows where GCPs reduce errors (green) or increase them (red).}
\label{fig:comparison}
\end{figure}

"""
    
    # Metrics summary
    if vis_files['metrics'].exists():
        rel_path = vis_files['metrics'].relative_to(latex_path.parent)
        latex += r"""\subsection{Quality Metrics Summary}

\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{""" + str(rel_path) + r"""}
\caption{Bar chart comparing quality metrics between orthomosaics with and without GCPs.}
\label{fig:metrics}
\end{figure}

"""
    
    # Seamline comparisons
    if vis_files['seamlines_no_gcps'].exists() and vis_files['seamlines_with_gcps'].exists():
        rel_path_no = vis_files['seamlines_no_gcps'].relative_to(latex_path.parent)
        rel_path_with = vis_files['seamlines_with_gcps'].relative_to(latex_path.parent)
        latex += r"""\subsection{Seamline Detection}

\begin{figure}[H]
\centering
\begin{subfigure}{0.48\textwidth}
\centering
\includegraphics[width=\textwidth]{""" + str(rel_path_no) + r"""}
\caption{Without GCPs}
\end{subfigure}
\hfill
\begin{subfigure}{0.48\textwidth}
\centering
\includegraphics[width=\textwidth]{""" + str(rel_path_with) + r"""}
\caption{With GCPs}
\end{subfigure}
\caption{Seamline detection showing potential stitching artifacts. Red regions indicate
high-gradient areas that may represent seamlines or discontinuities.}
\label{fig:seamlines}
\end{figure}

"""
    
    # Error visualizations
    if vis_files['error_no_gcps'].exists() and vis_files['error_with_gcps'].exists():
        rel_path_no = vis_files['error_no_gcps'].relative_to(latex_path.parent)
        rel_path_with = vis_files['error_with_gcps'].relative_to(latex_path.parent)
        latex += r"""\subsection{Error Maps - """ + basemap_source_primary + r"""}

\begin{figure}[H]
\centering
\begin{subfigure}{0.48\textwidth}
\centering
\includegraphics[width=\textwidth]{""" + str(rel_path_no) + r"""}
\caption{Without GCPs}
\end{subfigure}
\hfill
\begin{subfigure}{0.48\textwidth}
\centering
\includegraphics[width=\textwidth]{""" + str(rel_path_with) + r"""}
\caption{With GCPs}
\end{subfigure}
\caption{Error maps showing absolute differences between orthomosaics and """ + basemap_source_primary + r""" basemap.
Hotter colors indicate larger errors.}
\label{fig:errors}
\end{figure}

"""
    
    # Add OSM visualizations if available
    if report_osm and visualization_dir_osm:
        vis_files_osm = {
            'comparison': visualization_dir_osm / 'comparison_side_by_side.png',
            'metrics': visualization_dir_osm / 'metrics_summary.png',
            'seamlines_no_gcps': visualization_dir_osm / 'seamlines_no_gcps.png',
            'seamlines_with_gcps': visualization_dir_osm / 'seamlines_with_gcps.png',
            'error_no_gcps': visualization_dir_osm / 'error_no_gcps.png',
            'error_with_gcps': visualization_dir_osm / 'error_with_gcps.png',
        }
        
        latex += r"""\subsection{Visual Comparisons - OpenStreetMap}

"""
        
        # OSM Side-by-side comparison
        if vis_files_osm['comparison'].exists():
            rel_path = vis_files_osm['comparison'].relative_to(latex_path.parent)
            latex += r"""\subsubsection{Side-by-Side Comparison}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{""" + str(rel_path) + r"""}
\caption{Comparison of orthomosaics with and without GCPs against the OpenStreetMap basemap.
The improvement map (bottom right) shows where GCPs reduce errors (green) or increase them (red).}
\label{fig:comparison_osm}
\end{figure}

"""
        
        # OSM Metrics summary
        if vis_files_osm['metrics'].exists():
            rel_path = vis_files_osm['metrics'].relative_to(latex_path.parent)
            latex += r"""\subsubsection{Quality Metrics Summary}

\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{""" + str(rel_path) + r"""}
\caption{Bar chart comparing quality metrics between orthomosaics with and without GCPs (OpenStreetMap).}
\label{fig:metrics_osm}
\end{figure}

"""
        
        # OSM Seamline comparisons
        if vis_files_osm['seamlines_no_gcps'].exists() and vis_files_osm['seamlines_with_gcps'].exists():
            rel_path_no = vis_files_osm['seamlines_no_gcps'].relative_to(latex_path.parent)
            rel_path_with = vis_files_osm['seamlines_with_gcps'].relative_to(latex_path.parent)
            latex += r"""\subsubsection{Seamline Detection}

\begin{figure}[H]
\centering
\begin{subfigure}{0.48\textwidth}
\centering
\includegraphics[width=\textwidth]{""" + str(rel_path_no) + r"""}
\caption{Without GCPs}
\end{subfigure}
\hfill
\begin{subfigure}{0.48\textwidth}
\centering
\includegraphics[width=\textwidth]{""" + str(rel_path_with) + r"""}
\caption{With GCPs}
\end{subfigure}
\caption{Seamline detection showing potential stitching artifacts (OpenStreetMap comparison).
Red regions indicate high-gradient areas that may represent seamlines or discontinuities.}
\label{fig:seamlines_osm}
\end{figure}

"""
        
        # OSM Error visualizations
        if vis_files_osm['error_no_gcps'].exists() and vis_files_osm['error_with_gcps'].exists():
            rel_path_no = vis_files_osm['error_no_gcps'].relative_to(latex_path.parent)
            rel_path_with = vis_files_osm['error_with_gcps'].relative_to(latex_path.parent)
            latex += r"""\subsubsection{Error Maps}

\begin{figure}[H]
\centering
\begin{subfigure}{0.48\textwidth}
\centering
\includegraphics[width=\textwidth]{""" + str(rel_path_no) + r"""}
\caption{Without GCPs}
\end{subfigure}
\hfill
\begin{subfigure}{0.48\textwidth}
\centering
\includegraphics[width=\textwidth]{""" + str(rel_path_with) + r"""}
\caption{With GCPs}
\end{subfigure}
\caption{Error maps showing absolute differences between orthomosaics and OpenStreetMap basemap.
Hotter colors indicate larger errors.}
\label{fig:errors_osm}
\end{figure}

"""
    
    # Detailed analysis
    latex += r"""\section{Detailed Analysis}

"""
    
    # RMSE analysis
    if comparison.get('rmse_improvement'):
        rmse = comparison['rmse_improvement']
        latex += "\\subsection{Root Mean Square Error (RMSE)}\n\n"
        latex += "The RMSE measures the overall pixel intensity differences between the orthomosaic and reference.\n"
        latex += f"Without GCPs: {rmse['without_gcps']:.4f}, With GCPs: {rmse['with_gcps']:.4f}.\n"
        if rmse['percentage'] > 0:
            latex += f"This represents a {rmse['percentage']:.2f}\\% improvement, indicating that GCPs reduce overall error.\n\n"
        else:
            latex += f"This represents a {abs(rmse['percentage']):.2f}\\% increase in error.\n\n"
    
    # MAE analysis
    if comparison.get('mae_improvement'):
        mae = comparison['mae_improvement']
        latex += "\\subsection{Mean Absolute Error (MAE)}\n\n"
        latex += "The MAE measures the average absolute pixel differences.\n"
        latex += f"Without GCPs: {mae['without_gcps']:.4f}, With GCPs: {mae['with_gcps']:.4f}.\n"
        if mae['percentage'] > 0:
            latex += f"This represents a {mae['percentage']:.2f}\\% improvement.\n\n"
        else:
            latex += f"This represents a {abs(mae['percentage']):.2f}\\% increase.\n\n"
    
    # Similarity analysis
    if comparison.get('similarity_improvement'):
        sim = comparison['similarity_improvement']
        latex += "\\subsection{Structural Similarity}\n\n"
        latex += "The similarity metric measures how well the orthomosaic structure matches the reference.\n"
        latex += f"Without GCPs: {sim['without_gcps']:.4f}, With GCPs: {sim['with_gcps']:.4f}.\n"
        if sim['percentage'] > 0:
            latex += f"This represents a {sim['percentage']:.2f}\\% improvement in structural similarity.\n\n"
        else:
            latex += f"This represents a {abs(sim['percentage']):.2f}\\% decrease.\n\n"
    
    # Seamline analysis
    if comparison.get('seamline_reduction'):
        seam = comparison['seamline_reduction']
        latex += "\\subsection{Seamline Artifacts}\n\n"
        latex += "Seamline artifacts are detected by analyzing gradient magnitudes.\n"
        latex += f"Without GCPs: {seam['without_gcps']:.2f}\\% of pixels flagged, With GCPs: {seam['with_gcps']:.2f}\\%.\n"
        if seam['percentage'] > 0:
            latex += f"This represents a {seam['percentage']:.2f}\\% reduction in detected seamlines.\n\n"
        else:
            latex += f"This represents a {abs(seam['percentage']):.2f}\\% increase.\n\n"
    
    # 2D error analysis
    if errors_2d_with.get('rmse_2d_pixels'):
        latex += "\\subsection{2D Spatial Error}\n\n"
        latex += "Feature matching provides spatial error measurements indicating systematic shifts or misalignments.\n"
        if errors_2d_without.get('rmse_2d_pixels'):
            latex += f"Without GCPs: {errors_2d_without['rmse_2d_pixels']:.2f} pixels RMSE. "
        if errors_2d_with.get('rmse_2d_pixels'):
            latex += f"With GCPs: {errors_2d_with['rmse_2d_pixels']:.2f} pixels RMSE.\n\n"
    
    # Comprehensive Recommendations
    latex += r"""\section{Comprehensive Analysis and Recommendations}

"""
    
    # Analyze improvements across both basemaps
    improvements_esri = {
        'rmse': comparison.get('rmse_improvement', {}).get('percentage', 0),
        'mae': comparison.get('mae_improvement', {}).get('percentage', 0),
        'similarity': comparison.get('similarity_improvement', {}).get('percentage', 0),
        'seamline': comparison.get('seamline_reduction', {}).get('percentage', 0)
    }
    
    improvements_osm = {}
    if report_osm and comparison_osm:
        improvements_osm = {
            'rmse': comparison_osm.get('rmse_improvement', {}).get('percentage', 0),
            'mae': comparison_osm.get('mae_improvement', {}).get('percentage', 0),
            'similarity': comparison_osm.get('similarity_improvement', {}).get('percentage', 0),
            'seamline': comparison_osm.get('seamline_reduction', {}).get('percentage', 0)
        }
    
    # Count positive improvements
    positive_esri = sum(1 for v in improvements_esri.values() if v > 0)
    positive_osm = sum(1 for v in improvements_osm.values() if v > 0) if improvements_osm else 0
    
    # Calculate average improvement
    avg_improvement_esri = sum(improvements_esri.values()) / len(improvements_esri) if improvements_esri else 0
    avg_improvement_osm = sum(improvements_osm.values()) / len(improvements_osm) if improvements_osm else 0
    
    latex += r"""\subsection{Summary of Results Across Basemaps}

The analysis compared orthomosaics with and without GCPs against multiple reference basemaps to ensure robust conclusions.

"""
    
    # ESRI summary
    latex += r"""\subsubsection{Results Against """ + basemap_source_primary + r"""}

"""
    if positive_esri >= 2:
        latex += f"GCPs showed improvement in {positive_esri} out of 4 key metrics, with an average improvement of {avg_improvement_esri:.2f}\\%.\n\n"
    else:
        latex += f"Limited improvement observed with GCPs against {basemap_source_primary}. Only {positive_esri} out of 4 metrics showed improvement.\n\n"
    
    # OSM summary if available
    if report_osm and improvements_osm:
        latex += r"""\subsubsection{Results Against OpenStreetMap}

"""
        if positive_osm >= 2:
            latex += f"GCPs showed improvement in {positive_osm} out of 4 key metrics, with an average improvement of {avg_improvement_osm:.2f}\\%.\n\n"
        else:
            latex += f"Limited improvement observed with GCPs against OpenStreetMap. Only {positive_osm} out of 4 metrics showed improvement.\n\n"
    
    # Cross-basemap consistency
    if report_osm and improvements_osm:
        consistent_improvements = []
        consistent_degradations = []
        
        for metric in ['rmse', 'mae', 'similarity', 'seamline']:
            esri_val = improvements_esri.get(metric, 0)
            osm_val = improvements_osm.get(metric, 0)
            if esri_val > 0 and osm_val > 0:
                consistent_improvements.append(metric.upper())
            elif esri_val < 0 and osm_val < 0:
                consistent_degradations.append(metric.upper())
        
        latex += r"""\subsubsection{Cross-Basemap Consistency}

"""
        if consistent_improvements:
            latex += f"Consistent improvements across both basemaps were observed for: {', '.join(consistent_improvements)}.\n\n"
        if consistent_degradations:
            latex += f"Consistent degradations across both basemaps were observed for: {', '.join(consistent_degradations)}.\n\n"
        if not consistent_improvements and not consistent_degradations:
            latex += "Results varied between basemaps, suggesting that basemap-specific characteristics may influence the comparison.\n\n"
    
    # Final recommendation
    latex += r"""\subsection{Final Recommendation}

Based on comprehensive analysis across all metrics and reference basemaps:

"""
    
    # Determine overall recommendation
    total_positive = positive_esri + positive_osm
    total_metrics = 4 + (4 if report_osm else 0)
    
    if total_positive >= total_metrics * 0.5:  # At least 50% of metrics show improvement
        latex += r"""\textbf{Recommendation: Use GCPs for orthomosaic generation.}

The analysis demonstrates that incorporating ground control points provides measurable improvements across multiple quality metrics. The benefits include:
\begin{itemize}
    \item Improved absolute accuracy (lower RMSE and MAE)
    \item Better structural similarity to reference basemaps
    \item Reduced seamline artifacts
    \item More consistent georeferencing
\end{itemize}

These improvements justify the additional effort required to collect and incorporate GCPs into the processing workflow.

"""
    elif total_positive >= total_metrics * 0.25:  # At least 25% show improvement
        latex += r"""\textbf{Recommendation: GCPs provide marginal benefits; consider based on project requirements.}

The analysis shows mixed results, with GCPs providing some improvements but not consistently across all metrics. Consider using GCPs if:
\begin{itemize}
    \item High absolute accuracy is critical for the project
    \item GCPs are readily available and accurately surveyed
    \item The additional processing time is acceptable
\end{itemize}

However, if processing speed is prioritized and the image alignment quality is already high, processing without GCPs may be sufficient.

"""
    else:  # Less than 25% show improvement
        latex += r"""\textbf{Recommendation: Processing without GCPs is sufficient for this dataset.}

The analysis indicates that the image-based alignment achieves high quality without requiring ground control points. This suggests:
\begin{itemize}
    \item Excellent image overlap and feature matching
    \item High-quality camera calibration
    \item Sufficient image coverage for robust bundle adjustment
\end{itemize}

In this case, the additional effort to collect and incorporate GCPs may not provide sufficient benefit to justify the cost and time investment.

"""
    
    # Additional considerations
    latex += r"""\subsection{Additional Considerations}

When making the final decision, also consider:
\begin{itemize}
    \item \textbf{Project Requirements}: What level of accuracy is required for the intended application?
    \item \textbf{GCP Quality}: Are the available GCPs accurately surveyed and well-distributed?
    \item \textbf{Processing Time}: Can the project accommodate the additional processing time for GCP incorporation?
    \item \textbf{Cost-Benefit}: Does the improvement justify the cost of GCP collection and processing?
\end{itemize}

"""
    
    latex += r"""\end{document}
"""
    
    return latex


def compile_latex_to_pdf(latex_path: Path, pdf_path: Path) -> bool:
    """
    Attempt to compile LaTeX to PDF using pdflatex.
    
    Args:
        latex_path: Path to LaTeX file
        pdf_path: Path for output PDF
        
    Returns:
        True if compilation successful, False otherwise
    """
    # Check if pdflatex is available
    if not shutil.which('pdflatex'):
        logger.warning("pdflatex not found. Install LaTeX (e.g., MacTeX, TeX Live) to compile PDFs.")
        return False
    
    try:
        # Run pdflatex (twice for references)
        work_dir = latex_path.parent
        
        for run in [1, 2]:
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', '-output-directory', str(work_dir), str(latex_path)],
                capture_output=True,
                text=True,
                cwd=work_dir
            )
            
            if result.returncode != 0:
                logger.warning(f"pdflatex run {run} failed: {result.stderr}")
                return False
        
        # Check if PDF was created
        expected_pdf = latex_path.with_suffix('.pdf')
        if expected_pdf.exists():
            # Move to desired location if different
            if expected_pdf != pdf_path:
                expected_pdf.rename(pdf_path)
            return True
        else:
            logger.warning("PDF file not created after compilation")
            return False
            
    except Exception as e:
        logger.warning(f"PDF compilation failed: {e}")
        return False

