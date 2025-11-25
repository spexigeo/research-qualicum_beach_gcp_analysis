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
    visualization_dir: Optional[Path] = None
) -> Path:
    """
    Generate a LaTeX report from JSON metrics and visualizations.
    
    Args:
        json_report_path: Path to JSON report
        output_path: Path to save LaTeX file (and PDF if pdflatex available)
        visualization_dir: Directory containing visualization images
        
    Returns:
        Path to generated LaTeX file (or PDF if compilation successful)
    """
    with open(json_report_path, 'r') as f:
        report = json.load(f)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine visualization directory
    if visualization_dir is None:
        visualization_dir = output_path.parent / "visualizations"
    visualization_dir = Path(visualization_dir)
    
    # Generate LaTeX content
    latex_content = generate_latex_content(report, visualization_dir)
    
    # Write LaTeX file
    latex_path = output_path.with_suffix('.tex')
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


def generate_latex_content(report: Dict, visualization_dir: Path) -> str:
    """Generate LaTeX document content."""
    
    metadata = report.get('report_metadata', {})
    basemap_source = metadata.get('basemap_source', 'Unknown')
    generated_at = metadata.get('generated_at', datetime.now().isoformat())
    
    metrics_with = report.get('orthomosaic_with_gcps', {})
    metrics_without = report.get('orthomosaic_without_gcps', {})
    comparison = report.get('comparison', {})
    
    overall_with = metrics_with.get('overall', {})
    overall_without = metrics_without.get('overall', {})
    
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
This report compares orthomosaics generated with and without ground control points (GCPs)
against reference basemaps from """ + basemap_source + r""". The analysis evaluates
absolute accuracy (RMSE, MAE), structural similarity, seamline artifacts, and 2D spatial
errors to assess the impact of GCPs on orthomosaic quality.
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

\subsection{Reference Basemap}
The reference basemap used for comparison is from """ + basemap_source + r""", which provides
a high-quality georeferenced imagery baseline for accuracy assessment.

"""
    
    # Add metrics tables
    latex += r"""\section{Quality Metrics}

\subsection{Overall Metrics Comparison}

\begin{table}[H]
\centering
\caption{Overall Quality Metrics Comparison}
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
        latex += r"""\subsection{Side-by-Side Comparison}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{""" + str(vis_files['comparison'].relative_to(output_path.parent)) + r"""}
\caption{Comparison of orthomosaics with and without GCPs against the reference basemap.
The improvement map (bottom right) shows where GCPs reduce errors (green) or increase them (red).}
\label{fig:comparison}
\end{figure}

"""
    
    # Metrics summary
    if vis_files['metrics'].exists():
        latex += r"""\subsection{Quality Metrics Summary}

\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{""" + str(vis_files['metrics'].relative_to(output_path.parent)) + r"""}
\caption{Bar chart comparing quality metrics between orthomosaics with and without GCPs.}
\label{fig:metrics}
\end{figure}

"""
    
    # Seamline comparisons
    if vis_files['seamlines_no_gcps'].exists() and vis_files['seamlines_with_gcps'].exists():
        latex += r"""\subsection{Seamline Detection}

\begin{figure}[H]
\centering
\begin{subfigure}{0.48\textwidth}
\centering
\includegraphics[width=\textwidth]{""" + str(vis_files['seamlines_no_gcps'].relative_to(output_path.parent)) + r"""}
\caption{Without GCPs}
\end{subfigure}
\hfill
\begin{subfigure}{0.48\textwidth}
\centering
\includegraphics[width=\textwidth]{""" + str(vis_files['seamlines_with_gcps'].relative_to(output_path.parent)) + r"""}
\caption{With GCPs}
\end{subfigure}
\caption{Seamline detection showing potential stitching artifacts. Red regions indicate
high-gradient areas that may represent seamlines or discontinuities.}
\label{fig:seamlines}
\end{figure}

"""
    
    # Error visualizations
    if vis_files['error_no_gcps'].exists() and vis_files['error_with_gcps'].exists():
        latex += r"""\subsection{Error Maps}

\begin{figure}[H]
\centering
\begin{subfigure}{0.48\textwidth}
\centering
\includegraphics[width=\textwidth]{""" + str(vis_files['error_no_gcps'].relative_to(output_path.parent)) + r"""}
\caption{Without GCPs}
\end{subfigure}
\hfill
\begin{subfigure}{0.48\textwidth}
\centering
\includegraphics[width=\textwidth]{""" + str(vis_files['error_with_gcps'].relative_to(output_path.parent)) + r"""}
\caption{With GCPs}
\end{subfigure}
\caption{Error maps showing absolute differences between orthomosaics and reference basemap.
Hotter colors indicate larger errors.}
\label{fig:errors}
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
    
    # Conclusions
    latex += r"""\section{Conclusions}

"""
    
    improvements = []
    if comparison.get('rmse_improvement', {}).get('percentage', 0) > 0:
        improvements.append("RMSE reduction")
    if comparison.get('mae_improvement', {}).get('percentage', 0) > 0:
        improvements.append("MAE reduction")
    if comparison.get('similarity_improvement', {}).get('percentage', 0) > 0:
        improvements.append("Improved structural similarity")
    if comparison.get('seamline_reduction', {}).get('percentage', 0) > 0:
        improvements.append("Reduced seamline artifacts")
    
    if improvements:
        latex += "Using GCPs resulted in improvements in: " + ", ".join(improvements) + ".\n\n"
    else:
        latex += """Limited improvement observed with GCPs. This may indicate:
\begin{itemize}
    \item High-quality image alignment without GCPs
    \item GCP accuracy or distribution issues
    \item Reference basemap limitations
    \item Insufficient GCP coverage for the area
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

