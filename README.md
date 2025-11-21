# Qualicum Beach GCP Analysis

Tools for analyzing ground control points (GCPs) from Qualicum Beach survey data. This package provides functionality to:

1. Parse ground control points from KMZ files
2. Download basemaps from OpenStreetMap or Esri World Imagery
3. Visualize GCPs overlaid on basemaps

## Installation

```bash
cd qualicum_beach_gcp_analysis
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Quick Start

### Using the Python API

```python
from qualicum_beach_gcp_analysis import (
    load_gcps_from_kmz,
    download_basemap,
    visualize_gcps_on_basemap,
)
from qualicum_beach_gcp_analysis.visualization import calculate_gcp_bbox

# Load GCPs from KMZ file
kmz_path = "path/to/QualicumBeach_AOI.kmz"
gcps = load_gcps_from_kmz(kmz_path)

# Calculate bounding box
bbox = calculate_gcp_bbox(gcps, padding=0.01)

# Download basemap
basemap_path = download_basemap(
    bbox=bbox,
    output_path="basemap.tif",
    source="openstreetmap"
)

# Visualize GCPs on basemap
visualize_gcps_on_basemap(
    gcps=gcps,
    basemap_path=basemap_path,
    output_path="visualization.png",
    title="Qualicum Beach Ground Control Points"
)
```

### Using Jupyter Notebooks

1. **Local Jupyter Notebook**: Open `test_qualicum_beach_analysis.ipynb`
2. **Google Colab**: Open `test_qualicum_beach_analysis_colab.ipynb` in Google Colab

## Features

### KMZ Parser

The `kmz_parser` module can parse KMZ files containing ground control point data:

- Extracts GCP coordinates (latitude, longitude, elevation)
- Parses GCP IDs and descriptions
- Handles various KML formats and namespaces

### Basemap Downloader

The `basemap_downloader` module downloads map tiles and creates GeoTIFF files:

- Supports OpenStreetMap and Esri World Imagery
- Auto-calculates appropriate zoom levels
- Creates georeferenced GeoTIFF outputs

### Visualization

The `visualization` module creates maps with GCPs overlaid:

- Customizable point colors and sizes
- Optional GCP ID labels
- High-resolution output images

## Data Sources

This package is designed to work with ground control point data provided by the Town of Qualicum Beach. The data is typically provided in:

- **KMZ format**: `QualicumBeach_AOI.kmz` - Contains GCP coordinates and metadata
- **Geodatabase format**: `Spexi_Survey.gdb/` - ArcGIS geodatabase (not directly parsed by this package)

## Requirements

- Python 3.8+
- numpy
- pillow
- matplotlib
- rasterio
- requests

## Project Structure

```
qualicum_beach_gcp_analysis/
├── qualicum_beach_gcp_analysis/
│   ├── __init__.py
│   ├── kmz_parser.py          # KMZ file parsing
│   ├── basemap_downloader.py  # Basemap tile downloading
│   └── visualization.py       # GCP visualization
├── outputs/                    # Output directory for results
├── test_qualicum_beach_analysis.ipynb          # Local Jupyter notebook
├── test_qualicum_beach_analysis_colab.ipynb    # Google Colab notebook
├── requirements.txt
├── setup.py
└── README.md
```

## License

This project is provided as-is for analysis of Qualicum Beach ground control point data.


Tools for analyzing ground control points (GCPs) from Qualicum Beach survey data. This package provides functionality to:

1. Parse ground control points from KMZ files
2. Download basemaps from OpenStreetMap or Esri World Imagery
3. Visualize GCPs overlaid on basemaps

## Installation

```bash
cd qualicum_beach_gcp_analysis
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Quick Start

### Using the Python API

```python
from qualicum_beach_gcp_analysis import (
    load_gcps_from_kmz,
    download_basemap,
    visualize_gcps_on_basemap,
)
from qualicum_beach_gcp_analysis.visualization import calculate_gcp_bbox

# Load GCPs from KMZ file
kmz_path = "path/to/QualicumBeach_AOI.kmz"
gcps = load_gcps_from_kmz(kmz_path)

# Calculate bounding box
bbox = calculate_gcp_bbox(gcps, padding=0.01)

# Download basemap
basemap_path = download_basemap(
    bbox=bbox,
    output_path="basemap.tif",
    source="openstreetmap"
)

# Visualize GCPs on basemap
visualize_gcps_on_basemap(
    gcps=gcps,
    basemap_path=basemap_path,
    output_path="visualization.png",
    title="Qualicum Beach Ground Control Points"
)
```

### Using Jupyter Notebooks

1. **Local Jupyter Notebook**: Open `test_qualicum_beach_analysis.ipynb`
2. **Google Colab**: Open `test_qualicum_beach_analysis_colab.ipynb` in Google Colab

## Features

### KMZ Parser

The `kmz_parser` module can parse KMZ files containing ground control point data:

- Extracts GCP coordinates (latitude, longitude, elevation)
- Parses GCP IDs and descriptions
- Handles various KML formats and namespaces

### Basemap Downloader

The `basemap_downloader` module downloads map tiles and creates GeoTIFF files:

- Supports OpenStreetMap and Esri World Imagery
- Auto-calculates appropriate zoom levels
- Creates georeferenced GeoTIFF outputs

### Visualization

The `visualization` module creates maps with GCPs overlaid:

- Customizable point colors and sizes
- Optional GCP ID labels
- High-resolution output images

## Data Sources

This package is designed to work with ground control point data provided by the Town of Qualicum Beach. The data is typically provided in:

- **KMZ format**: `QualicumBeach_AOI.kmz` - Contains GCP coordinates and metadata
- **Geodatabase format**: `Spexi_Survey.gdb/` - ArcGIS geodatabase (not directly parsed by this package)

## Requirements

- Python 3.8+
- numpy
- pillow
- matplotlib
- rasterio
- requests

## Project Structure

```
qualicum_beach_gcp_analysis/
├── qualicum_beach_gcp_analysis/
│   ├── __init__.py
│   ├── kmz_parser.py          # KMZ file parsing
│   ├── basemap_downloader.py  # Basemap tile downloading
│   └── visualization.py       # GCP visualization
├── outputs/                    # Output directory for results
├── test_qualicum_beach_analysis.ipynb          # Local Jupyter notebook
├── test_qualicum_beach_analysis_colab.ipynb    # Google Colab notebook
├── requirements.txt
├── setup.py
└── README.md
```

## License

This project is provided as-is for analysis of Qualicum Beach ground control point data.


Tools for analyzing ground control points (GCPs) from Qualicum Beach survey data. This package provides functionality to:

1. Parse ground control points from KMZ files
2. Download basemaps from OpenStreetMap or Esri World Imagery
3. Visualize GCPs overlaid on basemaps

## Installation

```bash
cd qualicum_beach_gcp_analysis
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Quick Start

### Using the Python API

```python
from qualicum_beach_gcp_analysis import (
    load_gcps_from_kmz,
    download_basemap,
    visualize_gcps_on_basemap,
)
from qualicum_beach_gcp_analysis.visualization import calculate_gcp_bbox

# Load GCPs from KMZ file
kmz_path = "path/to/QualicumBeach_AOI.kmz"
gcps = load_gcps_from_kmz(kmz_path)

# Calculate bounding box
bbox = calculate_gcp_bbox(gcps, padding=0.01)

# Download basemap
basemap_path = download_basemap(
    bbox=bbox,
    output_path="basemap.tif",
    source="openstreetmap"
)

# Visualize GCPs on basemap
visualize_gcps_on_basemap(
    gcps=gcps,
    basemap_path=basemap_path,
    output_path="visualization.png",
    title="Qualicum Beach Ground Control Points"
)
```

### Using Jupyter Notebooks

1. **Local Jupyter Notebook**: Open `test_qualicum_beach_analysis.ipynb`
2. **Google Colab**: Open `test_qualicum_beach_analysis_colab.ipynb` in Google Colab

## Features

### KMZ Parser

The `kmz_parser` module can parse KMZ files containing ground control point data:

- Extracts GCP coordinates (latitude, longitude, elevation)
- Parses GCP IDs and descriptions
- Handles various KML formats and namespaces

### Basemap Downloader

The `basemap_downloader` module downloads map tiles and creates GeoTIFF files:

- Supports OpenStreetMap and Esri World Imagery
- Auto-calculates appropriate zoom levels
- Creates georeferenced GeoTIFF outputs

### Visualization

The `visualization` module creates maps with GCPs overlaid:

- Customizable point colors and sizes
- Optional GCP ID labels
- High-resolution output images

## Data Sources

This package is designed to work with ground control point data provided by the Town of Qualicum Beach. The data is typically provided in:

- **KMZ format**: `QualicumBeach_AOI.kmz` - Contains GCP coordinates and metadata
- **Geodatabase format**: `Spexi_Survey.gdb/` - ArcGIS geodatabase (not directly parsed by this package)

## Requirements

- Python 3.8+
- numpy
- pillow
- matplotlib
- rasterio
- requests

## Project Structure

```
qualicum_beach_gcp_analysis/
├── qualicum_beach_gcp_analysis/
│   ├── __init__.py
│   ├── kmz_parser.py          # KMZ file parsing
│   ├── basemap_downloader.py  # Basemap tile downloading
│   └── visualization.py       # GCP visualization
├── outputs/                    # Output directory for results
├── test_qualicum_beach_analysis.ipynb          # Local Jupyter notebook
├── test_qualicum_beach_analysis_colab.ipynb    # Google Colab notebook
├── requirements.txt
├── setup.py
└── README.md
```

## License

This project is provided as-is for analysis of Qualicum Beach ground control point data.

