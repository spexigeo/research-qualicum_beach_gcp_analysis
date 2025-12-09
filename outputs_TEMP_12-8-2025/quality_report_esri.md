# Orthomosaic Quality Comparison Report

**Generated:** 2025-12-02T15:58:33.843044
**Basemap Source:** ESRI World Imagery

## Overview

This report compares orthomosaics generated:
- **With GCPs:** Using ground control points for georeferencing
- **Without GCPs:** Using only image-based alignment

Both orthomosaics are compared against reference basemaps to evaluate:
- Absolute accuracy (RMSE, MAE)
- Structural similarity
- Seamline artifacts

## Overall Quality Metrics

| Metric | Without GCPs | With GCPs | Improvement |
|--------|--------------|-----------|-------------|
| RMSE | 10.1513 | 10.1597 | -0.08% |
| MAE | 152.9137 | 152.7436 | +0.11% |
| Similarity | 0.5715 | 0.5719 | +0.06% |
| Seamline % | 9.94% | 9.94% | -0.01% |

## Detailed Comparison

### Root Mean Square Error (RMSE)

- **Without GCPs:** 10.1513
- **With GCPs:** 10.1597
- **Improvement:** -0.0084 (-0.08%)

*Lower RMSE indicates better accuracy.*

### Mean Absolute Error (MAE)

- **Without GCPs:** 152.9137
- **With GCPs:** 152.7436
- **Improvement:** 0.1700 (+0.11%)

*Lower MAE indicates better accuracy.*

### Structural Similarity

- **Without GCPs:** 0.5715
- **With GCPs:** 0.5719
- **Improvement:** 0.0003 (+0.06%)

*Higher similarity (closer to 1.0) indicates better match to reference.*

### Seamline Artifacts

- **Without GCPs:** 9.94% of pixels
- **With GCPs:** 9.94% of pixels
- **Reduction:** -0.00% (-0.01%)

*Lower percentage indicates fewer visible seamlines.*

## Issues and Observations

### GCP Benefits

Using GCPs resulted in improvements in:
- MAE reduction
- Improved similarity

## File Paths

- **Orthomosaic (with GCPs):** `outputs/orthomosaics/orthomosaic_with_gcps.tif`
- **Orthomosaic (without GCPs):** `outputs/orthomosaics/orthomosaic_no_gcps.tif`
- **Reference Basemap:** `outputs/qualicum_beach_basemap_esri.tif`
