# Orthomosaic Quality Comparison Report

**Generated:** 2025-12-02T15:58:48.403562
**Basemap Source:** OpenStreetMap

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
| RMSE | 9.8338 | 9.8166 | +0.18% |
| MAE | 103.6522 | 103.7207 | -0.07% |
| Similarity | 0.4186 | 0.4185 | -0.03% |
| Seamline % | 9.94% | 9.94% | -0.01% |

## Detailed Comparison

### Root Mean Square Error (RMSE)

- **Without GCPs:** 9.8338
- **With GCPs:** 9.8166
- **Improvement:** 0.0173 (+0.18%)

*Lower RMSE indicates better accuracy.*

### Mean Absolute Error (MAE)

- **Without GCPs:** 103.6522
- **With GCPs:** 103.7207
- **Improvement:** -0.0685 (-0.07%)

*Lower MAE indicates better accuracy.*

### Structural Similarity

- **Without GCPs:** 0.4186
- **With GCPs:** 0.4185
- **Improvement:** -0.0001 (-0.03%)

*Higher similarity (closer to 1.0) indicates better match to reference.*

### Seamline Artifacts

- **Without GCPs:** 9.94% of pixels
- **With GCPs:** 9.94% of pixels
- **Reduction:** -0.00% (-0.01%)

*Lower percentage indicates fewer visible seamlines.*

## Issues and Observations

### GCP Benefits

Using GCPs resulted in improvements in:
- RMSE reduction

## File Paths

- **Orthomosaic (with GCPs):** `outputs/orthomosaics/orthomosaic_with_gcps.tif`
- **Orthomosaic (without GCPs):** `outputs/orthomosaics/orthomosaic_no_gcps.tif`
- **Reference Basemap:** `outputs/qualicum_beach_basemap_osm.tif`
