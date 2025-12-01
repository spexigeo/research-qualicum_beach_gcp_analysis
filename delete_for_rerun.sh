#!/bin/bash
# Script to delete intermediate output files while keeping MetaShape project files

OUTPUT_DIR="outputs"

echo "Files to DELETE (will be regenerated with new GCP accuracy):"
echo "============================================================"

# Delete exported orthomosaics (will be re-exported with new GCP accuracy)
echo "1. Orthomosaics:"
rm -v "$OUTPUT_DIR/orthomosaics/"*.tif 2>/dev/null || echo "  (none found)"

# Delete all comparison metrics and visualizations
echo ""
echo "2. Comparison metrics and visualizations:"
rm -rfv "$OUTPUT_DIR/comparisons/" 2>/dev/null || echo "  (none found)"
rm -rfv "$OUTPUT_DIR/visualizations/" 2>/dev/null || echo "  (none found)"

# Delete reports
echo ""
echo "3. Reports:"
rm -v "$OUTPUT_DIR/quality_report_"*.json 2>/dev/null || echo "  (none found)"
rm -v "$OUTPUT_DIR/quality_report_"*.md 2>/dev/null || echo "  (none found)"
rm -v "$OUTPUT_DIR/quality_report_"*.tex 2>/dev/null || echo "  (none found)"
rm -v "$OUTPUT_DIR/quality_report_"*.pdf 2>/dev/null || echo "  (none found)"

# Delete GCP export files (will be regenerated with new accuracy)
echo ""
echo "4. GCP export files (will be regenerated with new accuracy):"
rm -v "$OUTPUT_DIR/gcps_metashape.xml" 2>/dev/null || echo "  (none found)"
rm -v "$OUTPUT_DIR/gcps_metashape.csv" 2>/dev/null || echo "  (none found)"

echo ""
echo "============================================================"
echo "Files to KEEP (MetaShape project files with photo matching):"
echo "============================================================"
echo "  - $OUTPUT_DIR/intermediate/orthomosaic_no_gcps.psx"
echo "  - $OUTPUT_DIR/intermediate/orthomosaic_no_gcps.psx.files/"
echo "  - $OUTPUT_DIR/intermediate/orthomosaic_with_gcps.psx"
echo "  - $OUTPUT_DIR/intermediate/orthomosaic_with_gcps.psx.files/"
echo ""
echo "NOTE: If GCPs were already loaded in the project with old accuracy,"
echo "      you may need to delete the .psx files to reload with new accuracy."
echo ""
echo "============================================================"
