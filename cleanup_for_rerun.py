#!/usr/bin/env python3
"""
Script to clean up intermediate output files while preserving MetaShape project files.

This allows rerunning the notebook with new GCP accuracy settings while reusing
existing photo matching and camera alignment from MetaShape.
"""

from pathlib import Path
import shutil

output_dir = Path("outputs")

print("=" * 70)
print("Cleaning up intermediate files for rerun with new GCP accuracy")
print("=" * 70)
print()

# Files/directories to delete
to_delete = [
    # Exported orthomosaics
    output_dir / "orthomosaics" / "orthomosaic_no_gcps.tif",
    output_dir / "orthomosaics" / "orthomosaic_with_gcps.tif",
    
    # All comparison results
    output_dir / "comparisons",
    
    # All visualizations
    output_dir / "visualizations",
    
    # All reports
    output_dir / "quality_report_esri.json",
    output_dir / "quality_report_esri.md",
    output_dir / "quality_report_osm.json",
    output_dir / "quality_report_osm.md",
    output_dir / "quality_report_final.tex",
    output_dir / "quality_report_final.pdf",
    
    # GCP export files (will be regenerated with new accuracy)
    output_dir / "gcps_metashape.xml",
    output_dir / "gcps_metashape.csv",
]

deleted_count = 0
for item in to_delete:
    if item.exists():
        if item.is_file():
            item.unlink()
            print(f"✓ Deleted file: {item}")
            deleted_count += 1
        elif item.is_dir():
            shutil.rmtree(item)
            print(f"✓ Deleted directory: {item}")
            deleted_count += 1

print()
print(f"Deleted {deleted_count} items")
print()

# Files to keep
print("=" * 70)
print("Files KEPT (MetaShape project files):")
print("=" * 70)

to_keep = [
    output_dir / "intermediate" / "orthomosaic_no_gcps.psx",
    output_dir / "intermediate" / "orthomosaic_no_gcps.psx.files",
    output_dir / "intermediate" / "orthomosaic_with_gcps.psx",
    output_dir / "intermediate" / "orthomosaic_with_gcps.psx.files",
]

for item in to_keep:
    if item.exists():
        print(f"  ✓ {item}")
    else:
        print(f"  ✗ {item} (not found)")

print()
print("=" * 70)
print("IMPORTANT:")
print("=" * 70)
print("If GCPs were already loaded in the MetaShape project with old accuracy,")
print("you may need to delete the .psx files to force reloading with new accuracy:")
print()
print("  rm -rf outputs/intermediate/orthomosaic_with_gcps.psx")
print("  rm -rf outputs/intermediate/orthomosaic_with_gcps.psx.files/")
print()
print("This will force MetaShape to reload GCPs with the new 0.05m accuracy setting.")
print("=" * 70)
