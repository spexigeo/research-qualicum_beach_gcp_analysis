"""
S3 Image Downloader for Qualicum Beach GCP Analysis.

Handles downloading images from S3 based on multiple input manifest files,
with support for avoiding re-downloads if images already exist.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# AWS Account IDs (from main.py)
PROD_ACCOUNT_ID = "352378214697"
STAGING_ACCOUNT_ID = "538507596022"


def get_bucket_owner(bucket_name: str) -> str:
    """Get the owner of a bucket based on its name.
    
    Args:
        bucket_name: The name of the bucket
        
    Returns:
        The AWS account ID of the bucket owner
    """
    if "-dev" in bucket_name or "-staging" in bucket_name:
        return STAGING_ACCOUNT_ID
    else:
        return PROD_ACCOUNT_ID


def parse_manifest_file(manifest_path: Path) -> Dict:
    """Parse a manifest file to extract S3 prefix and image list.
    
    Args:
        manifest_path: Path to the manifest file
        
    Returns:
        Dictionary with 'prefix', 'bucket', 's3_prefix', and 'images' keys
    """
    with open(manifest_path, "r") as f:
        manifest_entries = json.load(f)
        
    prefix_entry = manifest_entries[0]
    prefix = prefix_entry["prefix"]
    
    # Extract bucket and S3 prefix
    bucket = prefix.replace("s3://", "").split("/")[0]
    s3_prefix = prefix.replace(f"s3://{bucket}/", "")
    
    # Get list of image filenames
    images = manifest_entries[1:]
    
    return {
        'prefix': prefix,
        'bucket': bucket,
        's3_prefix': s3_prefix,
        'images': images
    }


def download_images_from_manifest(
    manifest_path: Path,
    photos_dir: Path,
    skip_existing: bool = True
) -> Dict:
    """Download images from S3 based on a manifest file.
    
    Args:
        manifest_path: Path to the manifest file
        photos_dir: Directory to save downloaded images
        skip_existing: If True, skip images that already exist locally
        
    Returns:
        Dictionary with download statistics
    """
    photos_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse manifest
    manifest_data = parse_manifest_file(manifest_path)
    bucket = manifest_data['bucket']
    s3_prefix = manifest_data['s3_prefix']
    images = manifest_data['images']
    
    # Initialize S3 client
    s3 = boto3.client("s3")
    
    downloaded = 0
    skipped = 0
    failed = 0
    
    logger.info(f"Processing manifest: {manifest_path.name}")
    logger.info(f"  Bucket: {bucket}")
    logger.info(f"  S3 prefix: {s3_prefix}")
    logger.info(f"  Total images: {len(images)}")
    
    for image_name in images:
        local_path = photos_dir / image_name
        
        # Skip if already exists
        if skip_existing and local_path.exists():
            skipped += 1
            continue
        
        # Download from S3
        try:
            s3_key = f"{s3_prefix}{image_name}"
            s3.download_file(
                bucket,
                s3_key,
                str(local_path),
                ExtraArgs={"ExpectedBucketOwner": get_bucket_owner(bucket)},
            )
            downloaded += 1
            if downloaded % 10 == 0:
                logger.info(f"  Downloaded {downloaded}/{len(images)} images...")
        except ClientError as e:
            logger.error(f"  Failed to download {image_name}: {e}")
            failed += 1
        except Exception as e:
            logger.error(f"  Unexpected error downloading {image_name}: {e}")
            failed += 1
    
    stats = {
        'total': len(images),
        'downloaded': downloaded,
        'skipped': skipped,
        'failed': failed
    }
    
    logger.info(f"  Completed: {downloaded} downloaded, {skipped} skipped, {failed} failed")
    
    return stats


def download_all_images_from_input_dir(
    input_dir: Path,
    photos_dir: Path,
    skip_existing: bool = True
) -> Dict[str, Dict]:
    """Download images from all manifest files in the input directory.
    
    Args:
        input_dir: Directory containing manifest files (input-file_*.txt)
        photos_dir: Directory to save all downloaded images
        skip_existing: If True, skip images that already exist locally
        
    Returns:
        Dictionary mapping manifest filename to download statistics
    """
    photos_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all manifest files
    manifest_files = sorted(input_dir.glob("input-file_*.txt"))
    
    if not manifest_files:
        logger.warning(f"No manifest files found in {input_dir}")
        return {}
    
    logger.info(f"Found {len(manifest_files)} manifest files")
    
    all_stats = {}
    
    for manifest_path in manifest_files:
        stats = download_images_from_manifest(
            manifest_path,
            photos_dir,
            skip_existing=skip_existing
        )
        all_stats[manifest_path.name] = stats
    
    # Print summary
    total_images = sum(s['total'] for s in all_stats.values())
    total_downloaded = sum(s['downloaded'] for s in all_stats.values())
    total_skipped = sum(s['skipped'] for s in all_stats.values())
    total_failed = sum(s['failed'] for s in all_stats.values())
    
    logger.info("\n" + "="*60)
    logger.info("Download Summary")
    logger.info("="*60)
    logger.info(f"Total manifest files: {len(manifest_files)}")
    logger.info(f"Total images: {total_images}")
    logger.info(f"Downloaded: {total_downloaded}")
    logger.info(f"Skipped (already exist): {total_skipped}")
    logger.info(f"Failed: {total_failed}")
    logger.info("="*60)
    
    return all_stats

