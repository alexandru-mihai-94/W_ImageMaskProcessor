#!/usr/bin/env python3
"""
Wrapper for Image Mask Processor - BIOMERO compatible
2D image thresholding and region analysis
"""

import argparse
import sys
import os
import shutil
import csv
from types import SimpleNamespace
from typing import List, Sequence, Tuple
from pathlib import Path

import numpy as np
import cv2
from bioflows_local import CLASS_SPTCNT, BiaflowsJob, prepare_data, get_discipline


def _parse_bool(value) -> bool:
    """Parse boolean values from string."""
    if isinstance(value, bool):
        return value
    truthy = {"true", "1", "yes", "y", "on"}
    falsy = {"false", "0", "no", "n", "off"}
    normalised = value.strip().lower()
    if normalised in truthy:
        return True
    if normalised in falsy:
        return False
    raise argparse.ArgumentTypeError(f"Cannot interpret '{value}' as a boolean.")


def _parse_cli_args(argv: Sequence[str]) -> Tuple[argparse.Namespace, List[str]]:
    """Parse workflow-specific arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--min-thresh", "--min_thresh", dest="min_thresh", type=int)
    parser.add_argument("--min-area", "--min_area", dest="min_area", type=float)
    args, remaining = parser.parse_known_args(argv)

    # Set defaults if not provided
    if args.min_thresh is None:
        args.min_thresh = 10
    if args.min_area is None:
        args.min_area = 1.0

    return args, list(remaining)


def _clear_directory(directory: str) -> None:
    """Remove all content inside directory without deleting the directory itself."""
    if not os.path.isdir(directory):
        return
    for entry in os.scandir(directory):
        try:
            if entry.is_dir(follow_symlinks=False):
                shutil.rmtree(entry.path, ignore_errors=True)
            else:
                os.remove(entry.path)
        except OSError as exc:
            print(f"Warning: could not remove {entry.path}: {exc}")


def _clean_filename(filename: str) -> tuple[str, str]:
    """
    Clean OMERO filename and return (base_name, extension).

    OMERO appends .X.tif to filenames (e.g., image.tif.0.tif).
    This function removes the .X.tif suffix to get the original name.

    Args:
        filename: Input filename (e.g., "image.tif.0.tif")

    Returns:
        Tuple of (base_name, extension) (e.g., ("image", ".tif"))
    """
    import re

    # Check if filename matches OMERO pattern: *.ext.N.ext
    # e.g., "image.tif.0.tif" -> base="image", ext=".tif"
    match = re.match(r'^(.+?)(\.\w+)\.\d+(\.\w+)$', filename)
    if match:
        base = match.group(1)
        ext = match.group(2)
        print(f"  Cleaned OMERO filename: {filename} -> {base}{ext}")
        return base, ext

    # Standard filename: just split extension normally
    base, ext = os.path.splitext(filename)
    return base, ext


def process_image(image_path: str, min_thresh: int, min_area: float, output_mask_path: str, output_csv_path: str):
    """
    Process a single image: threshold, find regions, save mask and statistics.

    Args:
        image_path: Path to input image
        min_thresh: Minimum threshold value (0-255)
        min_area: Minimum area in pixels for region filtering
        output_mask_path: Path to save output mask
        output_csv_path: Path to save CSV statistics
    """
    print(f"Processing: {image_path}")

    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply threshold
    _, mask = cv2.threshold(image, min_thresh - 1, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Analyze regions and collect statistics
    regions = []
    for contour in contours:
        # Basic measurements
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if area < min_area:
            continue

        # Bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Centroid
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
        else:
            cx, cy = x + w/2, y + h/2

        # Shape metrics
        aspect_ratio = float(w) / h if h > 0 else 0
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

        # Simplified polygon
        epsilon = 0.005 * perimeter
        polygon = cv2.approxPolyDP(contour, epsilon, True)
        polygon_coords = [(int(pt[0][0]), int(pt[0][1])) for pt in polygon]
        polygon_str = ';'.join([f"{x},{y}" for x, y in polygon_coords])

        regions.append({
            'region_id': len(regions),
            'area_pixels': round(area, 2),
            'perimeter': round(perimeter, 2),
            'centroid_x': round(cx, 2),
            'centroid_y': round(cy, 2),
            'bbox_x': x,
            'bbox_y': y,
            'bbox_width': w,
            'bbox_height': h,
            'aspect_ratio': round(aspect_ratio, 3),
            'circularity': round(circularity, 3),
            'polygon': polygon_str
        })

    # Sort by area (largest first)
    regions.sort(key=lambda r: r['area_pixels'], reverse=True)
    for i, region in enumerate(regions):
        region['region_id'] = i

    # Save mask (uncompressed TIFF for OMERO compatibility)
    cv2.imwrite(output_mask_path, mask, [cv2.IMWRITE_TIFF_COMPRESSION, 1])

    # Save CSV
    if regions:
        with open(output_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=regions[0].keys())
            writer.writeheader()
            writer.writerows(regions)

    print(f"  ✓ Mask saved: {output_mask_path}")
    print(f"  ✓ CSV saved: {output_csv_path}")
    print(f"  ✓ Regions detected: {len(regions)}")


def main(argv):
    """Main execution function."""
    overrides, remaining = _parse_cli_args(argv)
    parameters = SimpleNamespace(
        min_thresh=int(overrides.min_thresh),
        min_area=float(overrides.min_area),
    )

    with BiaflowsJob.from_cli(remaining, parameters=parameters) as bj:
        min_thresh = parameters.min_thresh
        min_area = parameters.min_area

        print("Initializing...")

        # 1. Prepare data for workflow
        in_imgs, gt_imgs, in_path, gt_path, out_path, tmp_path = prepare_data(
            get_discipline(bj, default=CLASS_SPTCNT), bj, is_2d=True, **bj.flags
        )

        # Create temporary directory for this run
        tmp_path = os.path.join(tmp_path, "mask_processor_tmp")
        os.makedirs(tmp_path, exist_ok=True)

        print(f"Parameters: Min threshold: {min_thresh} | Min area: {min_area}")

        # 2. Run image analysis workflow
        print("Launching workflow...")

        for bfimg in in_imgs:
            print(f"Processing: {bfimg.__dict__}")

            # Read input image
            fn = os.path.join(in_path, bfimg.filename)

            # Generate output filenames (clean OMERO naming convention)
            base_name, ext = _clean_filename(bfimg.filename)
            mask_filename = f"{base_name}_mask{ext}"
            csv_filename = f"{base_name}_statistics.csv"

            # Process image
            process_image(
                image_path=fn,
                min_thresh=min_thresh,
                min_area=min_area,
                output_mask_path=os.path.join(tmp_path, mask_filename),
                output_csv_path=os.path.join(tmp_path, csv_filename)
            )

        # 3. Copy results to output folder
        print("Copying results to output folder...")
        for bimg in in_imgs:
            # Clean OMERO filename
            base_name, ext = _clean_filename(bimg.filename)

            # Copy mask
            mask_filename = f"{base_name}_mask{ext}"
            src_mask = os.path.join(tmp_path, mask_filename)
            if os.path.exists(src_mask):
                shutil.copy(src_mask, out_path)
                print(f"  Copied mask to {out_path}/{mask_filename}")

            # Copy CSV
            csv_filename = f"{base_name}_statistics.csv"
            src_csv = os.path.join(tmp_path, csv_filename)
            if os.path.exists(src_csv):
                shutil.copy(src_csv, out_path)
                print(f"  Copied CSV to {out_path}/{csv_filename}")

        # 4. Cleanup temporary directory
        _clear_directory(tmp_path)
        print("Finished.")


if __name__ == "__main__":
    main(sys.argv[1:])
