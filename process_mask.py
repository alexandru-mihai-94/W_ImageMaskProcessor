#!/usr/bin/env python3
"""
Core image processing module for mask generation and region analysis.

This module contains the core logic for:
- Binary mask generation via intensity thresholding
- Region detection and contour analysis
- Statistical measurement extraction (area, perimeter, shape metrics)
- CSV export of region properties
"""

import os
import csv
import re
import numpy as np
import cv2


def clean_omero_filename(filename: str) -> tuple:
    """
    Clean OMERO filename and return (base_name, extension).

    OMERO appends .X.tif to filenames (e.g., image.tif.0.tif for tiled images).
    This function removes the .X.tif suffix to get the original name.

    Args:
        filename: Input filename (e.g., "image.tif.0.tif")

    Returns:
        Tuple of (base_name, extension) (e.g., ("image", ".tif"))

    Examples:
        >>> clean_omero_filename("image.tif.0.tif")
        ('image', '.tif')
        >>> clean_omero_filename("photo.png")
        ('photo', '.png')
    """
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


def process_image(image_path: str, min_thresh: int, min_area: float,
                  output_mask_path: str, output_csv_path: str) -> dict:
    """
    Process a single image: threshold, find regions, save mask and statistics.

    This function performs the complete image analysis pipeline:
    1. Load and convert image to grayscale if needed
    2. Apply binary thresholding
    3. Detect contours (connected regions)
    4. Calculate geometric and shape properties for each region
    5. Filter regions by minimum area
    6. Export binary mask and CSV statistics

    Args:
        image_path: Path to input image file
        min_thresh: Minimum threshold value (0-255). Pixels >= threshold are foreground.
        min_area: Minimum area in pixels. Regions smaller than this are filtered out.
        output_mask_path: Path to save output binary mask image
        output_csv_path: Path to save output CSV statistics file

    Returns:
        Dictionary containing processing results:
        {
            'regions_detected': int,  # Number of regions found
            'regions_kept': int,      # Number of regions after filtering
            'mask_path': str,         # Path to saved mask
            'csv_path': str          # Path to saved CSV
        }

    Raises:
        ValueError: If image cannot be loaded
        IOError: If output files cannot be written

    Statistics exported per region:
        - region_id: Unique identifier (sorted by area, largest first)
        - area_pixels: Region size in pixels
        - perimeter: Boundary length
        - centroid_x, centroid_y: Center of mass coordinates
        - bbox_x, bbox_y, bbox_width, bbox_height: Bounding box
        - aspect_ratio: Width/height ratio
        - circularity: Shape metric (4π × area / perimeter²), 1.0 = perfect circle
        - polygon: Simplified segmentation coordinates (x,y pairs)
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

    # Convert to 8-bit if needed (cv2.findContours requires 8-bit images)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Analyze regions and collect statistics
    regions = []
    total_contours = len(contours)

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
    print(f"  ✓ Regions detected: {total_contours} total, {len(regions)} kept (area >= {min_area})")

    return {
        'regions_detected': total_contours,
        'regions_kept': len(regions),
        'mask_path': output_mask_path,
        'csv_path': output_csv_path
    }


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description='Process image to generate binary mask and region statistics'
    )
    parser.add_argument('--input', required=True, help='Input image path')
    parser.add_argument('--output-mask', required=True, help='Output mask path')
    parser.add_argument('--output-csv', required=True, help='Output CSV path')
    parser.add_argument('--min-thresh', type=int, default=10,
                       help='Minimum threshold (0-255)')
    parser.add_argument('--min-area', type=float, default=1.0,
                       help='Minimum area in pixels')

    args = parser.parse_args()

    result = process_image(
        image_path=args.input,
        min_thresh=args.min_thresh,
        min_area=args.min_area,
        output_mask_path=args.output_mask,
        output_csv_path=args.output_csv
    )

    print(f"\nProcessing complete!")
    print(f"  Total regions: {result['regions_detected']}")
    print(f"  Filtered regions: {result['regions_kept']}")
