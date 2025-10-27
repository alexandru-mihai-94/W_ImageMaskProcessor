#!/usr/bin/env python3
"""
Wrapper for Image Mask Processor - BIAFLOWS compatible
2D image thresholding and region analysis
"""

import sys
import os
import numpy as np
import cv2
import csv
from pathlib import Path

# Optional BIAFLOWS integration
try:
    from biaflows import CLASS_OBJSEG
    from biaflows.helpers import BiaflowsJob, prepare_data, upload_data, upload_metrics
    from cytomine.models import Job
    BIAFLOWS_AVAILABLE = True
except ImportError:
    BIAFLOWS_AVAILABLE = False
    print("BIAFLOWS utilities not available - running in standalone mode", file=sys.stderr)


def main(argv):
    """Main execution function."""

    # Base path for Singularity compatibility
    base_path = "{}".format(os.getenv("HOME"))

    if BIAFLOWS_AVAILABLE:
        # BIAFLOWS mode - use job context manager
        with BiaflowsJob.from_cli(argv) as nj:
            run_workflow(nj, base_path)
    else:
        # Standalone mode - parse arguments manually
        run_standalone(argv)


def run_workflow(nj, base_path):
    """
    Run workflow with BIAFLOWS integration.

    Args:
        nj: BiaflowsJob instance
        base_path: Base directory path
    """
    # Update job status
    nj.job.update(status=Job.RUNNING, progress=0, statusComment="Initializing...")

    # Problem class for BIAFLOWS
    problem_cls = CLASS_OBJSEG

    # Get parameters from BIAFLOWS job
    min_thresh = nj.parameters.min_thresh
    min_area = nj.parameters.min_area

    # Prepare data (uses BIAFLOWS helpers)
    nj.job.update(progress=1, statusComment="Preparing data...")
    in_imgs, gt_imgs, in_path, gt_path, out_path, tmp_path = prepare_data(
        problem_cls, nj, is_2d=True, **nj.flags
    )

    # Process each image
    for i, in_img in enumerate(in_imgs):
        progress = int(10 + (70 * i / len(in_imgs)))
        nj.job.update(progress=progress, statusComment=f"Processing image {i+1}/{len(in_imgs)}...")

        # Load image
        image = cv2.imread(str(in_img.filepath), cv2.IMREAD_UNCHANGED)

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply threshold
        _, mask = cv2.threshold(image, min_thresh - 1, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create labeled mask
        labeled_mask = np.zeros_like(mask, dtype=np.uint16)
        region_id = 1

        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                cv2.drawContours(labeled_mask, [contour], -1, region_id, -1)
                region_id += 1

        # Save labeled mask
        out_file = out_path / in_img.filename
        cv2.imwrite(str(out_file), labeled_mask)

    # Upload results to BIAFLOWS
    nj.job.update(progress=80, statusComment="Uploading results...")
    upload_data(problem_cls, nj, in_imgs, out_path, **nj.flags, monitor_params={
        "start": 80, "end": 90, "period": 0.1
    })

    # Compute and upload metrics
    nj.job.update(progress=90, statusComment="Computing metrics...")
    upload_metrics(problem_cls, nj, in_imgs, gt_path, out_path, tmp_path, **nj.flags)

    # Complete
    nj.job.update(status=Job.TERMINATED, progress=100,
                  statusComment=f"Completed - processed {len(in_imgs)} images")


def run_standalone(argv):
    """
    Run in standalone mode without BIAFLOWS.

    Args:
        argv: Command line arguments
    """
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Image Mask Processor")
    parser.add_argument('--input', required=True, help='Input image path')
    parser.add_argument('--output_mask', required=True, help='Output mask path')
    parser.add_argument('--output_csv', required=True, help='Output CSV path')
    parser.add_argument('--min_thresh', type=int, default=10,
                       help='Minimum threshold (0-255)')
    parser.add_argument('--min_area', type=float, default=1.0,
                       help='Minimum area in pixels')

    # BIAFLOWS params (ignored in standalone mode)
    parser.add_argument('--cytomine_host', default='')
    parser.add_argument('--cytomine_public_key', default='')
    parser.add_argument('--cytomine_private_key', default='')
    parser.add_argument('--cytomine_id_project', type=int, default=0)
    parser.add_argument('--cytomine_id_software', type=int, default=0)

    args = parser.parse_args(argv)

    print(f"Processing: {args.input}")
    print(f"Min threshold: {args.min_thresh}")
    print(f"Min area: {args.min_area}")

    # Validate input
    if not Path(args.input).exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    # Create output directories
    Path(args.output_mask).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)

    # Load image
    image = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to load image: {args.input}")

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply threshold
    _, mask = cv2.threshold(image, args.min_thresh - 1, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Analyze regions and collect statistics
    regions = []
    for contour in contours:
        # Basic measurements
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if area < args.min_area:
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

    # Save mask
    cv2.imwrite(args.output_mask, mask)

    # Save CSV
    if regions:
        with open(args.output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=regions[0].keys())
            writer.writeheader()
            writer.writerows(regions)

    print(f"✓ Mask saved: {args.output_mask}")
    print(f"✓ CSV saved: {args.output_csv}")
    print(f"✓ Regions detected: {len(regions)}")


if __name__ == "__main__":
    main(sys.argv[1:])
