#!/usr/bin/env python3
"""
Wrapper for Image Mask Processor - BIOMERO compatible
2D image thresholding and region analysis

This wrapper integrates with BIOMERO/OMERO workflows using bioflows_local.
The core image processing logic is in process_mask.py.
"""

import argparse
import sys
import os
import shutil
from types import SimpleNamespace
from typing import List, Sequence, Tuple
from pathlib import Path

from bioflows_local import CLASS_SPTCNT, BiaflowsJob, prepare_data, get_discipline
from process_mask import process_image, clean_omero_filename


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
            base_name, ext = clean_omero_filename(bfimg.filename)
            mask_filename = f"{base_name}_mask{ext}"
            csv_filename = f"{base_name}_statistics.csv"

            # Process image using core processing module
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
            base_name, ext = clean_omero_filename(bimg.filename)

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
