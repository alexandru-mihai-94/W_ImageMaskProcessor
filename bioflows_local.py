from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence

CLASS_SPTCNT = "LOCAL_CLASS_SPTCNT"

KNOWN_JOB_ATTRS = {
    "input_dir",
    "output_dir",
    "gt_dir",
    "temp_dir",
    "suffixes",
    "local",
    "parameters",
}

DEFAULT_SUFFIXES = (
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".npy",
)


@dataclass
class ImageResource:
    """Minimal image representation compatible with the original wrapper."""

    filename: str
    filename_original: str
    filepath: Path

    def __post_init__(self) -> None:
        self.filepath = Path(self.filepath)
        self.path = str(self.filepath)


class BiaflowsJob:
    """Local stand-in for the Cytomine/BIAFLOWS job helper."""

    def __init__(
        self,
        args: argparse.Namespace,
        *,
        parameters: Optional[SimpleNamespace] = None,
    ) -> None:
        if parameters is None:
            parameters = getattr(args, "parameters", None)
        if parameters is None:
            param_values = {
                key: value
                for key, value in vars(args).items()
                if key not in KNOWN_JOB_ATTRS
            }
            parameters = SimpleNamespace(**param_values)

        self.parameters = parameters
        self.flags = {}
        self.input_dir = Path(args.input_dir)
        self.output_dir = Path(args.output_dir)
        self.gt_dir = Path(args.gt_dir)

        temp_dir_value = getattr(args, "temp_dir", None)
        if temp_dir_value is None:
            # Mirror the behaviour of the hosted runner where a temp folder is optional.
            temp_dir_value = self.output_dir / "tmp"
        self.temp_dir = Path(temp_dir_value)
        self.suffixes = self._normalise_suffixes(args.suffixes)

    def __enter__(self) -> "BiaflowsJob":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return False

    @classmethod
    def from_cli(
        cls,
        argv: Sequence[str],
        **overrides,
    ) -> "BiaflowsJob":
        args = _parse_args(argv)
        parameters = overrides.pop(
            "parameters",
            getattr(args, "parameters", None),
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return cls(args, parameters=parameters)

    @staticmethod
    def _normalise_suffixes(
        suffixes: Optional[Sequence[str]],
    ) -> Optional[List[str]]:
        if not suffixes:
            return list(DEFAULT_SUFFIXES)
        normalised: List[str] = []
        for suffix in suffixes:
            clean = suffix.strip().lower()
            if not clean:
                continue
            if not clean.startswith("."):
                clean = f".{clean}"
            normalised.append(clean)
        return normalised or list(DEFAULT_SUFFIXES)


def prepare_data(
    discipline: str,
    job: BiaflowsJob,
    *,
    is_2d: bool = True,
    **flags,
):
    """Prepare input/output directories and enumerate available images."""

    del discipline, flags  # Not required for the local implementation
    if not is_2d:
        print("Warning: local workflow only supports 2D data; proceeding regardless.")

    job.input_dir.mkdir(parents=True, exist_ok=True)
    job.output_dir.mkdir(parents=True, exist_ok=True)
    job.temp_dir.mkdir(parents=True, exist_ok=True)

    in_imgs = _collect_images(job.input_dir, job.suffixes)
    gt_imgs = _collect_images(job.gt_dir, job.suffixes)

    return (
        in_imgs,
        gt_imgs,
        str(job.input_dir),
        str(job.gt_dir),
        str(job.output_dir),
        str(job.temp_dir),
    )


def get_discipline(job: BiaflowsJob, default: Optional[str] = None) -> Optional[str]:
    """Return the requested default discipline (placeholder for compatibility)."""

    del job
    return default


def _collect_images(directory: Path, suffixes: Optional[Sequence[str]]) -> List[ImageResource]:
    if not directory.exists():
        return []
    records: List[ImageResource] = []
    for entry in sorted(directory.iterdir()):
        if not entry.is_file():
            continue
        if suffixes and entry.suffix.lower() not in suffixes:
            continue
        records.append(
            ImageResource(
                filename=entry.name,
                filename_original=entry.name,
                filepath=entry,
            )
        )
    return records


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local runner for the Cell Expansion workflow (no Cytomine dependencies)."
    )
    parser.add_argument("--input-dir", dest="input_dir")
    parser.add_argument(
        "--infolder",
        dest="input_dir",
        help="Compatibility alias for --input-dir.",
    )
    parser.add_argument("--output-dir", dest="output_dir")
    parser.add_argument(
        "--outfolder",
        dest="output_dir",
        help="Compatibility alias for --output-dir.",
    )
    parser.add_argument("--gt-dir", dest="gt_dir")
    parser.add_argument(
        "--gtfolder",
        dest="gt_dir",
        help="Compatibility alias for --gt-dir.",
    )
    parser.add_argument("--temp-dir", dest="temp_dir")
    parser.add_argument(
        "--tmpfolder",
        dest="temp_dir",
        help="Compatibility alias for --temp-dir.",
    )
    parser.add_argument(
        "--suffix",
        dest="suffixes",
        action="append",
        help="Restrict processing to files matching this extension (can be repeated).",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Compatibility flag ignored by the local runner.",
    )
    # Use parse_known_args to ignore unknown arguments like -nmc from SLURM/OMERO
    import sys
    parsed, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"Warning: ignoring unknown arguments: {unknown}", file=sys.stderr)
    return parsed

def _parse_bool(value):
    if isinstance(value, bool):
        return value
    truthy = {"true", "1", "yes", "y", "on"}
    falsy = {"false", "0", "no", "n", "off"}
    normalised = value.strip().lower()
    if normalised in truthy:
        return True
    if normalised in falsy:
        return False
    raise argparse.ArgumentTypeError("Cannot interpret '%s' as a boolean" % value)
