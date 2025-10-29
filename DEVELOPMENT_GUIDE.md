# BIOMERO Workflow Development Guide
## Building a Custom Image Analysis Workflow for OMERO

This guide documents the complete process of developing a BIOMERO-compatible workflow from scratch, using the W_ImageMaskProcessor as a reference implementation.

---

## Table of Contents

1. [Overview](#overview)
2. [Core Python Functionality](#1-core-python-functionality)
3. [BIOMERO Integration with bioflows_local](#2-biomero-integration-with-bioflows_local)
4. [Docker Containerization](#3-docker-containerization)
5. [OMERO/BIOMERO Deployment](#4-omeromero-deployment)
6. [Testing and Validation](#5-testing-and-validation)
7. [Best Practices](#6-best-practices)

---

## Overview

### Key Components

1. **Core Processing Module** (`process_mask.py`): Pure Python image analysis logic
2. **Workflow Wrapper** (`wrapper.py`): BIOMERO/BIAFLOWS integration layer
3. **BIAFLOWS Helper** (`bioflows_local.py`): Standalone BIAFLOWS interface (no Cytomine)
4. **Descriptor** (`descriptor.json`): Workflow metadata and parameters
5. **Container** (`Dockerfile`): Reproducible execution environment

---

## 1. Core Python Functionality

### Philosophy

Separate your core image processing logic from workflow integration. This makes code:
- **Testable**: Can be tested independently
- **Reusable**: Can be used in other projects
- **Maintainable**: Changes to integration don't affect core logic

### Implementation: `process_mask.py`

This module contains all image analysis logic with no OMERO/BIOMERO dependencies.

#### Key Functions

**1. Image Processing Pipeline**

```python
def process_image(image_path: str, min_thresh: int, min_area: float,
                  output_mask_path: str, output_csv_path: str) -> dict:
    """
    Complete image analysis pipeline:
    1. Load image
    2. Convert to grayscale if needed
    3. Apply binary threshold
    4. Find contours (connected regions)
    5. Calculate properties for each region
    6. Filter by minimum area
    7. Export mask and CSV statistics
    """
```

**Key Implementation Details:**

```python
# Load image (supports 8-bit and 16-bit)
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

# Convert to grayscale if color
if len(image.shape) == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply threshold
_, mask = cv2.threshold(image, min_thresh - 1, 255, cv2.THRESH_BINARY)

# Convert to 8-bit (cv2.findContours requires uint8)
if mask.dtype != np.uint8:
    mask = mask.astype(np.uint8)

# Find contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

**2. OMERO Filename Handling**

OMERO appends `.X.tif` to tiled images (e.g., `image.tif.0.tif`). Clean this:

```python
def clean_omero_filename(filename: str) -> tuple:
    """
    Remove OMERO's .X.tif suffix to get original name.

    Examples:
        'image.tif.0.tif' -> ('image', '.tif')
        'photo.png' -> ('photo', '.png')
    """
    # Match pattern: *.ext.N.ext
    match = re.match(r'^(.+?)(\.\w+)\.\d+(\.\w+)$', filename)
    if match:
        return match.group(1), match.group(2)

    # Standard filename
    return os.path.splitext(filename)
```

**3. Statistics Extraction**

Calculate geometric and shape properties:

```python
# Basic measurements
area = cv2.contourArea(contour)
perimeter = cv2.arcLength(contour, True)

# Bounding box
x, y, w, h = cv2.boundingRect(contour)

# Centroid
M = cv2.moments(contour)
cx = M['m10'] / M['m00']
cy = M['m01'] / M['m00']

# Shape metrics
aspect_ratio = float(w) / h
circularity = 4 * np.pi * area / (perimeter * perimeter)
```

**4. Export Results**

Save mask as uncompressed TIFF (OMERO compatibility):

```python
# Uncompressed TIFF for OMERO compatibility
cv2.imwrite(output_mask_path, mask, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
```

Export statistics as CSV:

```python
with open(output_csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=regions[0].keys())
    writer.writeheader()
    writer.writerows(regions)
```

### Standalone Usage

The module can be used independently:

```bash
python process_mask.py \
    --input image.tif \
    --output-mask mask.tif \
    --output-csv stats.csv \
    --min-thresh 10 \
    --min-area 1.0
```

---

## 2. BIOMERO Integration with bioflows_local

### The Problem with Standard BIAFLOWS

BIAFLOWS (BioImage Analysis Workflows) typically requires:
- Cytomine server connection
- Authentication credentials
- Complex server-side dependencies
- Platform-specific APIs

This is **too heavy** for simple OMERO integration.

### Solution: bioflows_local.py

A **standalone BIAFLOWS-compatible interface** that:
- ✅ Provides BIAFLOWS API without Cytomine
- ✅ Handles input/output directories
- ✅ Works offline/locally
- ✅ Compatible with BIOMERO's SLURM integration

### Key Components

**1. BiaflowsJob Class**

```python
class BiaflowsJob:
    """
    Manages workflow execution environment.
    Handles input/output directories and parameter passing.
    """

    @classmethod
    def from_cli(cls, argv, parameters=None):
        """
        Create job from command-line arguments.

        Expected arguments from BIOMERO:
        --infolder /path/to/input
        --outfolder /path/to/output
        --gtfolder /path/to/groundtruth
        --tmpfolder /path/to/temp
        """
```

**2. prepare_data Function**

```python
def prepare_data(problem_cls, nj, is_2d=True, **kwargs):
    """
    Enumerate input images and create necessary directories.

    Returns:
        - in_imgs: List of BiaflowsImage objects
        - gt_imgs: Ground truth images (optional)
        - in_path: Input directory path
        - gt_path: Ground truth directory path
        - out_path: Output directory path
        - tmp_path: Temporary directory path
    """
```

**3. Argument Parsing**

Handles SLURM wrapper arguments gracefully:

```python
def _parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--infolder", dest="input_dir")
    parser.add_argument("--outfolder", dest="output_dir")
    parser.add_argument("--gtfolder", dest="gt_dir")
    parser.add_argument("--tmpfolder", dest="temp_dir")

    # Use parse_known_args to ignore SLURM flags like -nmc
    parsed, unknown = parser.parse_known_args(argv)

    if unknown:
        print(f"Warning: ignoring unknown arguments: {unknown}")

    return parsed
```

### Integration in wrapper.py

**1. Import bioflows_local**

```python
from bioflows_local import CLASS_SPTCNT, BiaflowsJob, prepare_data, get_discipline
```

**2. Parse Workflow Parameters**

```python
def _parse_cli_args(argv):
    """Parse workflow-specific parameters."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--min-thresh", "--min_thresh", dest="min_thresh", type=int)
    parser.add_argument("--min-area", "--min_area", dest="min_area", type=float)
    args, remaining = parser.parse_known_args(argv)
    return args, remaining
```

**3. Create BiaflowsJob Context**

```python
def main(argv):
    overrides, remaining = _parse_cli_args(argv)
    parameters = SimpleNamespace(
        min_thresh=int(overrides.min_thresh),
        min_area=float(overrides.min_area),
    )

    with BiaflowsJob.from_cli(remaining, parameters=parameters) as bj:
        # Prepare directories
        in_imgs, gt_imgs, in_path, gt_path, out_path, tmp_path = prepare_data(
            get_discipline(bj, default=CLASS_SPTCNT),
            bj,
            is_2d=True,
            **bj.flags
        )

        # Process images
        for bfimg in in_imgs:
            process_image(...)
```

**4. Process Images and Copy Results**

```python
# Process each image
for bfimg in in_imgs:
    fn = os.path.join(in_path, bfimg.filename)
    base_name, ext = clean_omero_filename(bfimg.filename)

    process_image(
        image_path=fn,
        output_mask_path=os.path.join(tmp_path, f"{base_name}_mask{ext}"),
        output_csv_path=os.path.join(tmp_path, f"{base_name}_statistics.csv"),
        ...
    )

# Copy results to output directory
for bimg in in_imgs:
    base_name, ext = clean_omero_filename(bimg.filename)
    shutil.copy(
        os.path.join(tmp_path, f"{base_name}_mask{ext}"),
        out_path
    )
```

### Why This Works

1. **No Cytomine dependencies**: Pure Python, only standard libraries
2. **SLURM-compatible**: Handles SLURM wrapper arguments gracefully
3. **Directory-based**: Works with BIOMERO's file-based workflow
4. **BIAFLOWS-compatible**: Maintains API compatibility for future upgrades

---

## 3. Docker Containerization

### Dockerfile Structure

```dockerfile
# Base image with Python 3.10
FROM python:3.10.15-slim-bookworm

# Metadata
LABEL description="2D image thresholding and region analysis"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        numpy==2.2.6 \
        opencv-python==4.12.0.88

# Copy application files
ADD descriptor.json /app/descriptor.json
ADD bioflows_local.py /app/bioflows_local.py
ADD process_mask.py /app/process_mask.py
ADD wrapper.py /app/wrapper.py

# Set entrypoint
ENTRYPOINT ["python3", "/app/wrapper.py"]
```

### Key Design Decisions

**1. Slim Base Image**

Use `python:3.10.15-slim-bookworm` instead of full Python image:
- ✅ Smaller image size (~150MB vs ~1GB)
- ✅ Faster build and deployment
- ✅ Includes essential system libraries

**2. System Dependencies**

OpenCV requires system libraries:
```dockerfile
libgl1        # OpenGL library for image display operations
libglib2.0-0  # GLib library for OpenCV core functions
```

**3. Pinned Versions**

Always pin dependency versions for reproducibility:
```dockerfile
numpy==2.2.6            # Specific version
opencv-python==4.12.0.88  # Specific version
```

**4. Layer Optimization**

Order operations from least to most frequently changing:
1. Base image
2. System packages (rarely change)
3. Python packages (occasionally change)
4. Application files (frequently change)

This optimizes Docker layer caching.

**5. Single Entrypoint**

```dockerfile
ENTRYPOINT ["python3", "/app/wrapper.py"]
```

BIOMERO passes all arguments after the entrypoint.

### .dockerignore

Exclude unnecessary files from build context:

```
# Python cache
__pycache__/
*.pyc
venv/

# Data files (never include in container)
*.tif
*.png
*.jpg
*.csv
*.json
!descriptor.json  # Exception: include descriptor.json

# Development files
.git/
.gitignore
README.md
```

### Building Multi-Platform Images

Use GitHub Actions for automated multi-platform builds:

```yaml
name: Build and Push Docker Image

on:
  push:
    tags:
      - 'v*'

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to DockerHub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          platforms: linux/amd64,linux/arm64
          push: true
          tags: |
            alexandrumihai2020/w-imagemaskprocessor:${{ github.ref_name }}
            alexandrumihai2020/w-imagemaskprocessor:latest
```

This builds for:
- `linux/amd64`: Standard x86_64 servers
- `linux/arm64`: ARM-based systems

### Testing Container Locally

```bash
# Build
docker build -t w-imagemaskprocessor:test .

# Test with sample data
docker run --rm \
  -v $(pwd)/test_data:/data \
  w-imagemaskprocessor:test \
  --infolder /data/in \
  --outfolder /data/out \
  --min-thresh 10 \
  --min-area 1.0
```

---

## 4. OMERO/BIOMERO Deployment

### descriptor.json

The workflow descriptor tells OMERO how to run your workflow:

```json
{
  "name": "ImageMaskProcessor",
  "description": "2D image thresholding and region analysis",
  "container-image": {
    "image": "alexandrumihai2020/w-imagemaskprocessor",
    "type": "singularity"
  },
  "command-line": "python wrapper.py --min-thresh @min_thresh --min-area @min_area",
  "inputs": [
    {
      "id": "min_thresh",
      "value-key": "@min_thresh",
      "command-line-flag": "--min-thresh",
      "name": "Minimum threshold",
      "description": "Minimum intensity threshold (0-255)",
      "default-value": 10,
      "set-by-server": false,
      "optional": true,
      "type": "Number"
    },
    {
      "id": "min_area",
      "value-key": "@min_area",
      "command-line-flag": "--min-area",
      "name": "Minimum area",
      "description": "Minimum area in pixels for region filtering",
      "default-value": 1.0,
      "set-by-server": false,
      "optional": true,
      "type": "Number"
    }
  ],
  "schema-version": "cytomine-0.1"
}
```

### Key Descriptor Elements

**1. Container Image**

```json
"container-image": {
  "image": "alexandrumihai2020/w-imagemaskprocessor",
  "type": "singularity"
}
```

- **Image name**: Your DockerHub repository
- **Type**: BIOMERO converts Docker to Singularity
- **No tag**: BIOMERO uses version-specific tags (e.g., `v1.0.0`)

**2. Command Line**

```json
"command-line": "python wrapper.py --min-thresh @min_thresh --min-area @min_area"
```

- Use `@parameter_id` as placeholders
- BIOMERO replaces these with actual values

**3. Input Parameters**

```json
{
  "id": "min_thresh",              // Unique identifier
  "value-key": "@min_thresh",      // Placeholder in command-line
  "command-line-flag": "--min-thresh",  // Actual CLI flag
  "name": "Minimum threshold",     // Display name in GUI
  "description": "...",            // Help text
  "default-value": 10,             // Default value
  "set-by-server": false,          // User sets this (not server)
  "optional": true,                // Can be omitted
  "type": "Number"                 // Data type
}
```

**Parameter Types:**
- `Number`: Integer or float
- `String`: Text
- `Boolean`: true/false

**Important: No Cytomine Parameters**

❌ **Old BIAFLOWS format** (don't use):
```json
{
  "id": "cytomine_host",
  "set-by-server": true,
  "optional": false
}
```

✅ **BIOMERO format** (use this):
```json
{
  "id": "min_thresh",
  "set-by-server": false,
  "optional": true
}
```

### GitHub Repository Setup

**1. Repository Structure**

```
username/W_WorkflowName/
├── wrapper.py
├── process_mask.py
├── bioflows_local.py
├── descriptor.json
├── Dockerfile
├── .dockerignore
├── .gitignore
├── README.md
└── .github/
    └── workflows/
        └── docker-image.yml
```

**2. Version Tagging**

Use semantic versioning:
```bash
git tag v1.0.0 -m "First production release"
git push origin v1.0.0
```

GitHub Actions automatically builds and pushes:
```
alexandrumihai2020/w-imagemaskprocessor:v1.0.0
alexandrumihai2020/w-imagemaskprocessor:latest
```

**3. GitHub URL Format**

BIOMERO pulls descriptor from:
```
https://github.com/username/W_WorkflowName/tree/v1.0.0
https://github.com/username/W_WorkflowName/raw/v1.0.0/descriptor.json
```

### SLURM Server Setup

**1. Convert Docker to Singularity**

On the SLURM server:
```bash
# Pull from DockerHub
singularity pull docker://alexandrumihai2020/w-imagemaskprocessor:v1.0.0

# Move to workflows directory
mkdir -p /home/slurm/my-scratch/singularity_images/workflows/minimal_python_thresh
mv w-imagemaskprocessor_v1.0.0.sif /home/slurm/my-scratch/singularity_images/workflows/minimal_python_thresh/
```

**2. BIOMERO Configuration**

Update BIOMERO configuration to recognize the workflow:
```python
WORKFLOWS = {
    'minimal_python_thresh': {
        'name': 'ImageMaskProcessor',
        'url': 'https://github.com/alexandru-mihai-94/W_ImageMaskProcessor/tree/{version}',
        'versions': ['v1.0.0', 'v0.27', 'v0.26']
    }
}
```

### OMERO Web Interface

Once deployed, users see:

**1. Workflow Selection**
- Workflow: "ImageMaskProcessor"
- Version dropdown: v1.0.0, v0.27, v0.26

**2. Parameter Input**
- Minimum threshold: [slider: 0-255, default: 10]
- Minimum area: [number input, default: 1.0]

**3. Output Options**
- ☑ Add as new images in NEW dataset
- ☑ Upload result CSVs as OMERO tables
- ☑ Attach as zip to project

**4. Execution Flow**

```
1. User selects images in OMERO
2. User selects workflow and parameters
3. OMERO exports images to SLURM
4. BIOMERO converts ZARR to TIFF
5. SLURM submits job:
   singularity run w-imagemaskprocessor_v1.0.0.sif \
     --infolder /data/in \
     --outfolder /data/out \
     --min-thresh 10 \
     --min-area 1.0
6. Workflow processes images
7. BIOMERO imports results back to OMERO:
   - Mask images → New dataset
   - CSV statistics → OMERO tables
   - Logs → Attached to project
```

---

## 5. Testing and Validation

### Local Testing

**1. Test Core Processing**

```bash
python process_mask.py \
  --input test_image.tif \
  --output-mask mask.tif \
  --output-csv stats.csv \
  --min-thresh 10 \
  --min-area 1.0
```

**2. Test Wrapper (Simulated BIOMERO)**

```bash
mkdir -p test_data/in test_data/out

python wrapper.py \
  --infolder test_data/in \
  --outfolder test_data/out \
  --min-thresh 10 \
  --min-area 1.0
```

**3. Test Docker Container**

```bash
docker run --rm \
  -v $(pwd)/test_data:/data \
  alexandrumihai2020/w-imagemaskprocessor:v1.0.0 \
  --infolder /data/in \
  --outfolder /data/out \
  --min-thresh 10 \
  --min-area 1.0
```

### OMERO Testing

**1. Test SLURM Job Submission**

From OMERO interface:
- Select single test image
- Run workflow with default parameters
- Check SLURM job log for errors

**2. Verify Outputs**

Expected outputs:
```
✅ Job completes successfully
✅ Files in output directory:
   - image_name_mask.tif (binary mask)
   - image_name_statistics.csv (region properties)
✅ CSV imported as OMERO table
✅ Mask image visible in new dataset
```

**3. Check for Common Issues**

❌ **Filename issues**: Look for `.0` in filenames (should be removed)
❌ **Compression errors**: "requires imagecodecs package" (use uncompressed TIFF)
❌ **Type errors**: "CV_16UC1 not supported" (convert to uint8 after threshold)
❌ **Import failures**: Check OMERO logs for silent failures

### Debugging

**1. Enable Verbose Logging**

Add debug prints:
```python
print(f"DEBUG: Processing {filename}")
print(f"DEBUG: Image shape: {image.shape}, dtype: {image.dtype}")
print(f"DEBUG: Found {len(contours)} contours")
```

**2. Check SLURM Logs**

```bash
# On SLURM server
cat /tmp/omero-{job_id}.log
```

**3. Check OMERO Server Logs**

```bash
# On OMERO server
tail -f /opt/omero/server/OMERO.server/var/log/biomero.log
```

**4. Manual Singularity Test**

```bash
# On SLURM server
singularity run \
  /path/to/w-imagemaskprocessor_v1.0.0.sif \
  --infolder /test/in \
  --outfolder /test/out \
  --min-thresh 10 \
  --min-area 1.0
```

---

## 6. Best Practices

### Code Organization

✅ **DO:**
- Separate core logic from workflow integration
- Use descriptive function and variable names
- Add comprehensive docstrings
- Handle errors gracefully
- Print progress messages

❌ **DON'T:**
- Mix OMERO-specific code with core algorithms
- Use hardcoded paths or parameters
- Assume specific file formats without checking
- Fail silently without error messages

### Parameter Design

✅ **DO:**
- Provide sensible defaults
- Use clear, descriptive parameter names
- Add helpful descriptions
- Validate parameter ranges

❌ **DON'T:**
- Require too many parameters
- Use technical jargon in parameter names
- Omit parameter descriptions
- Allow invalid parameter combinations

### File Handling

✅ **DO:**
- Clean OMERO filenames (remove `.X.tif`)
- Use uncompressed TIFF for masks
- Handle both 8-bit and 16-bit images
- Verify files exist before processing

❌ **DON'T:**
- Assume specific filename patterns
- Use compressed formats (LZW, ZIP)
- Only support one bit depth
- Crash on missing files

### Docker Best Practices

✅ **DO:**
- Use slim base images
- Pin dependency versions
- Optimize layer caching
- Build multi-platform images
- Use .dockerignore

❌ **DON'T:**
- Use `latest` tags for dependencies
- Include test data in images
- Install unnecessary packages
- Build only for one platform

### Version Control

✅ **DO:**
- Use semantic versioning (v1.0.0)
- Tag releases in git
- Update CHANGELOG.md
- Keep descriptor.json in sync with code

❌ **DON'T:**
- Use arbitrary version numbers
- Deploy untagged versions
- Skip documentation updates
- Change API without version bump

### BIOMERO Integration

✅ **DO:**
- Use bioflows_local.py (no Cytomine)
- Handle unknown arguments gracefully
- Support standard BIAFLOWS directory structure
- Return results to output folder

❌ **DON'T:**
- Require Cytomine server connection
- Crash on SLURM wrapper arguments
- Use non-standard directory structure
- Modify input files

### Testing

✅ **DO:**
- Test locally before deploying
- Test with real OMERO data
- Test edge cases (empty images, large images)
- Verify output formats

❌ **DON'T:**
- Deploy untested code
- Test only with synthetic data
- Skip edge case testing
- Assume outputs are correct without verification

---

## Summary

### Development Checklist

- [ ] **Core processing module** (`process_mask.py`)
  - [ ] Pure Python, no OMERO dependencies
  - [ ] Comprehensive docstrings
  - [ ] Can run standalone
  - [ ] Returns structured results

- [ ] **Workflow wrapper** (`wrapper.py`)
  - [ ] Imports bioflows_local
  - [ ] Parses workflow parameters
  - [ ] Calls core processing
  - [ ] Handles file copying

- [ ] **BIAFLOWS helper** (`bioflows_local.py`)
  - [ ] Provides BIAFLOWS API
  - [ ] No Cytomine dependencies
  - [ ] Handles unknown arguments

- [ ] **Descriptor** (`descriptor.json`)
  - [ ] Correct container image name
  - [ ] No Cytomine parameters
  - [ ] Clear parameter descriptions
  - [ ] Sensible defaults

- [ ] **Dockerfile**
  - [ ] Slim base image
  - [ ] Pinned dependencies
  - [ ] All required files included
  - [ ] Single entrypoint

- [ ] **GitHub Actions**
  - [ ] Builds on version tags
  - [ ] Multi-platform support
  - [ ] Pushes to DockerHub

- [ ] **Documentation**
  - [ ] README.md with usage examples
  - [ ] Parameter descriptions
  - [ ] Output format specification

- [ ] **Testing**
  - [ ] Local testing complete
  - [ ] Docker container tested
  - [ ] OMERO integration verified
  - [ ] Edge cases handled

### Success Criteria

Your workflow is production-ready when:

✅ Workflow completes successfully in OMERO
✅ Output files have correct names (no `.0`)
✅ Mask images import to OMERO
✅ CSV statistics appear as OMERO tables
✅ Works with various image types (8-bit, 16-bit, color, grayscale)
✅ Handles large images (100+ megapixels)
✅ Provides clear progress messages
✅ Fails gracefully with helpful error messages

---

## Additional Resources

### BIOMERO Documentation
- [BIOMERO GitHub](https://github.com/NL-BioImaging/biomero)
- [BIAFLOWS Format Specification](https://github.com/Neubias-WG5)

### Reference Implementations
- [W_CellExpansion](https://github.com/TorecLuik/W_CellExpansion)
- [W_NucleiSegmentation-Cellpose](https://github.com/TorecLuik/W_NucleiSegmentation-Cellpose)
- [W_ImageMaskProcessor](https://github.com/alexandru-mihai-94/W_ImageMaskProcessor)

### Tools
- [OpenCV Documentation](https://docs.opencv.org/)
- [Docker Documentation](https://docs.docker.com/)
- [Singularity/Apptainer Documentation](https://apptainer.org/docs/)

---

## Conclusion

Building a BIOMERO workflow involves:

1. **Core Processing**: Pure Python image analysis logic
2. **BIOMERO Integration**: bioflows_local.py for BIAFLOWS compatibility
3. **Containerization**: Docker for reproducible environments
4. **Deployment**: OMERO/SLURM integration via descriptor.json

By following this guide and separating concerns, you can create maintainable, reusable workflows that integrate seamlessly with OMERO's image management system and SLURM's high-performance computing capabilities.

**Key Takeaway**: Keep core logic separate from integration code. This makes your workflow:
- Easier to develop and test
- More maintainable
- Reusable in other contexts
- Compatible with future platform changes

---

*Document Version: 1.0.0*
*Last Updated: 2025-10-29*
*Author: Based on W_ImageMaskProcessor v1.0.0 development*
