# W_ImageMaskProcessor

2D image thresholding and region analysis workflow.

[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://hub.docker.com)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org)

## Description

This workflow processes grayscale or color images to:
1. **Generate binary masks** using intensity thresholding
2. **Detect and analyze regions** in the mask
3. **Export statistics** for each region to CSV

The workflow follows the [BIAFLOWS/Neubias WG5](https://github.com/Neubias-WG5) format and can run standalone without requiring the BIAFLOWS platform.

## Output Statistics

For each detected region, the workflow exports:

| Metric | Description |
|--------|-------------|
| `region_id` | Unique identifier (sorted by area, largest first) |
| `area_pixels` | Region size in pixels |
| `perimeter` | Boundary length |
| `centroid_x`, `centroid_y` | Center of mass coordinates |
| `bbox_x`, `bbox_y`, `bbox_width`, `bbox_height` | Bounding box |
| `aspect_ratio` | Width/height ratio |
| `circularity` | Shape metric (4π × area / perimeter²), 1.0 = perfect circle |
| `polygon` | Simplified segmentation coordinates (x,y pairs) |

## Parameters

- **min_thresh** (default: 10): Minimum intensity threshold (0-255). Pixels ≥ threshold are kept as foreground.
- **min_area** (default: 1.0): Minimum region area in pixels. Smaller regions are filtered out.

## Installation & Usage

### Docker (Recommended)

```bash
# Build
docker build -t your-username/w-imagemaskprocessor:v1.0.0 .

# Run
docker run --rm -v $(pwd):/data your-username/w-imagemaskprocessor:v1.0.0 \
  --input /data/input.tif \
  --output_mask /data/mask.tif \
  --output_csv /data/statistics.csv \
  --min_thresh 10 \
  --min_area 1.0
```

### Local Python

```bash
python wrapper.py \
  --input input.tif \
  --output_mask mask.tif \
  --output_csv statistics.csv \
  --min_thresh 10 \
  --min_area 1.0
```

## Dependencies

- Python 3.10+
- numpy==2.2.6
- opencv-python==4.12.0.88

## Repository Structure

```
W_ImageMaskProcessor/
├── wrapper.py          # Main script (BIAFLOWS compatible)
├── descriptor.json     # BIAFLOWS workflow descriptor
├── Dockerfile         # Container definition
├── .dockerignore      # Docker build exclusions
├── .gitignore         # Git exclusions
└── README.md          # This file
```

## Algorithm

The workflow implements a simple but effective pipeline:

1. **Load image**: Supports .tif, .png, .jpg, and other formats
2. **Convert to grayscale**: If input is color
3. **Apply threshold**: Binary threshold at specified intensity
4. **Find contours**: Detect connected white regions
5. **Analyze regions**: Calculate geometric and shape properties
6. **Filter**: Remove regions below minimum area
7. **Export**: Save mask image and CSV with statistics

## Publishing to DockerHub

```bash
# Login
docker login

# Tag
docker tag w-imagemaskprocessor:v1.0.0 your-username/w-imagemaskprocessor:v1.0.0

# Push
docker push your-username/w-imagemaskprocessor:v1.0.0
```

## Example Output

Processing a 122 megapixel image:
- **Processing time**: ~1 second
- **Regions detected**: 113,653 objects
- **CSV size**: 28 MB
- **Mask size**: 5.6 MB

## License

MIT License

## Citation

If you use this workflow in your research, please cite your publication here.

## Contact

- **Issues**: https://github.com/your-username/W_ImageMaskProcessor/issues

## Acknowledgments

This workflow follows the BIAFLOWS/Neubias WG5 workflow template format for image analysis workflows.
