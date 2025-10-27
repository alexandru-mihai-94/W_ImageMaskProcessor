# Image Mask Processor - Standalone version
# 2D image thresholding and region analysis
# Compatible with BIAFLOWS format but runs standalone

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

# Install Python dependencies (standalone mode - no BIAFLOWS)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        numpy==2.2.6 \
        opencv-python==4.12.0.88

# Copy application files
ADD descriptor.json /app/descriptor.json
ADD wrapper.py /app/wrapper.py

# Set entrypoint
ENTRYPOINT ["python3", "/app/wrapper.py"]
