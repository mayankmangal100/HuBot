# Use Python 3.10 as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y \
    git \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libmagic1 \
    build-essential \
    python3-dev \
    cmake \
    libopenblas-dev \
    ninja-build \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*



# Pre-install wheel, setuptools
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Copy requirements first to leverage Docker cache
COPY Requirements.txt .

# Set environment variables for llama-cpp-python build
ENV CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS -DLLAMA_CUDA=OFF"
ENV FORCE_CMAKE=1

# Install Python dependencies with special handling for llama-cpp-python
RUN pip install --no-cache-dir torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r Requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p Documents models

# Copy the document and model
COPY Documents/EveriseHandbook.pdf Documents/
COPY models/Qwen2.5-1.5B-Instruct.Q5_K_M_MP.gguf models/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# Set memory limits and other optimizations
ENV GGML_MAX_TENSOR_SIZE=2048
ENV GGML_MAX_NODES=2048
ENV OMP_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=4

# Start the application with gevent
CMD ["python", "api.py"]
