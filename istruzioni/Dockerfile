# Use official CUDA 11.7 base image
FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    g++-11 \
    gcc-11 \
    cmake \
    ninja-build \
    python3.7 \
    python3.7-dev \
    python3-pip \
    python3.7-venv \
    && rm -rf /var/lib/apt/lists/*

# Set python3.7 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Create working directory
WORKDIR /workspace

# Clone Gaussian Splatting repository
RUN git clone https://github.com/graphdeco-inria/gaussian-splatting.git

# Set working directory
WORKDIR /workspace/gaussian-splatting

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip

# Install PyTorch and CUDA
RUN pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# Install dependencies
RUN pip install plyfile tqdm

# Set g++/gcc to version 11 for building the submodules
ENV CC=/usr/bin/gcc-11
ENV CXX=/usr/bin/g++-11

# Install Gaussian Splatting submodules
RUN pip install ./submodules/diff-gaussian-rasterization
RUN pip install ./submodules/simple-knn

# Set entrypoint
CMD ["/bin/bash"]

