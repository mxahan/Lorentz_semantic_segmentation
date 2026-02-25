# Use a base image with CUDA 12.4 (adjust tag if needed)
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04

LABEL maintainer="Masud Ahmed <mahmed10@umbc.edu>"

# Set environment variable to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Add deadsnakes PPA and install Python 3.11 and necessary dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      python3.11 \
      python3.11-dev \
      python3.11-distutils \
      ffmpeg \
      git \
      libsm6 \
      libxext6 \
      libgl1 \
      ninja-build \
      wget \
      curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    pip install --upgrade pip

# Install PyTorch 2.6.0 and torchvision 0.21.0
RUN pip --no-cache-dir install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install additional Python libraries
RUN pip --no-cache-dir install jupyter==1.0.0 jupyterlab==3.6.5

RUN pip --no-cache-dir install \
    hydra-core \
    iopath \
    matplotlib \
    opencv-python==4.8.0.76 \
    numpy==1.26.4

# Create a symbolic link for python3 to python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Expose the port for JupyterLab
EXPOSE 8443

# Start JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--notebook-dir=/notebooks"]
