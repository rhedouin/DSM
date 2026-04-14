# ---- Configurable build arguments ----
# Multi-stage Dockerfile for scientific workloads on Azure
# Each stage builds a main dependency from source for caching, maintainability, and performance.

# Base image configuration
ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=20.04
ARG BUILD_TYPE=cpu
# Default base image for CPU builds; GPU builds should pass a CUDA-enabled base explicitly
ARG BASE_IMAGE=ubuntu:20.04
ARG INSTALL_GEO_DEPS=true
ARG INSTALL_HDF5_DEPS=true

# --- Stage 1: System dependencies ---
FROM ${BASE_IMAGE} AS system-deps

ARG INSTALL_GEO_DEPS=true
ARG INSTALL_HDF5_DEPS=true

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install minimal system packages, geospatial dependencies and GitHub CLI in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    gnupg \
    curl \
    dirmngr \
    gnupg2 \
    && if [ "$INSTALL_GEO_DEPS" = "true" ]; then \
        apt-get install -y --no-install-recommends \
        libgdal-dev \
        libgeos-dev \
        libproj-dev \
        libnetcdf-dev \
        libspatialindex-dev \
        libcurl4-openssl-dev \
        libssl-dev \
    ; fi \
    && if [ "$INSTALL_HDF5_DEPS" = "true" ]; then \
        apt-get install -y --no-install-recommends libhdf5-dev \
    ; fi \
    && update-ca-certificates \
    && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg && \
    chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null && \
    apt-get update && apt-get install -y --no-install-recommends gh && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install micromamba for fast, reproducible installs
RUN wget --quiet https://micro.mamba.pm/api/micromamba/linux-64/latest -O /tmp/micromamba.tar.bz2 && \
    tar -xjf /tmp/micromamba.tar.bz2 -C /tmp && \
    mv /tmp/bin/micromamba /usr/local/bin/micromamba && \
    rm -rf /tmp/*

# --- Stage 2: Conda environment setup ---
FROM system-deps AS conda

# Set conda environment variables
ENV MAMBA_ROOT_PREFIX=/opt/conda \
    CONDA_PREFIX=/opt/conda/envs/ri-ml-base \
    CONDA_DEFAULT_ENV=ri-ml-base

# Create conda environment directory
RUN mkdir -p /opt/conda/envs

# --- Stage 3: Main environment ---
FROM conda AS env-main

# Copy environment files
COPY .devcontainer/environment.yml /tmp/environment.yml
COPY .devcontainer/conda-lock.lock* /tmp/

# Create the conda environment
ARG USE_CONDA_LOCK=auto
RUN --mount=type=cache,target=/opt/conda/pkgs \
    set -e && \
    if [ "$USE_CONDA_LOCK" = "true" ] || { [ "$USE_CONDA_LOCK" = "auto" ] && [ -f /tmp/conda-lock.lock ]; }; then \
        echo "Using conda lock file for reproducible build"; \
        micromamba create -n ri-ml-base -f /tmp/conda-lock.lock -y && \
        echo "Installing pip packages from lock file" && \
        grep "^# pip" /tmp/conda-lock.lock | sed 's/^# pip //' | xargs -I {} sh -c 'micromamba run -n ri-ml-base pip install "{}"'; \
    else \
        echo "Using environment.yml (lock file not available or disabled)"; \
        micromamba create -n ri-ml-base -f /tmp/environment.yml -y; \
    fi

# Install libgdal-jp2openjpeg for JP2 support in GDAL
RUN --mount=type=cache,target=/opt/conda/pkgs \
    micromamba install -c conda-forge libgdal-jp2openjpeg -y
    
# --- Stage 4: IPython kernel setup ---
FROM env-main AS ipykernel

# Install IPython kernel
RUN --mount=type=cache,target=/opt/conda/pkgs \
    micromamba run -n ri-ml-base python -m ipykernel install --user --name ri-ml-base --display-name "RI-ML Base"

# --- Stage 5: Runtime environment ---
FROM ipykernel AS runtime

# Set PATH and conda environment
ENV PATH=/opt/conda/envs/ri-ml-base/bin:/opt/conda/bin:${PATH}

# Set variant-specific environment variables
ARG BUILD_VARIANT=cpu-geo
ENV BUILD_VARIANT=${BUILD_VARIANT}

# Set CUDA-specific environment variables for GPU builds
RUN if [ "${BUILD_VARIANT}" != "cpu-geo" ]; then \
      echo "export CUDA_HOME=/usr/local/cuda" >> /etc/profile.d/cuda.sh && \
      echo "export NCCL_DEBUG=INFO" >> /etc/profile.d/nccl.sh && \
      echo "export NCCL_SOCKET_IFNAME=^docker0,lo" >> /etc/profile.d/nccl.sh; \
    fi

# Create cache directories with proper permissions
RUN mkdir -p /tmp/matplotlib /tmp/jupyter /tmp/cache /tmp/numba && \
    chmod -R 777 /tmp/matplotlib /tmp/jupyter /tmp/cache /tmp/numba

# Set working directory
WORKDIR /workspaces

# Default command
CMD ["/bin/bash"]