FROM nvidia/cuda:12.3.1-base-ubuntu22.04

ARG USERNAME=renaud

ARG USERNAME
RUN useradd -m $USERNAME

# Set the user's home directory as the working directory
WORKDIR /home/$USERNAME

ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"

ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install packages
RUN apt-get update && apt-get install -y \
   build-essential \
   libffi-dev \
   libgdbm-dev \
   libc6-dev \
   libssl-dev \
   zlib1g-dev \
   libbz2-dev \
   libreadline-dev \
   libsqlite3-dev \
   libncurses5-dev \
   libncursesw5-dev \
   xz-utils \
   tk-dev \
   libxml2-dev \
   libxmlsec1-dev \
   liblzma-dev \
   wget \
   netcdf-bin \
   screen \
   texlive-full
# New package for Latex paper

# Download and install Python 3.11.8 from source
ENV PYTHON_VERSION=3.11.8
RUN wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tar.xz && \
   tar -xf Python-$PYTHON_VERSION.tar.xz && \
   cd Python-$PYTHON_VERSION && \
   ./configure --enable-optimizations --enable-shared && \
   make -j 8 && \
   make altinstall && \
   cd ..

# Permettre au système de trouver libpython3.11.so au runtime
# Sans ça, python3.11 crash avec "libpython3.11.so.1.0: cannot open shared object file"
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"
RUN ldconfig

# Remove the Python tarball to reduce image size
RUN rm Python-$PYTHON_VERSION.tar.xz && \
   rm -rf Python-$PYTHON_VERSION

# Set Python 3.11.8 as the default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.11 1

# # Upgrade pip
RUN python3.11 -m pip install --upgrade pip

# # Create symlink for pip
# RUN ln -s /usr/local/bin/pip3.11 /usr/local/bin/pip

# RUN python3.11 -m pip install torch torchvision torchaudio
RUN python3.11 -m pip install numcodecs --upgrade

RUN apt-get update && \
    apt-get -y install ncl-ncarg && \
    apt-get -y install cdo && \
    apt-get -y install vim && \
    apt-get -y install r-base r-base-dev
    
RUN python3.11 -m pip install numpy==1.26.3 

RUN python3.11 -m pip install pandas netcdf4 matplotlib seaborn scikit-learn scipy Pyarrow notebook ipython

# RUN apt-get update && apt-get install -y sudo

# RUN python3.11 -m pip install neuralhydrology

RUN python3.11 -m pip install geopandas geemap

RUN python3.11 -m pip install --upgrade xclim==0.47.0

RUN python3.11 -m pip install xarray[complete] rioxarray wxee 

RUN python3.11 -m pip install wget 

# RUN python3.11 -m pip install rhealpixdggs DGGSForPoly

RUN python3.11 -m pip install fiona exactextract contextily  

RUN python3.11 -m pip install pathos==0.3.0 multiprocess==0.70.15

# RUN python3.11 -m pip install xgboost 

RUN python3.11 -m pip install numcodecs==0.12.1 zarr==2.18.4 itables cartopy  pyreadr rpy2 prince imblearn

RUN python3.11 -m pip install pymc arviz

RUN python3.11 -m pip install jax[cuda12] numpyro

# RUN python3.11 -m pip install hda

RUN python3.11 -m pip install pystac_client

RUN python3.11 -m pip install torch

RUN pip install --no-cache-dir \
    "torch==2.7.1+cu128" torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu128

RUN pip install --no-cache-dir xgboost