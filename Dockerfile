FROM nvcr.io/nvidia/pytorch:24.08-py3

# Set the working directory inside the container
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
# 1. Update system and install Python 3.12
# We use the 'deadsnakes' PPA to get the latest Python 3.12 version.
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    slurm-client \
    && rm -rf /var/lib/apt/lists/*

# 2. Set Python 3.12 as the default 'python3' interpreter
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# 3. Create a virtual environment using Python 3.12
RUN python3 -m venv /app/venv

# 4. Install main Python dependencies
# We install the standard packages first. Note that 'zarr' here might install an older stable version.
RUN /app/venv/bin/pip install --no-cache-dir \
    numpy==2.2.6 \
    git+https://github.com/huggingface/transformers.git \
    matplotlib \
    timm \   
    omegaconf \
    scipy \
    scikit-learn \
    peft \
    opencv-python-headless \
    safetensors \
    accelerate \ 
    fvcore \ 
    seaborn \
    zarr \
    datasets\ 
    webdataset\
    umap-learn\
    datashader\
    weightwatcher\
    imagecodecs\
    tifffile\
    optuna\
    optuna-dashboard\
    h5py\
    wandb\
    nibabel


RUN /app/venv/bin/pip install --upgrade zarr 

COPY . /app

# 7. Configure Environment Variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Add virtual environment to PATH so we don't need to type /app/venv/bin/python every time
ENV PATH="/app/venv/bin:$PATH"

# 8. Create necessary directories and set permissions
RUN mkdir -p /app/outputs /app/dataset /app/configs /app/neuropath_dataset /app/checkpoints && \
    chmod -R 777 /app/outputs /app/dataset /app/configs /app/neuropath_dataset /app/checkpoints
 
# Check Python version on startup
CMD ["python", "--version"]