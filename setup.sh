#!/bin/bash

# Download and install Miniconda:
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -u -p ~/miniconda3
rm Miniconda3-latest-Linux-x86_64.sh

# After installing, close and reopen your terminal application or refresh it by running the following command:
source ~/miniconda3/bin/activate

# To initialize conda on all available shells, run the following command:
conda init --all

# Creating the environment:
conda create --name c6-team1 python==3.10.14 -y
conda activate c6-team1

# GCC compilers for Detectron2
conda install -c conda-forge gcc_linux-64=11 gxx_linux-64=11 -y

# Install FFmpeg
conda install -c conda-forge ffmpeg -y

## PyTorch installation

# In the cluster the cuda version is the 12.1, so the you need to install a torch version that fits in this cuda version
# This is the stablest one that supports cuda 12.1
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121  

# Detectron2 installation
pip install git+https://github.com/facebookresearch/detectron2.git --no-build-isolation

# Other packages are in the requirements.txt
pip install -r requirements.txt