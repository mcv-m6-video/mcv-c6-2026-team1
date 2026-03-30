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
conda create --name c6-project2-team1 python==3.10.14 -y
conda activate c6-project2-team1

# Install FFmpeg
conda install -c conda-forge ffmpeg -y

# Install for extracting SoccerNet videos
conda install -c conda-forge p7zip -y

# Install dependencies
pip install -r requirements.txt

echo "Setup complete."