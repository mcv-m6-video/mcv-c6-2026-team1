#!/usr/bin/env bash
set -e

echo "Initializing git submodules..."
git submodule update --init --recursive

EXTERNAL_DIR="./external"
mkdir -p "$EXTERNAL_DIR"

echo "Installing gdown..."
pip install -q gdown

echo "Downloading pretrained models..."

# Download GMFLow models
FILE_ID=1d5C5cgHIxWGsFR1vYs5XrQbbUiZl9TX2
ZIP_PATH="$EXTERNAL_DIR/pretrained.zip"
gdown --id "$FILE_ID" -O "$ZIP_PATH"
unzip -o "$ZIP_PATH" -d "$EXTERNAL_DIR"
rm "$ZIP_PATH"

# Download MemFLow models
wget -O "$EXTERNAL_DIR/pretrained/MemFlowNet_kitti.pth" "https://github.com/DQiaole/MemFlow/releases/download/v1.0.0/MemFlowNet_kitti.pth"
wget -O "$EXTERNAL_DIR/pretrained/MemFlowNet_T_kitti.pth" "https://github.com/DQiaole/MemFlow/releases/download/v1.0.0/MemFlowNet_T_kitti.pth"
wget -O "$EXTERNAL_DIR/pretrained/MemFlowNet_sintel.pth" "https://github.com/DQiaole/MemFlow/releases/download/v1.0.0/MemFlowNet_sintel.pth"
wget -O "$EXTERNAL_DIR/pretrained/MemFlowNet_T_sintel.pth" "https://github.com/DQiaole/MemFlow/releases/download/v1.0.0/MemFlowNet_T_sintel.pth"


echo "Compiling PyFlow files..."
(
cd external/pyflow
python setup.py build_ext --inplace
)

echo "Setup complete."