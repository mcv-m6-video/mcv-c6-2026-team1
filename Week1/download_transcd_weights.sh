#!/bin/bash
set -e

pip install -q gdown

# Create weights folder
mkdir -p weights
gdown 1hPTbJhkyOnb1_FXdZtT-Y6j0yyU58Z1d -O weights/Res_SViT_E4_D4_16.pth

echo "Download completed: weights/Res_SViT_E4_D4_16.pth"