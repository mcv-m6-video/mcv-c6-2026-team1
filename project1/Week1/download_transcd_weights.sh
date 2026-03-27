#!/bin/bash
set -e

# Create weights folder
path="src/TransCD/Res_SViT_E4_D4_16.pth"
gdown 1hPTbJhkyOnb1_FXdZtT-Y6j0yyU58Z1d -O "$path"

echo "Download completed: $path"