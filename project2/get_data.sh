#!/bin/bash

# Stop execution if any command fails
set -e

# 1. Define constant paths
SOCCERNET_DIR="SoccerNet"
VIDEO_DIR="$SOCCERNET_DIR/SN-BAS-2025"
OUT_DIR="$SOCCERNET_DIR/SN-BAS-2025-FRAMES"

echo "====================================================="
echo " Setting up SoccerNet Ball Action Spotting"
echo " Target Directory: $SOCCERNET_DIR"
echo "====================================================="

# 2. Create the directories
mkdir -p "$VIDEO_DIR"
mkdir -p "$OUT_DIR"

# 3. Ask dataset password through terminal (and clear it from it)
echo -n "Please enter the password for the SoccerNet zip files: "
read ZIP_PASS
tput cuu1 && tput el

# 4. Download the dataset
echo "Step 1: Downloading dataset..."
python download_frames_snb.py --local_dir "$VIDEO_DIR"

# 5. Unzip everything using 7z
echo "Step 2: Extracting video files..."
find "$VIDEO_DIR" -maxdepth 1 -type f -name "*.zip" -exec 7z x {} -p"$ZIP_PASS" -o"$VIDEO_DIR" -y \;

# 6. Extract the frames
echo "Step 3: Extracting frames from videos..."
python extract_frames_snb.py --video_dir "$VIDEO_DIR" --out_dir "$OUT_DIR" \
    --width 398 --height 224 --sample_fps 25 --num_workers 5

echo "====================================================="
echo " Setup Complete!"
echo " Your dataset frames are ready at: $OUT_DIR/398x224/"
echo "====================================================="