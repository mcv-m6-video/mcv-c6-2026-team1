# Week 2: Object Detection and Object Tracking

This folder contains the source code and documentation for Week 2. The goal is to implement and evaluate object detection and tracking algorithms in traffic camera footage.

## Data Setup

Before running the code, the data directory must be configured:

1. Create `Week2/data/`.
2. Extract `AICity_data.zip` into it.
3. Copy the annotation file `ai_challenge_s03_c010-full_annotation.xml` into it.

## Source Code

⚠️ **IMPORTANT:** All scripts must be executed from `Week2/`.

The main entry point is `Week2/src/main.py`. This script handles **TODO: CHANGE FROM HERE** foreground extraction (with object detection), and COCO metric evaluation ($AP_{0.5}$).

### Basic Execution

To run the code with default parameters (it stores the result of the evaluation in `Week1/result/eval`):

```bash
python src/main.py
```

To store the output videos with the predictions (in `Week1/result/videos`):

```bash
python src/main.py -s
```

To also specify a custom folder (relative to `Week1/`) for the result:

```bash
python src/main.py -s -o "folder_desired"
```

### Command-Line Arguments

The pipeline can be customized using the following arguments:

- `--model`: Selects the background substraction model. Choices are `sg` (Single Gaussian), `sga` (Single Gaussian Adaptive), `mog2` (Mixture-of-Gaussians), `lsbp` (Local SVD Binary Pattern), `rvm` (Robust Video Matting), and `transcd` (Transformer scene Change Detection).
- `--alpha`: Controls the minimum deviation for a pixel value to be detected as foreground.
- `--rho`: Controls the adaptation of the `sga` model.

**NOTE:** Before using the Transformer scene Change Detection model, its weights must be downloaded with `Week1/download_transcd_weights.sh`. 

## Directory Structure

```bash
Week1/
├── README.md
├── data/
│   ├── AICity_data/
│   └── ai_challenge_s03_c010-full_annotation.xml
├── src/
│   ├── TransCD/                 # Code for the TransCD model
│   ├── color_utils.py           # Grayscale conversion helpers
│   ├── evaluation.py            # Evaluation code for Object Detection
│   ├── main.py                  # Main execution script
│   ├── models.py                # Background subtraction models
│   ├── runner.py                # Video processing
│   ├── video_utils.py           # Video utilities
│   └── view_video.ipynb         # Notebook for video visualization on server
└── download_transcd_weights.sh  # Download weights for the TransCD model
```
