# Week 4: Multi-Target Multi-Camera Tracking

This folder contains the source code and documentation for Week 4. The goal is to implement and evaluate Multi-Target Multi-Camera (MTMC) Vehicle Tracking from the code in Week 3. We apply vehicle Re-Identification across different cameras.

## Data Setup

Before running the code, the data directory must be configured:

1. Create `Week4/data/`.
2. Obtain MTMC tracking data.
    - 4.1. Download the <a href="https://www.aicitychallenge.org/2022-data-and-evaluation/">CVPR 2022 AI City Challenge Track 1 dataset</a> (either the official one or the private copy for the master's project).
    - 4.2. Extract the ZIP file and ensure the path of the README file inside it is `Week4/data/AI_CITY_CHALLENGE_2022_TRAIN/ReadMe.txt`. Restructure the folders if needed.

## Source Code

⚠️ **IMPORTANT:** All scripts must be executed from `Week4/`.

Our code for this week includes the `re_id` module, but we focus on `multi_camera.py`, the entry point for running our inference and evaluation code for the 2022 CVPR AI City Challenge Track 1.

### Basic Execution

To run the code (as Python module) with default parameters for a specific sequence (choices: `[1, 3, 4]`):

```bash
python -m src.multi_camera seq_id
```

### Command-Line Arguments

The main script is for evaluating a sequence when the predictions for the sequence has been already saved in the official challenge format. If you want to execute inference, use the `-e` flag. Please refer directly to the `parse_args()` function inside the script for more details on other arguments. Alternatively, you can run it with the `--help` flag to generate a complete list of all available parameters in your terminal:

```bash
python -m src.multi_camera --help
```

### Dependence on Previous Weeks

Since this week we only target the Re-Identification step, we rely on Multi-Target Single-Camera (MTSC) predictions generated from the entry point of Week 3: `single_camera.py`. Its execution is similar to `multi_camera.py`, but refer to `Week3/README.md` if needed.

## Directory Structure

```bash
Week4/
├── README.md
├── data/
│   └── AI_CITY_CHALLENGE_2022_TRAIN/
├── src/
│   ├── detection/                 # Detection code (Week 2)
│   ├── tracking/                  # Tracking code (Week 2)
│   ├── optical_flow/              # Optical flow code (Week 3)
│   ├── re_id/                     # Vehicle Re-Identification (Week 4)
│   │   ├── box_grained.py             # Bbox filtering
│   │   ├── build_reid_dataset.py      # Build data for training TransReID
│   │   ├── projector.py               # Get spatio-temporal information for a sequence
│   │   ├── tracker.py                 # ReID logic
│   │   ├── train_transreid.py         # Training script for TransReID
│   │   └── trans_reid.py              # TransReID model wrapper
│   ├── eval.py                     # Adapted official evaluation code
│   ├── multi_camera.py             # Main entry point of this week
│   ├── single_camera.py            # Main entry point of Week 3
│   ├── video_utils.py              # Video manipulation utilities
└── └── view_video.ipynb            # Notebook for video visualization
```
