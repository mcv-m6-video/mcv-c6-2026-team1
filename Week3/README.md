# Week 3: Optical Flow and Multi-Object Single-Camera Tracking

This folder contains the source code and documentation for Week 3. The goal is to implement and evaluate optical flow methods to improve our tracking algorithm from Week 2. We apply the new tracking pipeline to multi-object, single camera tracking.

## Data Setup

Before running the code, the data directory must be configured:

1. Create `Week3/data/`.
2. Obtain initial project data.
    - 2.1. Extract `AICity_data.zip` inside `Week3/data`.
    - 2.2. Copy the annotation file `ai_challenge_s03_c010-full_annotation.xml` into `Week3/data`.
3. Obtain optical flow data.
    - 3.1. Download the <a href="http://www.cvlibs.net/datasets/kitti/">KITTI optical flow dataset</a> (`Flow 2012 > Stereo / Optical Flow`).
    - 3.2. Extract the ZIP file into `Week3/data/data_stereo_flow/`.
4. Obtain multi-object vehicle tracking data.
    - 4.1. Download the <a href="https://www.aicitychallenge.org/2022-data-and-evaluation/">CVPR 2022 AI City Challenge Track 1 dataset</a> (either the official one or the private copy for the master's project).
    - 4.2. Extract the ZIP file and ensure the path of the README file inside it is `Week3/data/AI_CITY_CHALLENGE_2022_TRAIN/ReadMe.txt`. Restructure the folders if needed.

## Source Code

⚠️ **IMPORTANT:** All scripts must be executed from `Week3/`.

Our code for this week includes the `optical_flow` module, but we focus on the entry point for running our inference and evaluation code for the 2022 CVPR AI City Challenge Track 1 (this week, though, in a Single-Camera Tracking context).

### Basic Execution

To run the code (as Python module) with default parameters for a specific sequence (choices: `[1, 3, 4]`):

```bash
python -m src.single_camera seq_id
```

### Command-Line Arguments

The main script is for evaluating a sequence when the predictions for its cameras have been already saved in the official challenge format. If you want to execute inference, use the `-e` flag. Please refer directly to the `parse_args()` function inside the script for more details on other arguments. Alternatively, you can run it with the `--help` flag to generate a complete list of all available parameters in your terminal:

```bash
python -m src.single_camera --help
```

## Directory Structure

```bash
Week3/
├── README.md
├── data/
│   ├── AI_CITY_CHALLENGE_2022_TRAIN/
│   ├── AICity_data/
│   ├── data_stereo_flow/
│   └── ai_challenge_s03_c010-full_annotation.xml
├── src/
│   ├── detection/                 # Detection code (Week 2)
│   ├── tracking/                  # Tracking code (Week 2)
│   ├── optical_flow/              # Optical flow code (Week 3)
│   │   ├── evaluation.py
│   │   ├── gmflow_method.py
│   │   ├── memflow_method.py
│   │   ├── optimize_gmflow.py
│   │   ├── optimize_memflow.py
│   │   ├── optimize_pyflow.py
│   │   ├── pyflow_method.py
│   │   └── runner.py               # Run estimation on the KITTI dataset
│   ├── eval.py                     # Adapted official evaluation code
│   ├── single_camera.py            # Main entry point of this week
│   ├── video_utils.py              # Video manipulation utilities
└── └── view_video.ipynb            # Notebook for video visualization
```
