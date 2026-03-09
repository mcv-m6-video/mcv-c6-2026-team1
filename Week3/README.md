# Week 3: Optical Flow and Multi-Object Single-Camera Tracking

This folder contains the source code and documentation for Week 3. The goal is to implement and evaluate optical flow methods to improve our tracking algorithm from Week 2. We apply the new tracking pipeline to multi-object, single camera tracking.

## Data Setup

Before running the code, the data directory must be configured:

1. Create `Week3/data/`.
2. Obtain initial project data.
    - 2.1. Extract `AICity_data.zip` into `Week3/data`.
    - 2.2. Copy the annotation file `ai_challenge_s03_c010-full_annotation.xml` into `Week3/data`.
3. Obtain optical flow data.
    - 3.1. Download the <a href="http://www.cvlibs.net/datasets/kitti/">KITTI optical flow dataset</a> (`Flow 2012 > Stereo / Optical Flow`).
    - 3.2. Create `Week3/data/data_stereo_flow/` and extract the downloaded ZIP file there.
4. Obtain multi-object vehicle tracking data.
    - 4.1. Download the <a href="https://www.aicitychallenge.org/2022-data-and-evaluation/">CVPR 2022 AI City Challenge Track 1 dataset</a> (either the official one or the private copy for the master's project).
    - 4.2. Extract the ZIP file and ensure the path of the README file inside it is `Week3/data/AI_CITY_CHALLENGE_2022_TRAIN/ReadMe.txt`. Restructure the folders if needed.

## Source Code

⚠️ **IMPORTANT:** All scripts must be executed from `Week3/`.

**TODO FROM HERE**

Our code for this week is split in 2 modules (one for each task): `detection` and `tracking`. Therefore, there is a main entry point for each: `Week2/src/detection/run_detection.py` and `Week2/src/tracking/run_tracking.py`, respectively. The former handles car detection, whereas the latter targets tracking.

### Basic Execution

To run the code (as Python modules) with default parameters:

```bash
python -m src.detection.run_detection
python -m src.detection.run_tracking
```

### Command-Line Arguments

Both scripts feature command-line customization. Please refer directly to the `parse_args()` function inside each script. Alternatively, you can run them with the `--help` flag to generate a complete list of all available parameters in your terminal:

```bash
python -m src.detection.run_detection --help
python -m src.detection.run_tracking --help
```

## Directory Structure

```bash
Week2/
├── README.md
├── data/
│   ├── AICity_data/
│   └── ai_challenge_s03_c010-full_annotation.xml
├── src/
│   ├── detection/
│   │   ├── weights/
│   │   │   └── yolo_best.pt                        # Best trained weights (strategy A)
│   │   ├── cross_validate.py                       # Runs K-Fold cross-validation
│   │   ├── evaluation.py                           # Utilities for detection evaluation
│   │   ├── faster_rcnn.py                          # Faster R-CNN model definition
│   │   ├── prepare_datasets.py                     # Prepare data according to both Strategy B and C
│   │   ├── run_bayes_sweep.sh                      # Run Bayesian search for YOLO
│   │   ├── run_detection.sh                        # Main entry point for detection inference
│   │   ├── run_sweep.sh                            # Run grid search fine-tuning sweep
│   │   ├── sweep_bayes_yolo.yaml                   # Configuration file for the Bayesian search for YOLO
│   │   ├── sweep_faster_rcnn.yaml                  # Configuration file for training Faster R-CNN
│   │   ├── sweep_yolo.yaml                         # Configuration file for training YOLO
│   │   ├── train_faster_rcnn.py                    # Run to train Faster R-CNN
│   │   ├── train_yolo.py                           # Run to train YOLO
│   │   └── yolo.py                                 # YOLO model definition
│   ├── tracking/
│   │   ├── evaluation/
│   │   │   ├── _base_metric.py                     # Base tracking metric class
│   │   │   ├── _timing_.py                         # Tracks execution time performance
│   │   │   ├── main.py                             # Evaluates tracking performance metrics
│   │   │   ├── methods.py                          # Tracking metric calculation algorithms
│   │   │   └── utils.py                            # Tracking evaluation helper functions
│   │   ├── experiments.py                          # Runs multi-tracker experiment configurations
│   │   ├── plotting.py                             # Generates tracking visualization graphs
│   │   ├── run_tracking.py                         # Main entry point for tracking inference
│   │   ├── sort.py                                 # SORT tracking algorithm implementation
│   │   ├── trackers.py                             # Defines custom tracking classes
│   │   └── tracking_utils.py                       # General object tracking helpers
│   ├── video_utils.py                              # Video manipulation utilities
└── └── view_video.ipynb                            # Notebook for video visualization
```
