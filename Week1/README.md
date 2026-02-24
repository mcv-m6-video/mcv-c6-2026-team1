# Week 1: Background Substraction and Object Detection

This folder contains the source code and documentation for Week 1. The goal is to implement and evaluate background substraction algorithms for object detection in traffic camera footage.

## Data Setup

Before running the code, the data directory must be configured:

1. Create `Week1/data/`.
2. Extract `AICity_data.zip` into it.
3. Copy the annotation file `ai_challenge_s03_c010-full_annotation.xml` into it.

## Source Code

⚠️ **IMPORTANT:** All scripts must be executed from `Week1/`.

The main entry point is `Week1/src/main.py`. This script handles foreground extraction (with object detection), and COCO metric evaluation ($AP_{0.5}$).

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

# FOR THE SLIDES:

TASK 1:

- Slide 1: Explain formula and how we do the modeling + mean/variance of the modeling
- Slide 2: Show with/without preprocessing (a video / frame) + post-processing w/ mathematical morphology
  Show result with opening for removing noise (disconnected regions)
  Show result with closing for connecting regions
  Show result with dilation for expanding surviving regions
- Slide 3: Show shadow removal + NMS of BBoxes
  Min + max coordinates of connected components for the bounding box (of the FG mask)

- TODO: Improve noise reduction, try to increase the opening filter!

TASK 1.2:
- Slide 1: Explain how annotation is used (show outside/occlusion GTs) + parked cars (explain why) + all to 1 class (OBJECT)
- Slide 2: Best alpha based on mAP (quantitative results + plot?) -> explain a bit which alphas were tried AND WHICH IS THE BEST!
- Slide 3: Qualitative (baseline + best alpha) + some discussion + future work

TASK 2.1:
- Slide 1: Explain formula and how we do the modeling + mean/variance of the modeling (now mean + variance video) + hyperparams procedure
- Slide 2: Quantitative results (best alpha based on mAP) -> explain which parameters tried (if needed) AND WHICH IS THE BEST!
- Slide 3: Shadow adaptation: we not only adapt to grayscale, but also to the RGB mean so that shadow is adapted? (See comparison)

TASK 2.2:
- Slide 1: Quantitative: best hyperparams, compare mAP with precision/recall curve and explain theoretical differences + common limit
- Slide 2: Non-adaptive vs adaptive masks + video w/GTs (explain visual differences + similarity in single detection) => NO TO STATIC OBJS VIDEO
- Slide 3: Show video stopped and see how they differ. Also both single gaussian, better model changes we could use GMMs (see SOTA) + explain how?

TASK 3:
- Slide 1: Explain mog2 + lbsp + how it differs with our approaches. mAP + relative with ours (+-) and analyze why + algun frame qualitatiu
- Slide 2: Mas de lo mismo con rvm (oh...no cars due to training to people. Dir DL models depends on "training") + transcd
- Slide 3: Comparison w/ our best. "Original" video w/ GTs + "Best ours + mAP" (mask det) + "Best SOTA + mAP" (mask det). DISCUSSION AT THE BOTTOM!

