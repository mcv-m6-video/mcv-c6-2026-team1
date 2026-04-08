### Master in Computer Vision (Barcelona) 2025/26
# Project 2 (Task 1) @ C6 - Video Analysis

This file documents the code for Task 1 of Project 2: Action classification on the SoccerNet Ball Action Spotting 2025 (SN-BAS-2025) dataset.

Instructions on running the classification baseline and our best model are detailed next.

## Running the code

`main_classification.py` is designed to train and evaluate the baseline using the settings specified in a configuration file. You can run it using the following command:

```
python main_classification.py --model <model_name>
```

Here, `<model_name>` can be chosen freely but must match the name of a configuration file (e.g. `baseline.json`) located in the config directory [config](./config/). For example, to chose the baseline model, you would run 

```
python main_classification.py --model baseline
```

For additional details on configuration options using the configuration file, refer to the README in the [config](./config/) directory.

## Running our best model

First, weights must be downloaded. Modify `SAVE_DIR` within `get_best_weights.sh` so that it points to your save directory (the same used in the [config](./config/) files). Then, run the following script:

```
./get_best_weights.sh
```

After that, just run the code with the `best` model:

```
python main_classification.py --model best
```

## Important notes

- Before running the code, ensure that you have downloaded the dataset frames and updated the directory-related configuration parameters in the relevant [config](./config/) files.
- Make sure to run the `main_classification.py` with the `mode` parameter set to `store` at least once to generate the clips and save them. After this initial run, you can set the `mode` to `load` to reuse the same clips in subsequent executions.
