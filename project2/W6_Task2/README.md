### Master in Computer Vision (Barcelona) 2025/26
# Project 2 (Task 2) @ C6 - Video Analysis

This file documents the code for Task 2 of Project 2: Action spotting on the SoccerNet Ball Action Spotting 2025 (SN-BAS-2025) dataset.

Instructions on running the baseline and our best model are detailed next.

## Running the code

`main_spotting.py` is designed to train and evaluate the baseline using the settings specified in a configuration file. You can run `main_spotting.py` using the following command:

```
python main_spotting.py --model <model_name>
```

Here, `<model_name>` can be chosen freely but must match the name of a configuration file (e.g. `baseline.json`) located in the config directory [config](./config/). For example, to choose the baseline model, you would run

```
python main_spotting.py --model baseline
```

For additional details on configuration options using the configuration file, refer to the README in the [config](./config/) directory.

## Running our best models

First, weights must be downloaded. Modify `SAVE_DIR` within `get_best_weights.sh` so that it points to your save directory (the same used in the [config](./config/) files). Then, run the script:

```
./get_best_weights.sh
```

After that, just run the code with our `w6_best` or `best` models:

```
python main_spotting.py --model w6_best
python main_spotting.py --model best
```

## Important notes

- Before running the model, ensure that you have downloaded the dataset frames and updated the directory-related configuration parameters in the relevant [config](./config/) files.
- Make sure to run the `main_spotting.py` with the `mode` parameter set to `store` at least once to generate the clips and save them. After this initial run, you can set the `mode` to `load` to reuse the same clips in subsequent executions.
