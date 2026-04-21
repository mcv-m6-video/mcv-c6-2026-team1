### Master in Computer Vision (Barcelona) 2025/26
# Project 2 (Task 2) @ C6 - Video Analysis

This file documents the code for Task 2 of Project 2: Action spotting on the SoccerNet Ball Action Spotting 2025 (SN-BAS-2025) dataset.

Instructions on running our DETR-based model are detailed next.

## Running the code

`main_spotting.py` is designed to train and evaluate the baseline using the settings specified in a configuration file. You can run it using the following command:

```
python main_spotting.py --model <model_name>
```

Here, `<model_name>` can be chosen freely but must match the name of a configuration file located in the [config](./config/) directory. 

To choose our best DETR model (which gives pretty bad results), you first need to download the weights. Modify `SAVE_DIR` within `get_detr_weights.sh` so that it points to your save directory (the same used in the [config](./config/) files). Then, run the script:

```
./get_detr_weights.sh
```

After that, just run the code with the `detr` model:

```
python main_spotting.py --model detr
```

For additional details on configuration options using the configuration file, refer to the README in the [config](./config/) directory.

## Important notes

- Before running the model, ensure that you have downloaded the dataset frames and updated the directory-related configuration parameters in the relevant [config](./config/) files.
- Make sure to run the `main_spotting.py` with the `mode` parameter set to `store` at least once to generate the clips and save them. After this initial run, you can set the `mode` to `load` to reuse the same clips in subsequent executions.
