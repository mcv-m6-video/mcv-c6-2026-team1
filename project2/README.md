# MCV C6 Project 2: Ball Action Classification & Spotting (Team 1)

<a href="https://docs.google.com/presentation/d/1NDdARKTsPzL3hPxtixxzdKtxFjQSCqWHMYrkn-1ncvY/edit?usp=drive_link"><b>Final Presentation Link</b></a>

## Quick Setup

Move to this project's root and setup the environment:

```bash
cd project2
./setup.sh
conda activate c6-project2-team1
```

## Getting the dataset and data preparation

Refer to the README files in the `data/soccernetball` directory within each task folder for instructions on how to download the SN-BAS-2025 dataset, preparation of directories, and extraction of the video frames. We prepared the script **`get_data.sh`** to process everything directly, but you will need the password to unzip the data splits.

## Project Structure

`WX_TaskY/` contains everything developed during week `X` for task `Y`. Data should be downloaded in the project root.

```bash
project2/
├── .gitignore
├── download_frames_snb.py # Download (protected) data
├── extract_frames_snb.py  # Extract frames from unzipped data
├── get_data.sh            # Script to prepare the data
├── README.md
├── requirements.txt
├── setup.sh               # Script to setup the environment
├── SoccerNet/             # Project data
├── WX_TaskY/
│   ├── ...
└── └── README.md          # Documentation for Week X
```
