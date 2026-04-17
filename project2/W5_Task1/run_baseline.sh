#!/bin/bash
#SBATCH --job-name=baseline_cls
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err
#SBATCH --partition=mlow
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1


# Activate environment
source ~/miniconda3/bin/activate c6-project2-team1

# Go to project directory
cd ~/C5/oriol/mcv-c6-2026-team1/project2

# Run
python3 Task1/main_classification.py --model baseline