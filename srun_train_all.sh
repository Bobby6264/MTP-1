#!/bin/bash
#SBATCH --job-name=moxgate_retrain_all
#SBATCH --partition=gpupart_24hour
#SBATCH --time=23:59:00
#SBATCH --output=training_all_out.log

source ~/miniconda3/bin/activate base

echo "--- STARTING STANDARD TRAINING ---"
python -u train.py

echo "--- STARTING ESCA HOLDOUT TRAINING ---"
python -u train_esca_holdout.py
