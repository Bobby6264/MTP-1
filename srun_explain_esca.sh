#!/bin/bash
#SBATCH --job-name=moxgate_esca_explain
#SBATCH --partition=gpupart_24hour
#SBATCH --time=23:59:00
#SBATCH --output=explain_esca_out.log

source ~/miniconda3/bin/activate base
python -u explain_model_esca_holdout.py
