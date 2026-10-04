#!/bin/bash
#SBATCH --job-name=moxgate_explain
#SBATCH --partition=gpupart_24hour
#SBATCH --time=23:59:00
#SBATCH --output=explain_out.log

source ~/miniconda3/bin/activate base
python -u explain_model.py
