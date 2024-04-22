#!/bin/bash

# Slurm options
#SBATCH -p gpu --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=8G
#SBATCH -t 06:00:00
#SBATCH --partition=carney-afleisc2-condo
#SBATCH --mail-type=begin       # send email when job begins
#SBATCH --mail-type=end         # send email when job ends
#SBATCH --mail-user=andrea_pierre@brown.edu

# Load Slurm modules
module load python/3.11.0s-ixrhc3q
module load texlive/20220321-pocclov

# Run program
. /users/apierre3/RL_Olfaction/.venv/
python run_experiment.py
