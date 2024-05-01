#!/bin/bash

# Slurm options
#SBATCH -p gpu --gres=gpu:1
#SBATCH -N 1  # Number of node
#SBATCH -n 4
#SBATCH --mem=8G
#SBATCH -t 06:00:00
##SBATCH -t 01:00:00
##SBATCH --account=carney-afleisc2-condo
#SBATCH --partition=gpu-debug
#SBATCH --mail-type=begin       # send email when job begins
#SBATCH --mail-type=end         # send email when job ends
#SBATCH --mail-user=andrea_pierre@brown.edu
#SBATCH -o slurm/stdout-%j.txt
#SBATCH -e slurm/stderr-%j.txt

# Load Slurm modules
module load python/3.11.0s-ixrhc3q
module load texlive/20220321-pocclov

# Run program
. /users/apierre3/RL_Olfaction/.venv/bin/activate
python ./TriangleTask/run_experiment.py
