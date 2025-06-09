#!/bin/bash

# Run with:
# export PARAMSFILE="params.ini"; sbatch slurm_job.sh

# Interactive job:
# interact -q gpu -g 1 

########## Begin Slurm header ##########
#SBATCH -p gpu --gres=gpu:1
##SBATCH --partition=gpu-debug
#SBATCH -N 1  # Number of node
#SBATCH -n 2
#SBATCH --mem=3G
##SBATCH -t 01:00:00
#SBATCH -t 03:00:00
#SBATCH --account=carney-afleisc2-condo
#SBATCH --mail-type=begin       # send email when job begins
#SBATCH --mail-type=end         # send email when job ends
#SBATCH --mail-user=juan_mendez@brown.edu
#SBATCH -o slurm-logs/%j-stdout.txt
#SBATCH -e slurm-logs/%j-stderr.txt
########### End Slurm header ##########

# Load Slurm modules
module load python/3.11.0s-ixrhc3q
module load texlive/20220321-pocclov

# Run program
. /users/jjmendez/data/jmendez/RL_Olfaction/cartesian_polar/.venv/bin/activate # Activate virtual environment
# pip install -Ue .
# python ./TriangleTask/run_experiment.py
runexp $PARAMSFILE
# python run_experiment.py