#!/bin/bash

# Run with:
# export PARAMSFILE="cartesian_polar/params.ini"; sbatch cartesian_polar/slurm_job.sh

########## Begin Slurm header ##########
#SBATCH -p gpu --gres=gpu:1
#SBATCH -N 1  # Number of node
#SBATCH -n 2
#SBATCH --mem=3G
##SBATCH -t 00:15:00
#SBATCH -t 20:00:00
##SBATCH --partition=gpu-debug
#SBATCH --account=carney-afleisc2-condo
#SBATCH --mail-type=begin       # send email when job begins
#SBATCH --mail-type=end         # send email when job ends
#SBATCH --mail-user=andrea_pierre@brown.edu
#SBATCH -o slurm-logs/stdout-%j.txt
#SBATCH -e slurm-logs/stderr-%j.txt
########### End Slurm header ##########

# Load Slurm modules
module load python/3.11.0s-ixrhc3q
module load texlive/20220321-pocclov

# Run program
. /users/apierre3/RL_Olfaction/.venv/bin/activate
pip install -Ue .
# python ./TriangleTask/run_experiment.py
runexp $PARAMSFILE
