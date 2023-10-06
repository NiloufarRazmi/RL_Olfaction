#!/bin/bash

# Job Name
#SBATCH -J arrayjob 

# Walltime requested
#SBATCH -t 24:00:00

#SBATCH --mem-per-cpu=30G

# Provide index values (TASK IDs)
#SBATCH --array=1-5

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e SGD_RL_June_19_ego-%a.err
#SBATCH -o SGD_RL_June_19_ego-%a.out

# single core
#SBATCH -c 1
#SBATCH --account=carney-mnassar-condo 

# Use the $SLURM_ARRAY_TASK_ID variable to provide different inputs for each job
 
echo "Running job array number: "$SLURM_ARRAY_TASK_ID

module load matlab/R2017b

matlab-threaded -nodisplay -nojvm -r "SGD_RL_Oct_4_Allo_triangle($SLURM_ARRAY_TASK_ID), exit"
