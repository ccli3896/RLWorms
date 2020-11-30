#!/bin/bash
#SBATCH -n 96               # num cores
#SBATCH -N 3                # Ensures all cores are on one machine
#SBATCH -t 0-5:00           # Runtime D-HH:MM, min 10 minutes
#SBATCH --mem=5000          # Memory pool for all cores (mb)
#SBATCH --array=0-750       # array number


module purge

module load Anaconda3
source activate r1127

python ./11_27_plain_mod.py ${SLURM_ARRAY_TASK_ID}
