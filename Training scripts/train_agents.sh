#!/bin/bash
#
#SBATCH -p gpu_requeue # partition (queue)
#SBATCH -n 1
#SBATCH --gres=gpu
#SBATCH --gpu-freq=high
#SBATCH --constraint=cc3.7

#SBATCH --mem 10000 # memory pool for all cores
#SBATCH -t 0-1:00 # time (D-HH:MM)
#SBATCH -o e4.%N.%j.out # STDOUT
#SBATCH -e e4.%N.%j.err # STDERR

module load Anaconda3/2020.11
source activate BMM

python3 train_agents_main.py $SLURM_ARRAY_TASK_ID 1. 0
