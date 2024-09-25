#!/usr/bin/env bash

# Get the current date and time in the format YYYY-MM-DD_HH-MM-SS
current_date_time=$(date +"%Y-%m-%d_%H-%M-%S")
#SBATCH --job-name=SRP
#SBATCH --output=../out/$current_date_time/%x_%j.log
#SBATCH --error=../out/$current_date_time/%x_%j.err
#SBATCH --mail-user=eyvazkhani@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

source activate SRP_ENV
srun python main.py --r 42 --d "dna" --a "random" --m "MLP"