#!/usr/bin/env bash
#SBATCH --job-name=SRP
#SBATCH --output=../out/%x_%j.log
#SBATCH --error=../out/%x_%j.err
#SBATCH --mail-user=eyvazkhani@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

source activate SRP_ENV
# srun python main.py --r 42 --d "dna" --a "random" --m "MLP"
srun python hyperparameters/main.py