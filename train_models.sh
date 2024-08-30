#!/bin/bash -l

#SBATCH --job-name=das_aug24
#SBATCH --time=48:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --qos=qos_gpu
#SBATCH --mail-user=vtiyyal1@jh.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH -A mdredze1_gpu
#SBATCH --output="/home/vtiyyal1/askdocs/outputs/das_aug24.out"
#SBATCH --export=ALL

module load anaconda
module load cuda/11.7

conda info --envs

conda activate llmtrain_env

export TOKENIZERS_PARALLELISM=false

echo "Running python script for augmented model responses..."

python train_models.py