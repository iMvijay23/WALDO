#!/bin/bash -l

#SBATCH --job-name=ae_annotation
#SBATCH --time=48:00:00
#SBATCH --partition=ica100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --qos=qos_gpu
#SBATCH --mail-user=vtiyyal1@jh.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH -A mdredze80_gpu
#SBATCH --output="/home/vtiyyal1/askdocs/outputs/ae_annotation.out"
#SBATCH --export=ALL

module load anaconda
module load cuda/11.7

conda info --envs

conda activate llmtrain_env

export TOKENIZERS_PARALLELISM=false
echo "Running python script..."

python annotate_and_save.py
