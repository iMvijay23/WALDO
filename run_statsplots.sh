#!/bin/bash -l

#SBATCH --job-name=generate_stats_plots
#SBATCH --time=02:00:00
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
#SBATCH --output="/home/vtiyyal1/askdocs/outputs/generate_stats_plots.out"
#SBATCH --export=ALL

module load anaconda

conda info --envs

conda activate llmtrain_env

export TOKENIZERS_PARALLELISM=false
echo "Running python script for stats and plots..."

python getstats.py
