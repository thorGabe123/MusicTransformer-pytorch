#!/bin/bash

#SBATCH --job-name=music_transformer
#SBATCH --output=music_transformer_result-%J.out
#SBATCH --cpus-per-task=8
#SBATCH --time=5-00:00:00
#SBATCH --mem=128gb
#SBATCH --gres=gpu:8
#SBATCH --mail-user=s204461@student.dtu.dk
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=titans

## INFO (NOTE: ENSURE myenv IS ACTIVATED WHEN SUBMITTING THIS FILE)
echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\n"

python train.py -c config/full.yml -m model

echo "Done: $(date +%F-%R:%S)"
