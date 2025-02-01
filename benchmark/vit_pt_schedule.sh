#!/bin/bash
#SBATCH --job-name=vit
#SBATCH --output=/checkpoint/miguelmartin/chkpt/vit/logs/%A_%a.out
#SBATCH --error=/checkpoint/miguelmartin/chkpt/vit/logs/%A_%a.err
#SBATCH --nodes=1
#SBATCH --partition learnfair,scavenge
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=10g
#SBATCH --time=24:00:00

python vit_pytorch.py train \
    --chkpt_dir /checkpoint/miguelmartin/chkpt/vit \
    --name my_run_nw10 \
    -nw 10 \
    --batch_size 512
