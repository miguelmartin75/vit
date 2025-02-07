#!/bin/bash
#SBATCH --job-name=lr1-e3_vit
#SBATCH --output=/checkpoint/miguelmartin/chkpt/vit/logs/%j.out
#SBATCH --error=/checkpoint/miguelmartin/chkpt/vit/logs/%j.err
#SBATCH --nodes=1
#SBATCH --partition learnfair,scavenge
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=10g
#SBATCH --constraint=volta32gb
#SBATCH --time=72:00:00

python vit_pytorch.py train \
    --chkpt_dir /checkpoint/miguelmartin/chkpt/vit \
    --name imagenet_vit-s/32_lr1e-3_warmup30_wd0.03_adamW \
    --lr 0.001 \
    --weight_decay 0.03 \
    --epochs 300 \
    --warmup_epochs 30 \
    --patch_size 32 \
    -nw 10 \
    --batch_size 1024
