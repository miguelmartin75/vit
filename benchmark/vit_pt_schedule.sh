#!/bin/bash
#SBATCH --job-name=vit
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

# python vit_pytorch.py train \
#     --chkpt_dir /checkpoint/miguelmartin/chkpt/vit_ref \
#     --name ref_run_nw10 \
#     --ref \
#     -nw 10 \
#     --batch_size 512

# python vit_pytorch.py train \
#     --chkpt_dir /checkpoint/miguelmartin/chkpt/vit \
#     --name imagenet_bs512 \
#     -nw 10 \
#     --batch_size 512

python vit_pytorch.py train \
    --chkpt_dir /checkpoint/miguelmartin/chkpt/vit_tinyimagenet \
    --name tinyimagenet_bs512 \
    --num_classes 200 \
    --img_size 64 \
    -nw 10 \
    --batch_size 512
