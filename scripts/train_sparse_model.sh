#!/bin/bash
#SBATCH --job-name=aml3_train_sparse_models
#SBATCH --qos=long
#SBATCH --time=0-12:00
#SBATCH --mem=100G
#SBATCH --partition=shared-gpu-ampere,shared-redstone,general,shared-gpu
#SBATCH --constraint="gpu_vendor:nvidia&[gpu_count:2|gpu_count:4|gpu_count:8]"
#SBATCH --output=/vast/home/jomojola/project/logs/train_sparse_model_distrbu_dst3000.out
#SBATCH --requeue

rm core*
sh /vast/home/jomojola/project/scripts/setup.sh

# Distributed GPU
GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
poetry run torchrun \
    --nproc_per_node=$GPUS_PER_NODE --nnodes=1 \
    seisnet/pipelines/dist_train.py \
    -sdm 3000 -lr 1e-3 -e 200 -b 1024 -v

# Single GPU or CPU
# poetry run sparse_train -sdm 100 -lr 1e-3 -e 1000 -b 1024 -v

rm core*
