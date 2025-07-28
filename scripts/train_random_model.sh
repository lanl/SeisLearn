#!/bin/bash
#SBATCH --job-name=aml3_train_baseline_all
#SBATCH --qos=long
#SBATCH --time=0-12:00
##SBATCH --mem=100G
#SBATCH --partition=shared-gpu-ampere,shared-redstone,general,shared-gpu
#SBATCH --constraint="gpu_vendor:nvidia&[gpu_count:8]"
#SBATCH --output=/vast/home/jomojola/project/logs/train_random_model_baseline_all-%j.out
#SBATCH --requeue

rm core*
sh /vast/home/jomojola/project/scripts/setup.sh

# Distributed GPU
GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
poetry run torchrun \
    --nproc_per_node=$GPUS_PER_NODE --nnodes=1 \
    seisnet/pipelines/dist_train_random.py \
    -ds 1019616 -lr 1e-3 -e 100 -b 1024 -v

# Single GPU or CPU
# poetry run random_train -ds 2534 -lr 1e-3 -e 1000 -b 256 -v

rm core*
