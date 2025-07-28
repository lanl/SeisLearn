#!/bin/bash
#SBATCH --job-name=aml3_train_sparse_models_100
#SBATCH --qos=long
#SBATCH --time=0-12:00
#SBATCH --mem=100G
#SBATCH --partition=shared-gpu-ampere,shared-redstone,general,shared-gpu
#SBATCH --constraint="gpu_vendor:nvidia&[gpu_count:8]"
#SBATCH --output=/vast/home/jomojola/project/logs/sparse/train_sparse_distributed_dst100_job-%j.out
#SBATCH --array=1-10%1
#SBATCH --requeue

rm core*
sh /vast/home/jomojola/project/scripts/setup.sh
GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
poetry run torchrun \
    --nproc_per_node=$GPUS_PER_NODE --nnodes=1 \
    seisnet/pipelines/dist_train.py \
    -sdm 100 -lr 1e-3 -e 300 -b 1024 -v
rm core*
