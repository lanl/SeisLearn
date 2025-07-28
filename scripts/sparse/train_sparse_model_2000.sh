#!/bin/bash
#SBATCH --job-name=aml3_train_sparse_models_2000
#SBATCH --qos=long
#SBATCH --time=0-3:00
#SBATCH --partition=shared-gpu-ampere,shared-redstone,general,shared-gpu
#SBATCH --constraint="gpu_vendor:nvidia&[gpu_count:1|gpu_count:2]"
#SBATCH --output=/vast/home/jomojola/project/logs/sparse/train_sparse_distributed_dst2000_job-%j.out
#SBATCH --array=1-10%1
#SBATCH --requeue

rm core*
sh /vast/home/jomojola/project/scripts/setup.sh
GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
poetry run torchrun \
    --nproc_per_node=$GPUS_PER_NODE --nnodes=1 \
    seisnet/pipelines/dist_train.py \
    -sdm 2000 -lr 1e-3 -e 300 -b 512 -v
rm core*
