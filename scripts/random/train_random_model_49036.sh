#!/bin/bash
#SBATCH --job-name=aml3_train_sparse_models_49036
#SBATCH --qos=long
#SBATCH --time=0-3:00
#SBATCH --partition=shared-gpu-ampere,shared-redstone,general,shared-gpu
#SBATCH --constraint="gpu_vendor:nvidia&[gpu_count:1|gpu_count:2]"
#SBATCH --output=/vast/home/jomojola/project/logs/random/train_random_distributed_dst49036_job-%j.out
#SBATCH --array=1-10%1
#SBATCH --requeue

rm core*
sh /vast/home/jomojola/project/scripts/setup.sh
GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
poetry run torchrun \
    --nproc_per_node=$GPUS_PER_NODE --nnodes=1 \
    seisnet/pipelines/dist_train_random.py \
    -ds 49036 -lr 1e-3 -e 300 -b 1024 -v
rm core*
