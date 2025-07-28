#!/bin/bash
#SBATCH --job-name=aml3_train_sparse_models_125030
#SBATCH --qos=long
#SBATCH --time=0-5:00
#SBATCH --partition=shared-gpu-ampere,shared-redstone,general,shared-gpu
#SBATCH --constraint="gpu_vendor:nvidia&[gpu_count:2|gpu_count:4]"
#SBATCH --output=/vast/home/jomojola/project/logs/random/train_random_distributed_dst125030_job-%j.out
#SBATCH --array=1-10%1
#SBATCH --requeue

rm core*
sh /vast/home/jomojola/project/scripts/setup.sh
GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
poetry run torchrun \
    --nproc_per_node=$GPUS_PER_NODE --nnodes=1 \
    seisnet/pipelines/dist_train_random.py \
    -ds 125030 -lr 1e-3 -e 300 -b 1024 -v
rm core*
