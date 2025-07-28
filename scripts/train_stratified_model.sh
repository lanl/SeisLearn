#!/bin/bash
#SBATCH --job-name=aml3_train_models
#SBATCH --qos=long
#SBATCH --time=0-48:00
#SBATCH --nodes=1
#SBATCH --partition=shared-gpu --constraint="gpu_vendor:nvidia&[gpu_count:1|gpu_count:2|gpu_count:4]"
#SBATCH --output=/vast/home/jomojola/project/logs/train_stratified_model_100000.out

sh /vast/home/jomojola/project/scripts/setup.sh
poetry run stratified_train -ds 100000 -lr 1e-3 -e 1000 -b 512 -v