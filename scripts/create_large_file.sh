#!/bin/bash
#SBATCH --job-name=aml3_create_single_npz
#SBATCH --mem=256G
#SBATCH --qos=long
#SBATCH --time=0-48:00
#SBATCH --output=/vast/home/jomojola/project/logs/create_single_npz.out

sh /vast/home/jomojola/project/scripts/setup.sh
poetry run python seisnet/pipelines/create_large_npz.py