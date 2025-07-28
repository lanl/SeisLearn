#!/bin/bash
#SBATCH --job-name=aml3_cross_corr_data
#SBATCH --time=05:00:00
#SBATCH --nodes=5
#SBATCH --output=/vast/home/jomojola/project/logs/cross_corr.out

sh /vast/home/jomojola/project/scripts/setup.sh
poetry run cross_corr
