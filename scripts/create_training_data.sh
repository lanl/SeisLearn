#!/bin/bash
#SBATCH --job-name=aml3_create_train_data
#SBATCH --time=05:00:00
#SBATCH --output=/vast/home/jomojola/project/logs/create_train_data.out

sh /vast/home/jomojola/project/scripts/setup.sh
poetry run label_training