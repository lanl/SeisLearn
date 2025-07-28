#!/bin/bash
#SBATCH --job-name=aml3_create_test
#SBATCH --qos=long
#SBATCH --time=0-48:00
#SBATCH --output=/vast/home/jomojola/project/logs/create_test_data.out

sh /vast/home/jomojola/project/scripts/setup.sh
poetry run label_testing -data hawaii
poetry run label_testing -data ridgecrest
poetry run label_testing -data yellowstone