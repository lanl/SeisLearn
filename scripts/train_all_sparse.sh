#!/bin/bash

sbatch --begin=now+2hours scripts/sparse/train_sparse_model_2000.sh
sbatch --begin=now+2hours scripts/sparse/train_sparse_model_1750.sh
sbatch --begin=now+2hours scripts/sparse/train_sparse_model_1500.sh
sbatch --begin=now+2hours scripts/sparse/train_sparse_model_1250.sh
sbatch --begin=now+2hours scripts/sparse/train_sparse_model_1000.sh
sbatch --begin=now+2hours scripts/sparse/train_sparse_model_750.sh
sbatch --begin=now+2hours scripts/sparse/train_sparse_model_500.sh
sbatch --begin=now+2hours scripts/sparse/train_sparse_model_250.sh
sbatch --begin=now+2hours scripts/sparse/train_sparse_model_100.sh
