# SeisLearn
Understanding data amount impact on seismic deep learning [O4968]

**Table of Content**
- [Setup](#setup)
- [Data Labeling](#data-labeling)
- [Training](#training)
- [Evaluation](#evaluation)
- [HPC sync and SLURM jobs](#hpc-sync-and-slurm-jobs)
- [Clean up](#clean-up)


### Setup
The repo is setup to use [`poetry`](https://python-poetry.org/docs/) as a package manager. Install it if 
you don't have it locally and run `poetry install` in the root directory. This creates a local virtual 
environment. Poetry is used because processing pipelines are designed as embedded functions linked to python 
scripts that work like cli programs. The code depends on a data directory structured as 
```bash
.Data
├── AML
├── large_files
├── metadata
├── train_h5
└── train_npz
```
The data directory should be outside the repos root directory in it's parent folder. 
```bash
.<parent_dir>
├── Data/
└── SeisLearn/
```
This repo directory should have the following subfolders
```bash
.Project
├── .venv
├── figures
├── mlruns *
├── notebooks *
├── outputs
├── scripts *
└── seisnet *
```
> [!Note]
> `*` means the folder has created automatically from the repo. Other directories can be created manually.
> `outputs` folder is where the model checkpoints are saved. Important that its created before training any models.

### Data Labeling
The waveforms are saved as a series of hdf5 and npz files in the data directory. Run the programs below to 
crop them into individual files that can be easily passed into the training-test pipeline
- **Training** - To crop the North California h5 files to 3000 samples for , run `poetry run label_training`
- **Validatation** - Run in a 2 step workflow. First run `poetry run label_testing -data ridgecrest` to crop 
    individual waveforms, then run `poetry run python seisnet/pipelines/create_large_npz.py` to merge the files 
    into a single large npz file for faster ingestion during distributed training.
- **Testing** - data is split across 2 datasets namely `hawaii`, `ridgecrest`, and `yellowstone`. To view 
    help arguments for each command run `poetry run label_testing --help` in terminal. To generate data for a 
    specific dataset
    | Dataset | Command |
    | :---- | :---- |
    | hawaii | `poetry run label_testing -data hawaii` |
    | yellowstone | `poetry run label_testing -data yellowstone` |

    This creates multiple directories in the main *Data* directory with a test prefix e.g. `test_yellowstone`

### Training
Training scripts are also defined as cli tools with poetry. There are two types of training, namely _random_ and 
_sparse_ training. All training experiments are tracked with mlflow in the root directory. 
- View tracked experiments with `poetry mlflow ui`. This launches a flask server on port 5000. http://127.0.0.1:5000 

- **Random Training**
    - To view valid arguments `poetry run random_train --help`
    - Single model run for 10 epochs with 100 waveforms . I'm using the default train directory here 
        but a custom directory can be specified with the `-pth` flag
        `poetry run random_train -ds 100 -lr 1e-3 -e 10 -b 32 -v`
    - The verbose `-v` is used to print output to terminal. Omitting it suppresses output. Training 
        progress can also be tracked in ***mlflow*** locally. You can run commands verbosely on HPCs where 
        GUI is unavailable.
- **Sparse Training**
    - To view valid arguments `poetry run sparse_train --help`
    - Sparse training requires an additional argument that specifies a minimum separation distance (meters) that's 
        used to decluster the training catalog prior to selecting input waveforms.
    - An example training command is `poetry run sparse_train -sdm 10000 -lr 1e-3 -e 1000 -b 128 -v`

Training can be run either locally with a CPU or submitted as slurm job with GPUs. Checkpoints are saved every 100 
epochs, and the best model is saved based on the validation loss value.

> [!Important]
> When using GPUs with slurm jobs, distributed training across multiple GPUs is default. See the [scripts](./scripts/) directory 
> for specific scripts. Read the SLURM section for additional help.

### Evaluation
Evaluation jobs are used to populated test metrics like loss, recall, and f1-score in the mlflow experiment directory
- To view valid arguments `poetry run evaluate_model --help`
- Example of evaluation run for P- & S- predictions are `poetry run evaluate_model -mid 79c5e22e -rid 368efa49ba13491e90d94faeb44905d0 -ncls 3`
- Example of evaluation run for P- only predictions for 
    - random model - `poetry run evaluate_model -mid 46795a8d -rid fa2aa6d092314f93bf4fb98b4df4a27f -ncls 1 -r`
    - stratified model - `poetry run evaluate_model -mid 46795a8d -rid fa2aa6d092314f93bf4fb98b4df4a27f -ncls 1`


### HPC sync and SLURM jobs
After setting up the directory, you can pull the repo to a HPC via github or use `rsync` to copy files from terminal. 
- Copy files manually to HPC - `rsync -a ./SeisLearn/* <username>@<ssh-suffix>:/home/<username>/SeisLearn`
- Copy mlflow training results to local repo - `rsync -a <username>@<ssh-suffix>:/home/<username>/SeisLearn/mlruns ./SeisLearn`
- Copy model checkpoints to local repo - `rsync -a <username>@<ssh-suffix>:/home/<username>/SeisLearn/outputs ./SeisLearn`

<br>

> [!Important]
> First check the [`scripts/setup.sh`](./scripts/setup.sh) file, and update the appropriate modules on your HPC that
> lets you run python and poetry. This script is called with every submitted slurm job. **Don't** trigger any jobs on 
> your login node.

To submit jobs on the HPC, _cd_ into the repo directory and run the scripts `sbatch scripts/<script_name>.sh`. 

> [!Important]
> Main scripts are `sbatch scripts/train_random_model.sh` and `sbatch scripts/train_sparse_model.sh`. 
> The commands to run the distributed and single GPU/CPU commands are included. Single CPU/GPU commands might 
> need additional debugging because I migrated the code to distributed training for efficiency purposes and 
> didn't maintain backward compatibility everywhere.

Because I was submitting multiple jobs for uncertainty estimation, I created scripts that can run multiple jobs 
with different configurations on darwin. These scripts submit multiple sbatch jobs and requires access to darwin 
at lanl. You can change the partition names and sbatch commands in the _scripts/random_ and _scripts/sparse_ 
directories. 
- To run the jobs for all the random experiments - `sh scripts/train_all_random.sh`
- To run the jobs for all the sparse experiments - `sh scripts/train_all_sparse.sh`

> [!Note]
> I used `sh` and not `sbatch` for the bulk training scripts

To run the evaluation jobs on slurm, I copy the experiment id and model uuid from the mlflow GUI, and paste it into 
`scripts/evaluate_model.sh` file ***manually***. Include the proper flags to delineate between sparse and random 
experiment models because they're saved in different experiment directories to avoid errors. Evaluation is fast enough  
to be run on CPU only, but you can extend functionality to GPUs if you prefer.

**Tips**
- Check running jobs overview `squeue -u $USER -l`
- Cancel a running job 
    - Using the job name *-n* `scancel --me -n aml3_cross_corr_data`. You can use any job name specified in your sh script.
    - Using the job id `scancel <jobid>`
    - All jobs linked to your username `scancel --me`
- Check job status to see metrics like cpu efficiency etc.
    - `sstat -j <jobid>.batch --format=JobID,AveCPU,AveRSS,MaxRSS,AveVMSize`
    - `sacct -j <jobid> --format=JobID,JobName%20,Elapsed,CPUTime,AllocCPUs,State`
    - Check job run history `scontrol show job <jobid>`
    - Check GPU utilization `srun --jobid=<jobid> nvidia-smi -l 1`
- Check storage size in active directory: `du -sh`. When storage is full, jobs fail silently without coherent error logs. 


### Clean up
When distributed GPU is used, mlflow creates duplicated experiments for each additional  GPU. No metrics/params are logged 
in this experiment, and it clutters the experiment and output directories. I include the details in a clean up 
[file](./seisnet/pipelines/cleanup.py). For experiments that fail, I also include their details in the clean file. Run the 
clean up file with `poetry run python seisnet/pipelines/cleanup.py` to delete the experiments and their associated artifacts. 

> [!Tip]
> If you dislike _poetry_, after installing the virtual env, you can activate it with `source .venv/bin/activate` and run 
> python scripts directly with `python <script_name>.py` in terminal.

<br><br><br>

© 2025. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit
others to do so.
(End of Notice)



Copyright 2025
Permission is hereby granted, free of charge, to any person obtaining a copy of this software 
and associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, 
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
(End of Notice)
