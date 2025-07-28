- Run job. `darwin` -> `cd project` -> `sbatch scripts/<script_name>.sh`
- Sync files `rsync -a ./project/scripts/* jomojola@darwin-fe1.lanl.gov:/home/jomojola/project/scripts`
- Check job status `squeue -u jomojola`
- Cancel a job that I started `scancel --me -n aml3_cross_corr_data` where -n refers to the job name
- Check job efficiency `sstat -j <jobid>.batch --format=JobID,AveCPU,AveRSS,MaxRSS,AveVMSize`
- Check job run history `scontrol show job <jobid>`
- Check GPU utilization `srun --jobid=<jobid> nvidia-smi`

srun --jobid=<jobid> --pty bash
sacct -j <jobid> --format=JobID,JobName%20,Elapsed,CPUTime,AllocCPUs,State

sacct -j 16590674 --format=JobID,JobName%20,Elapsed,CPUTime,AllocCPUs,State
srun --jobid=16582387 --pty bash


### Sync
- Local 2 Darwin - `rsync -a ./project/* jomojola@darwin-fe1.lanl.gov:/home/jomojola/project`
- Darwin 2 Local - `rsync -a jomojola@darwin-fe1.lanl.gov:/home/jomojola/project/mlruns ./project`
- Darwin 2 Local - `rsync -a jomojola@darwin-fe1.lanl.gov:/home/jomojola/project/outputs ./project`


poetry run evaluate_model -mid e4481a9e -rid eeb0a78596184f57beea6c5373bcfba3 -ncls 1 -r
srun --jobid=16575449 nvidia-smi dmon
srun --jobid=16588734 nvidia-smi -l 1
scancel --me -n aml3_train_sparse_models
Check storage size (limit 500G): `du -sh`



### Retrieve model results
rsync -a jomojola@darwin-fe1.lanl.gov:/home/jomojola/project/mlruns ./project
rsync -a jomojola@darwin-fe1.lanl.gov:/home/jomojola/project/outputs ./project

### Push local changes to remote
rsync -a ./project/* jomojola@darwin-fe1.lanl.gov:/home/jomojola/project
