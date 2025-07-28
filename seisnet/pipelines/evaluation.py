
from glob import glob

import click
import mlflow
import torch.nn as nn

from seisnet.models import (PhaseNetBase, eval_loss_workflow,
                            vector_cross_entropy)
from seisnet.utils import get_data_dir, get_repo_dir


@click.command()
@click.option("-mid","--model_uuid", nargs=1, type=click.STRING, required=True,
              help="Model uuid value to load saved weights.")
@click.option("-rid","--run_id", nargs=1, type=click.STRING, required=True,
              help="MLFlow experiment_id to save the loss info within the appropriate experiment run id.")
@click.option("-ncls","--num_classes", nargs=1, type=click.INT, required=True,
              help="Number of classes for phasenet prediction. Typically 1 or 3")
@click.option("-b","--batch_size", nargs=1, type=click.INT, default=256,
              help="Batch size. Defaults to 256")
@click.option("-st","--samples_thresh", nargs=1, type=click.INT, default=10,
              help="Samples threshold for classification. Defaults to 10")
@click.option("-pt","--prob_thresh", nargs=1, type=click.FLOAT, default=0.5,
              help="Probability threshold for phase peak detection. Defaults to 0.5")
@click.option("-r","--random", nargs=1, is_flag=True, help="Evaluate Random model. When omitted, evaluate stratified model.")
def eval_loss_pipeline_cli(model_uuid,run_id,num_classes,batch_size,samples_thresh,prob_thresh,random):
    """
    Perform inference on the hawaii, ridgecrest, and yellowstone datasets.
    Saves the loss values for each test dataset as a parameter in the experiment 
    run folder
    """

    model = PhaseNetBase(classes=num_classes)
    model_file = glob(f"{get_repo_dir()}/outputs/*_{model_uuid}.pth")[0]
    model.load_checkpoint(model_file)

    mlflow.set_tracking_uri(f"{get_repo_dir()}/mlruns")
    print("Tracking URI:", mlflow.get_tracking_uri())
    if random:
        mlflow.set_experiment("Phasenet-Pytorch-Random-Waveform-Experiment")
    else:
        mlflow.set_experiment("Phasenet-Pytorch-Sparse-Waveform-Experiment")

    if num_classes == 1:
        loss_func = nn.BCELoss()
    else:
        loss_func = vector_cross_entropy
    
    # Perform inference on all the test datasets
    with mlflow.start_run(run_id=run_id) as run:
        hawaii = eval_loss_workflow(model,f"{get_data_dir()}/test_hawaii",loss_func,
                                    num_classes,batch_size,samples_thresh,prob_thresh,named="hawaii",)
        yellowstone = eval_loss_workflow(model,f"{get_data_dir()}/test_yellowstone",loss_func,
                                         num_classes,batch_size,samples_thresh,prob_thresh,named="yellowstone")
        test_all = eval_loss_workflow(model,f"{get_data_dir()}/test_*",loss_func,num_classes,
                                      batch_size,samples_thresh,prob_thresh,named="all")

        # Combine the param dictionaries from all runs
        params = hawaii
        params.update(yellowstone)
        params.update(test_all)

        # Log test parameters.
        mlflow.log_params(params)
        
        # Log the residual files as an artifact
        # mlflow.log_artifact("hawaii_residuals.csv")
        # mlflow.log_artifact("yellowstone_residuals.csv")
        # mlflow.log_artifact("all_residuals.csv")