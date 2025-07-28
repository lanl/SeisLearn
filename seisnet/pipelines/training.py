import logging
import os
from uuid import uuid4

import click
import mlflow
import numpy as np
import polars as pl
import torch
import torch.distributed as dist
import torch.nn as nn
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau

from seisnet.dataloaders import (random_dataloader, sparse_dataloader,
                                 stratified_dataloader)
from seisnet.models import (PhaseNetBase, train_model_cycle,
                            vector_cross_entropy)
from seisnet.utils import get_data_dir, get_repo_dir, load_picks_within_radius

fp_help = """
Path to training directory.
Defaults to ../Data/train_npz
Other directory paths can be specified
"""

def setup_distributed(rank):
    "Sets up the process group and configuration for PyTorch Distributed Data Parallelism"
    # os.environ["MASTER_ADDR"] = 'localhost'
    # os.environ["MASTER_PORT"] = "12355"
    # Initialize the process group
    os.environ["NCCL_DEBUG"]="INFO" # Extensive debug info
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    return True

def cleanup_distributed(rank):
    "Cleans up the distributed environment"
    if dist.is_initialized():
        curr_device = torch.cuda.current_device()
        try:
            dist.barrier(device_ids=[curr_device])  # 🔒 synchronize all ranks
        except Exception as e:
            print(f"Rank {dist.get_rank()} failed at barrier: {e}", flush=True)
        finally:
            dist.destroy_process_group()
    return True

def set_shared_seed(local_rank):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            seed = np.random.randint(25, 2025)
        else:
            seed = 0  # placeholder

        # Broadcast seed from rank 0 to all other ranks
        seed_tensor = torch.tensor(seed, dtype=torch.int32, device=local_rank)
        dist.broadcast(seed_tensor, src=0)
        seed = seed_tensor.item()
        torch.cuda.manual_seed_all(seed)
    else:
        seed = np.random.randint(25, 2025)
    # Now set seed for all libraries
    np.random.seed(seed)
    torch.manual_seed(seed)

    return seed

def random_train_model(file_path, data_size, learning_rate, epochs, batch_size, 
                       num_classes, rank, loss_func=vector_cross_entropy, verbose=True):
    """
    Train a model with random sampling
    """

    mlflow.set_tracking_uri(f"{get_repo_dir()}/mlruns")
    mlflow.set_experiment("Phasenet-Pytorch-Random-Waveform-Experiment")
    # logging.getLogger("mlflow").setLevel(logging.ERROR)
    mlflow.autolog(silent=True)

    model = PhaseNetBase(classes=num_classes)

    if torch.cuda.is_available():
        if rank==0:
            logger.info(f"Available GPUs: {torch.cuda.device_count()}")
            logger.info("Running on GPU")
        setup_distributed(rank)
        local_rank = int(os.environ.get("LOCAL_RANK",0))
        seed = set_shared_seed(local_rank)
        torch.cuda.set_device(local_rank)
        model = model.to(local_rank)
        model = DDP(model, device_ids=[local_rank])
    else:
        device = torch.device("cpu")
        logger.info("Running on CPU")
        seed = set_shared_seed()
        model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=8, 
                                  min_lr=1e-6)
    
    # Create the training dataloaders
    train_dl, val_dl = random_dataloader(file_path, data_size, batch_size, seed=seed)
    
    if rank==0:
        model_uuid = str(uuid4())[:8]
        logger.info(f"Starting random waveform experiment {model_uuid}")
    else:
        model_uuid = "dist-gpu"

    loss_name = loss_func.__name__ if hasattr(loss_func,"__name__") else loss_func.__class__.__name__
    
    with mlflow.start_run() as run:
        params = {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "data_size": data_size,
            "batch_size": batch_size,
            "loss_function": loss_name,
            "optimizer": optimizer.__class__.__name__,
            "model_uuid": model_uuid,
            "num_classes":num_classes,
            "init_seed": seed,
        }
        # Log training parameters.
        if rank == 0:
            mlflow.log_params(params)

        history = train_model_cycle(model_uuid=model_uuid, model=model, loss_fn=loss_func, 
                                    optimizer=optimizer, scheduler=scheduler, rank=rank, 
                                    train_dataloader=train_dl, validation_dataloader=val_dl, 
                                    max_epochs=epochs, data_size=data_size, verbose=verbose)
    
    cleanup_distributed(rank=rank)

    return history


def stratified_train_model(label_df_path, files_path, data_size, learning_rate, 
                           epochs, batch_size, num_classes, 
                           loss_func=vector_cross_entropy, verbose=True):
    """
    Train a model with stratified sampling - select number of samples from each 
    clustered class e.g. if there are 5 classes and 100 total samples, 20 samples 
    will be selected from each class.
    """
    seed = np.random.randint(25,2025)
    np.random.seed(seed)
    torch.manual_seed(seed)

    mlflow.set_tracking_uri(f"{get_repo_dir()}/mlruns")
    mlflow.set_experiment("Phasenet-Pytorch-Stratified-Waveform-Experiment")
    mlflow.autolog(silent=True)

    # Load the stratified waveforms dataframe
    file_df = pl.read_parquet(label_df_path)
    file_df = file_df.with_columns(
        (files_path + "/" + pl.col("waveform_name") + ".npz").alias("waveform_name")
    )

    train_dl, val_dl = stratified_dataloader(file_df, data_size, seed, batch_size)

    model = PhaseNetBase(classes=num_classes)

    if torch.cuda.is_available():
        model = nn.parallel.DistributedDataParallel(model)
        print("Running on GPU")
    else:
        device = torch.device("cpu")
        print("Running on CPU")
        model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model_uuid = str(uuid4())[:8]
    logger.info(f"Starting stratified waveform experiment {model_uuid}")

    loss_name = loss_func.__name__ if hasattr(loss_func,"__name__") else loss_func.__class__.__name__

    with mlflow.start_run() as run:
        params = {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "data_size": data_size,
            "batch_size": batch_size,
            "loss_function": loss_name,
            "optimizer": optimizer.__class__.__name__,
            "model_uuid": model_uuid,
            "num_classes":num_classes,
            "init_seed": seed,
        }
        # Log training parameters.
        mlflow.log_params(params)

        history = train_model_cycle(model_uuid, model, loss_func, optimizer, 
                                    train_dl, val_dl, epochs, data_size, verbose)
    
    return history


def sparse_train_model(npz_files_path, picks_file_path, sep_dist_m, 
                       learning_rate, epochs, batch_size, num_classes, 
                       rank, loss_func=vector_cross_entropy, verbose=True):
    """
    Train a model with sparse sampling - select picks from a declustered 
    earthquake catalog. Catalogs are declustered by inter-earthquake distance.
    """
    mlflow.set_tracking_uri(f"{get_repo_dir()}/mlruns")
    mlflow.set_experiment("Phasenet-Pytorch-Sparse-Waveform-Experiment")
    mlflow.autolog(silent=True)

    model = PhaseNetBase(classes=num_classes)

    if torch.cuda.is_available():
        if rank==0:
            logger.info(f"Available GPUs: {torch.cuda.device_count()}")
            logger.info("Running on GPU")
        setup_distributed(rank)
        local_rank = int(os.environ.get("LOCAL_RANK",0))
        seed = set_shared_seed(local_rank)
        torch.cuda.set_device(local_rank)
        model = model.to(local_rank)
        model = DDP(model, device_ids=[local_rank])
    else:
        device = torch.device("cpu")
        logger.info("Running on CPU")
        seed = set_shared_seed()
        model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=8, 
                                  min_lr=1e-6)

    # Load the sparse waveforms file list
    file_list = load_picks_within_radius(npz_files_path, picks_file_path, 
                                         sep_dist_m, seed=seed)
    data_size = len(file_list)
    
    # Create the training dataloaders
    train_dl, val_dl = sparse_dataloader(file_list, batch_size, seed=seed)
    
    if rank==0:
        model_uuid = str(uuid4())[:8]
        logger.info(f"Starting sparse waveform experiment {model_uuid}")
    else:
        model_uuid = "dist-gpu"

    loss_name = loss_func.__name__ if hasattr(loss_func,"__name__") else loss_func.__class__.__name__
    
    with mlflow.start_run() as run:
        params = {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "data_size": data_size,
            "batch_size": batch_size,
            "loss_function": loss_name,
            "optimizer": optimizer.__class__.__name__,
            "model_uuid": model_uuid,
            "num_classes":num_classes,
            "init_seed": seed,
            "sep_dist_meters": sep_dist_m,
        }
        # Log training parameters.
        if rank == 0:
            mlflow.log_params(params)

        history = train_model_cycle(model_uuid=model_uuid, model=model, loss_fn=loss_func, 
                                    optimizer=optimizer, scheduler=scheduler, rank=rank, 
                                    train_dataloader=train_dl, validation_dataloader=val_dl, 
                                    max_epochs=epochs, data_size=data_size, verbose=verbose)
    
    cleanup_distributed(rank=rank)
    
    return history


@click.command()
@click.option("-pth","--files_path", nargs=1, type=click.Path(exists=True, readable=True), 
              default=f"{get_data_dir()}/train_npz", help=fp_help)
@click.option("-ds","--data_size", nargs=1, required=True, help="Number of waveforms to train model",
              type=click.IntRange(min=100, max=int(1.1e6)))
@click.option("-lr","--learning_rate", nargs=1, required=True, type=click.FLOAT)
@click.option("-e","--epochs", nargs=1, required=True, type=click.IntRange(min=1))
@click.option("-b","--batch_size", nargs=1, type=click.INT, default=64)
@click.option("-ncls","--num_classes", nargs=1, type=click.INT, default=1,
              help="Number of classes for phasenet prediction. Typically 1 or 3")
@click.option("-v","--verbose", nargs=1, is_flag=True)
def random_train_model_cli(files_path,data_size,learning_rate,epochs,batch_size,num_classes,verbose):
    
    if num_classes == 1:
        loss = nn.BCELoss()
    else:
        loss = vector_cross_entropy

    rank = int(os.environ.get("RANK", 0))
    
    return random_train_model(
        file_path=files_path,data_size=data_size,learning_rate=learning_rate,
        epochs=epochs,batch_size=batch_size,num_classes=num_classes,
        loss_func=loss,rank=rank,verbose=verbose
    )


@click.command()
@click.option("-lpt","--label_df_path", nargs=1, type=click.Path(exists=True, readable=True), 
              default=f"{get_repo_dir()}/grp_wvfms.parquet", help="Path to clustered waveforms dataframe")
@click.option("-pth","--file_path", nargs=1, type=click.Path(exists=True, readable=True), 
              default=f"{get_data_dir()}/train_npz", help=fp_help)
@click.option("-ds","--data_size", nargs=1, required=True, help="Number of waveforms to train model",
              type=click.IntRange(min=100, max=int(1e6)))
@click.option("-lr","--learning_rate", nargs=1, required=True, type=click.FLOAT)
@click.option("-e","--epochs", nargs=1, required=True, type=click.IntRange(min=1))
@click.option("-b","--batch_size", nargs=1, type=click.INT, default=64)
@click.option("-ncls","--num_classes", nargs=1, type=click.INT, default=1,
              help="Number of classes for phasenet prediction. Typically 1 or 3")
@click.option("-v","--verbose", nargs=1, is_flag=True)
def stratified_train_model_cli(
    label_df_path,file_path,data_size,learning_rate,epochs,batch_size,num_classes,verbose):
    
    if num_classes == 1:
        loss = nn.BCELoss()
    else:
        loss = vector_cross_entropy

    return stratified_train_model(
        label_df_path,file_path,data_size,learning_rate,epochs,batch_size,
        num_classes=num_classes,loss_func=loss,verbose=verbose
    )


@click.command()
@click.option("-nfp","--npz_files_path", nargs=1, type=click.Path(exists=True, readable=True), 
              default=f"{get_data_dir()}/train_npz", help="Path to cropped npz waveforms.")
@click.option("-pfp","--picks_file_path", nargs=1, type=click.Path(exists=True, readable=True), 
              default=f"{get_data_dir()}/metadata/picks.csv", help="Path to NC phase picks dataframe")
@click.option("-sdm","--sep_dist_m", nargs=1, help="Separation distance to decluster catalog",
              required=True, type=click.IntRange(min=0, max=int(10000)))
@click.option("-lr","--learning_rate", nargs=1, required=True, type=click.FLOAT)
@click.option("-e","--epochs", nargs=1, required=True, type=click.IntRange(min=1))
@click.option("-b","--batch_size", nargs=1, type=click.INT, default=64)
@click.option("-ncls","--num_classes", nargs=1, type=click.INT, default=1,
              help="Number of classes for phasenet prediction. Typically 1 or 3")
@click.option("-v","--verbose", nargs=1, is_flag=True)
def sparse_train_model_cli(
    npz_files_path,picks_file_path,sep_dist_m,learning_rate,epochs,batch_size,num_classes,verbose):
    
    if num_classes == 1:
        loss = nn.BCELoss()
    else:
        loss = vector_cross_entropy

    rank = int(os.environ.get("RANK", 0))

    return sparse_train_model(npz_files_path=npz_files_path, picks_file_path=picks_file_path, 
                              sep_dist_m=sep_dist_m, learning_rate=learning_rate, epochs=epochs, 
                              batch_size=batch_size, num_classes=num_classes, 
                              loss_func=loss, rank=rank, verbose=verbose)
