import os
from glob import glob
from multiprocessing import cpu_count
from random import shuffle

import numpy as np
import polars as pl
import torch
import torch.distributed as dist
from natsort import natsorted
from loguru import logger
from numpy.random import choice
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler as DDS
from torchvision.transforms import Compose

from seisnet.utils import get_data_dir
from seisnet.dataloaders.datasets import (RidgecrestDatasetDist, SeisDataset,
                                          SeisDatasetDist)
from seisnet.dataloaders.transforms_dist import (Normalize,
                                                 RandomShiftChannels,
                                                 RandomZeroChannels, ToTensor)


class Random_Train_Val_Split:
    def __init__(self, root_dir, data_size, validation=True, val_pct=0.15):
        self.files = natsorted(glob(f"{root_dir}/*.npz"))
        assert data_size <= len(self.files), "Data size is greater than total number of available files"
        self.data_size = data_size
        self.validation = validation
        self.val_pct = val_pct

    def split_data(self):
        val_files = None

        lim_data = choice(self.files, self.data_size, replace=False)

        if self.validation:
            val_size = np.ceil(self.val_pct * self.data_size).astype(int)
            train_size = self.data_size - val_size
            train_files = lim_data[:train_size]
            val_files = lim_data[-val_size:]
        else:
            train_files = lim_data
        
        file_path = {
            "train": train_files,
            "val": val_files
        }
        return file_path
    

class Random_Train_Val_Split_Dist:
    def __init__(self, root_dir, data_size, validation=True, val_pct=0.15, seed=42):
        self.files = natsorted(glob(f"{root_dir}/*.npz"))
        assert data_size <= len(self.files), "Data size is greater than total number of available files"
        self.data_size = data_size
        self.validation = validation
        self.val_pct = val_pct
        self.seed = seed

        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

    def shard(self,files):
        """
        Manually shard the  ensure each GPU processes a unique subset of the data, 
        replaces the DDS shuffling and saves memory instead of reading all data into mem 
        """
        return files[self.rank::self.world_size]

    def split_data(self, manual_shard=True):
        rng = np.random.default_rng(self.seed)
        selected = rng.choice(self.files, self.data_size, replace=False)

        if self.validation:
            val_size = int(np.ceil(self.val_pct * self.data_size))
            train_files = selected[:-val_size]
            val_files = selected[-val_size:]
        else:
            train_files = selected
            val_files = []

        if manual_shard:
            return {
                "train": self.shard(train_files),
                "val": self.shard(val_files) if self.validation else None
            }
        else:
            return {
                "train": train_files,
                "val": val_files if self.validation else None
            }


class Stratified_Train_Val_Split:
    def __init__(self, file_df, data_size, validation=True, 
                 val_pct=0.15, seed=42):
        self.files = file_df
        assert data_size <= len(self.files), "Data size is greater than total number of available files"
        self.data_size = data_size
        self.validation = validation
        self.val_pct = val_pct
        self.seed = seed

    def split_data(self):
        val_files = None

        if self.validation:
            val_size = np.ceil(self.val_pct * self.data_size).astype(int)
            train_size = self.data_size - val_size
            train_files = stratified_sampler(self.files,train_size,seed=self.seed)
            val_files = stratified_sampler(self.files,val_size,exclude=train_files,seed=self.seed)
        else:
            train_files = stratified_sampler(self.files,self.data_size)
        
        file_path = {
            "train": train_files,
            "val": val_files
        }
        return file_path
    

class Sparse_Train_Val_Split:
    def __init__(self, file_list, validation=True, val_pct=0.15):
        self.files = file_list
        self.data_size = len(file_list)
        self.validation = validation
        self.val_pct = val_pct

    def split_data(self):
        val_files = None

        lim_data = choice(self.files, self.data_size, replace=False)

        if self.validation:
            val_size = np.ceil(self.val_pct * self.data_size).astype(int)
            train_size = self.data_size - val_size
            train_files = lim_data[:train_size]
            val_files = lim_data[-val_size:]
        else:
            train_files = lim_data
        
        file_path = {
            "train": train_files,
            "val": val_files
        }
        return file_path


class Sparse_Train_Val_Split_Dist:
    def __init__(self, file_list, validation=True, val_pct=0.15, seed=42):
        self.files = file_list
        self.validation = validation
        self.val_pct = val_pct
        self.seed = seed

        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.data_size = len(file_list)

    def split_data(self, manual_shard=True):
        # Shuffle data
        rng = np.random.default_rng(self.seed)
        shuffled = rng.permutation(self.files)

        if self.validation:
            val_size = int(np.ceil(self.val_pct * self.data_size))
            train_files = shuffled[:-val_size]
            val_files = shuffled[-val_size:]
        else:
            train_files = shuffled
            val_files = []

        # Manually slice train/val files by rank. Replace with DDS
        def shard(files):
            return files[self.rank::self.world_size]

        if manual_shard:
            # Manually shard the  ensure each GPU processes a unique subset of the data, 
            # replaces the DDS shuffling and saves memory instead of reading all data into mem 
            return {
                "train": shard(train_files),
                "val": shard(val_files) if self.validation else None
            }
        else:
            return {
                "train": train_files,
                "val": val_files if self.validation else None
            }


def random_dataloader(
        wvfm_dir:str, sample_size:int, batch_size:int=64, transform_type:str="trainval", seed=42
    )->tuple:
    """
    Create a training and validation dataloader for a dataset given an input directory

    Arguments:
        wvfm_dir(str):       File path to npz files that have been cropped to model input length
        sample_size(int):    Number of samples that should be randomly chosen for experiment
        batch_size(int):     Batch size for dataloader
        transform_type(str): Argument to include waveform normalization or channel zeroing 
                             in transformations
    
    Returns:
        (tuple): train_dataloader, val_dataloader
    
    Example:
        train_dl100, val_dl100 = random_dataloader("training_npz_files", 100)
    """
    # Transformations that can be applied to the dataset
    if not transform_type:
        transforms = ToTensor()
    elif transform_type=="trainval":
        transforms = Compose([RandomShiftChannels(),Normalize("mnstd"), ToTensor()])
    elif transform_type=="test":
        transforms = Compose([Normalize("mnstd"), ToTensor()])
    elif transform_type=="complex":
        transforms = Compose([Normalize("mnstd"), RandomZeroChannels(), ToTensor()])
    else:
        raise ValueError("Invalid transform value. Should be one of [None, `trainval`, `test`, `complex`]")
    
    local_rank = int(os.environ.get("LOCAL_RANK",0)) # Track global seed
    logger.info(f"Device {local_rank:.0f} with global seed - {seed:.0f}")

    device = "cpu"      # Load the data on cpu to avoid running out of memory
    manual_shard = True # Saves memory compared to DDS

    # strat_data_dict = Random_Train_Val_Split(wvfm_dir, sample_size).split_data()
    # train_dataset = SeisDataset(strat_data_dict["train"], transforms)
    # val_dataset = SeisDataset(strat_data_dict["val"], transforms)
    rtvsd = Random_Train_Val_Split_Dist(wvfm_dir,sample_size,validation=False,seed=seed)
    strat_data_dict = rtvsd.split_data(manual_shard)
    train_dataset = SeisDatasetDist(strat_data_dict["train"], transforms,device)
    val_transforms = Compose([Normalize("mnstd"), ToTensor()])
    val_dataset = RidgecrestDatasetDist(f"{get_data_dir()}/large_files/ridgecrest_file.npz",
                                        val_transforms, device)
    

    if dist.is_initialized():
        if manual_shard:
            logger.info("Manually sharding dataset....")
            train_sampler = val_sampler = None
            train_shuffle = True
            val_shuffle = False
        else:
            # To ensure each GPU processes a unique subset of the data, 
            # replace the standard DataLoader shuffling with DDS
            logger.info("Distributed sharding with pytorch....")
            train_sampler = DDS(train_dataset)
            val_sampler = DDS(val_dataset, shuffle=False)
            train_shuffle = False # DDS already has shuffle enabled
            val_shuffle = False
        num_workers = 0 if device!="cpu" else 8
    else:
        train_sampler = val_sampler = None
        train_shuffle = True
        val_shuffle = False
        num_workers = 4
    
    dl_config = {
        "batch_size": batch_size,
        "num_workers": num_workers,            # Use your CPU cores
        "pin_memory": num_workers > 0,         # Faster CPU -> GPU transfer
        # "prefetch_factor": 1,                # Prepare batches ahead of time
        "persistent_workers": num_workers > 0  # Don't recreate workers every epoch
    }

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, shuffle=train_shuffle, **dl_config)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, shuffle=val_shuffle, **dl_config)

    return train_dataloader, val_dataloader


def stratified_dataloader(
        file_df:pl.DataFrame, sample_size:int, seed=42,
        batch_size:int=64, transform_type:str="trainval"
    )->tuple:
    # Transformations that can be applied to the dataset
    if not transform_type:
        transforms = ToTensor()
    elif transform_type=="trainval":
        transforms = Compose([RandomShiftChannels(),Normalize("mnstd"), ToTensor()])
    elif transform_type=="test":
        transforms = Compose([Normalize("mnstd"), ToTensor()])
    elif transform_type=="complex":
        transforms = Compose([Normalize("mnstd"), RandomZeroChannels(), ToTensor()])
    else:
        raise ValueError("Invalid transform value. Should be one of [None, " \
        "`trainval`, `test`, `complex`]")
    
    strat_data_dict = Stratified_Train_Val_Split(file_df, sample_size, seed=seed).split_data()
    train_dataset = SeisDataset(strat_data_dict["train"], transforms)
    val_dataset = SeisDataset(strat_data_dict["val"], transforms)

    workers = int(np.ceil(cpu_count() * 0.9))
    dl_config = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": workers,     # Use your CPU cores
        "pin_memory": True,         # Faster CPU -> GPU transfer
        "prefetch_factor": 1,       # Prepare batches ahead of time
        "persistent_workers": True  # Don't recreate workers every epoch
    }
    train_dataloader = DataLoader(train_dataset, **dl_config)
    val_dataloader = DataLoader(val_dataset, **dl_config)

    return train_dataloader, val_dataloader


def stratified_sampler(df:pl.DataFrame, n_samples:int, file_col:str="waveform_name", 
                       label_col:str="labels", exclude:list=None, seed=42)->list:
    """
    Stratified file sampler. Given a dataframe of filepaths and labels,
    select a required number of samples that match the training input length.

    Returns:
        (list) - String of waveform filepaths
    """
    if exclude:
        df = df.filter(~pl.col(file_col).is_in(exclude))
    
    # Determine the number of unique classes
    classes = df[label_col].n_unique()
    n_per_class = np.floor(n_samples / classes)

    # Select number of samples per group.
    balanced = (
        df.group_by("labels", maintain_order=True)
        .map_groups(lambda df: df.sample(n=min(n_per_class, df.height), 
                                         with_replacement=False, seed=seed))
    )

    # Fill incomplete samples
    current_n = balanced.height
    if current_n < n_samples:
        remaining = n_samples - current_n
        extra = df.filter(~pl.col(file_col).is_in(balanced[file_col].implode())).sample(n=remaining)
        balanced = balanced.vstack(extra)
    
    return balanced[file_col].to_list()


def sparse_dataloader(
        file_list:list, batch_size:int=64, transform_type:str="trainval", seed=42
    )->tuple:
    # Transformations that can be applied to the dataset
    if not transform_type:
        transforms = ToTensor()
    elif transform_type=="trainval":
        transforms = Compose([RandomShiftChannels(),Normalize("mnstd"), ToTensor()])
    elif transform_type=="test":
        transforms = Compose([Normalize("mnstd"), ToTensor()])
    elif transform_type=="complex":
        transforms = Compose([Normalize("mnstd"), RandomZeroChannels(), ToTensor()])
    else:
        raise ValueError("Invalid transform value. Should be one of [None, " \
        "`trainval`, `test`, `complex`]")
    
    device = "cpu"      # Load the data on cpu to avoid running out of memory
    manual_shard = True # Saves memory compared to DDS

    # strat_data_dict = Sparse_Train_Val_Split(file_list).split_data()
    # val_dataset = SeisDatasetDist(strat_data_dict["val"], transforms,device)
    strat_data_dict = Sparse_Train_Val_Split_Dist(file_list,validation=False,seed=seed).split_data(manual_shard)
    train_dataset = SeisDatasetDist(strat_data_dict["train"], transforms,device)
    val_transforms = Compose([Normalize("mnstd"), ToTensor()])
    val_dataset = RidgecrestDatasetDist(f"{get_data_dir()}/large_files/ridgecrest_file.npz",
                                        val_transforms, device)

    if dist.is_initialized():
        if manual_shard:
            logger.info("Manually sharding dataset....")
            train_sampler = val_sampler = None
            train_shuffle = True
            val_shuffle = False
        else:
            # To ensure each GPU processes a unique subset of the data, 
            # replace the standard DataLoader shuffling with DDS
            logger.info("Distributed sharding with pytorch....")
            train_sampler = DDS(train_dataset)
            val_sampler = DDS(val_dataset, shuffle=False)
            train_shuffle = False # DDS already has shuffle enabled
            val_shuffle = False
        num_workers = 0 if device!="cpu" else 8
    else:
        train_sampler = val_sampler = None
        train_shuffle = True
        val_shuffle = False
        num_workers = 4
    
    dl_config = {
        "batch_size": batch_size,
        "num_workers": num_workers,            # Use your CPU cores
        "pin_memory": num_workers > 0,         # Faster CPU -> GPU transfer
        # "prefetch_factor": 1,                # Prepare batches ahead of time
        "persistent_workers": num_workers > 0  # Don't recreate workers every epoch
    }

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, shuffle=train_shuffle, **dl_config)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, shuffle=val_shuffle, **dl_config)

    return train_dataloader, val_dataloader


