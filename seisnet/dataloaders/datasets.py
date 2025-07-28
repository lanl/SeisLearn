import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset
from tqdm import tqdm


class SeisDatasetDist(Dataset):
    def __init__(self, file_list, transform=None, preload_to_device=None, workers=8):
        self.transform = transform
        self.device = preload_to_device or "cpu"
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        logger.info(f"The active device is {self.device} - rank {local_rank}")
        N = len(file_list)

        # Preallocate arrays
        self.X = torch.empty((N, 3, 6000), dtype=torch.float32, device=self.device)
        self.y = torch.empty((N, 1, 6000), dtype=torch.float32, device=self.device)
        self.pIdx = torch.empty((N,), dtype=torch.int32, device=self.device)
        self.sIdx = torch.empty((N,), dtype=torch.int32, device=self.device)

        # Define the file processing function
        def load_sample(i_file):
            i, file_name = i_file
            npz = np.load(file_name, allow_pickle=True)
            meta = npz["metadata"]
            metadata = meta[()] if isinstance(meta, np.ndarray) else meta

            return (
                i,
                torch.tensor(npz["X"], dtype=torch.float32),
                torch.tensor(npz["y"], dtype=torch.float32),
                torch.tensor(metadata["p_phase_index"], dtype=torch.int32),
                torch.tensor(metadata.get("s_phase_index", float("nan")), dtype=torch.float32)
            )

        # Load using a thread pool
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for i, x, y, pidx, sidx in tqdm(executor.map(load_sample, enumerate(file_list)), 
                                            desc=f"\nLoading train data for Device {local_rank}",
                                            total=N, position=0):
                self.X[i] = x
                self.y[i] = y
                self.pIdx[i] = pidx
                self.sIdx[i] = sidx

        # for i, file_name in tqdm(enumerate(file_list),position=0,total=N):
        #     npz = np.load(file_name, allow_pickle=True)
        #     meta = npz["metadata"]
        #     metadata = meta[()] if isinstance(meta, np.ndarray) else meta

        #     self.X[i] = torch.tensor(npz["X"], dtype=torch.float32)
        #     self.y[i] = torch.tensor(npz["y"], dtype=torch.float32)
        #     self.pIdx[i] = torch.tensor(metadata["p_phase_index"], dtype=torch.float32)
        #     self.sIdx[i] = torch.tensor(metadata.get("s_phase_index", float("nan")), dtype=torch.float32)

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        sample = {
            "X": self.X[idx],
            "y": self.y[idx],
            "pIdx": self.pIdx[idx],
            "sIdx": self.sIdx[idx]
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class RidgecrestDatasetDist(Dataset):
    def __init__(self, file_path, transform=None, preload_to_device=None):
        self.transform = transform
        self.device = preload_to_device or "cpu"
        logger.info(f"Memory-mapping Ridgecrest dataset to device: {self.device}")

        data = np.load(file_path, allow_pickle=True, mmap_mode="r")

        required_keys = {"X", "y", "pIdx", "sIdx"}
        if not required_keys.issubset(data.files):
            raise ValueError(f"Missing required keys in Ridgecrest .npz file.")

        # Keep mmap-backed arrays; don't convert to torch.Tensor to avoid running out of mem.
        self.X = data["X"]
        self.y = data["y"]
        self.pIdx = data["pIdx"]
        self.sIdx = np.where(self.pIdx == None, np.nan, data["sIdx"]).astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        sample = {
            "X": torch.tensor(self.X[idx], dtype=torch.float32),
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
            "pIdx": torch.tensor(self.pIdx[idx], dtype=torch.int32),
            "sIdx": torch.tensor(self.sIdx[idx], dtype=torch.float32)
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


# class RidgecrestDatasetDist(Dataset):
#     def __init__(self, file_path, transform=None, preload_to_device=None):
#         """
#         file_path: str
#             Path to a single .npz file containing keys: 'X', 'y', 'pIdx', 'sIdx'
#         """
#         self.transform = transform
#         self.device = preload_to_device or "cpu"
#         logger.info(f"Loading Ridgecrest validation dataset to device: {self.device}")

#         # Load full .npz file
#         data = np.load(file_path, allow_pickle=True)

#         # Validate keys
#         required_keys = {"X", "y", "pIdx", "sIdx"}
#         if not required_keys.issubset(set(data.keys())):
#             raise ValueError(f".npz file must contain keys {required_keys}")
        
#         sidx_clean = np.where(data["sIdx"] == None, np.nan, data["sIdx"]).astype(np.float32)


#         # Convert numpy arrays to torch tensors and move to device
#         self.X = torch.tensor(data["X"], dtype=torch.float32, device=self.device)
#         self.y = torch.tensor(data["y"], dtype=torch.float32, device=self.device)
#         self.pIdx = torch.tensor(data["pIdx"], dtype=torch.float32, device=self.device)
#         self.sIdx = torch.tensor(sidx_clean, dtype=torch.float32, device=self.device)

#     def __len__(self):
#         return self.X.size(0)

#     def __getitem__(self, idx):
#         sample = {
#             "X": self.X[idx],
#             "y": self.y[idx],
#             "pIdx": self.pIdx[idx],
#             "sIdx": self.sIdx[idx]
#         }

#         if self.transform:
#             sample = self.transform(sample)

#         return sample


class SeisDataset(Dataset):
    def __init__(self, file_list:list, transform=None):
        self.transform = transform
        self.files = file_list
    
    def __len__(self)->int:
        return len(self.files)
    
    def __getitem__(self, idx:int)->dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if idx > len(self.files)-1:
            raise ValueError("Index does not exist in dataset")
        file_name = self.files[idx]
        npz_file = np.load(file_name,allow_pickle=True)
        meta = npz_file["metadata"]
        if isinstance(meta, np.ndarray):
            metadata_dict = meta[()]
        else:
            metadata_dict = meta
        pIdx = metadata_dict["p_phase_index"]
        sIdx = metadata_dict["s_phase_index"] or np.nan

        sample = { "X":npz_file["X"], "y":npz_file["y"], "pIdx":pIdx, "sIdx":sIdx  }

        if self.transform:
            sample = self.transform(sample)

        return sample