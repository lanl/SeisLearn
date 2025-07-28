from concurrent.futures import ThreadPoolExecutor
from glob import glob

import numpy as np
from natsort import natsorted
from tqdm import tqdm
from loguru import logger

from seisnet.utils import get_data_dir, get_repo_dir


def process_file(file):
    npz_file = np.load(file, allow_pickle=True)
    wv = npz_file["X"]
    lb = npz_file["y"]
    meta = npz_file["metadata"]
    
    metadata_dict = meta[()] if isinstance(meta, np.ndarray) else meta
    pIdx = metadata_dict["p_phase_index"]
    sIdx = metadata_dict.get("s_phase_index", np.nan)

    return wv, lb, pIdx, sIdx



if __name__ == "__main__":
    logger.info("Retrieving file paths")
    all_files = natsorted(glob(f"{get_data_dir()}/test_ridgecrest/*.npz"))#train_npz

    # Use threads since np.load is I/O-bound
    waveforms = []
    labels = []
    plab = []
    slab = []

    logger.info("Starting thread pool")
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_file, all_files), total=len(all_files)))

    logger.info("Unpacking results")
    # Unpack results
    for wv, lb, pIdx, sIdx in results:
        waveforms.append(wv)
        labels.append(lb)
        plab.append(pIdx)
        slab.append(sIdx)

    logger.info("Creating arrays")
    # Convert to arrays
    wv_arr = np.stack(waveforms, axis=0)
    lb_arr = np.stack(labels, axis=0)
    plab_arr = np.array(plab)
    slab_arr = np.array(slab)

    logger.info("Writing out results")
    # Save the final merged file
    np.savez(f"{get_data_dir()}/large_files/ridgecrest_file.npz", X=wv_arr, y=lb_arr, pIdx=plab_arr, sIdx=slab_arr)
