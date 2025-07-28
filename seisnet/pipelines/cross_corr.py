
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import islice
from multiprocessing import cpu_count

import click
import numpy as np
import polars as pl
from loguru import logger
from scipy.stats import pearsonr
from tqdm import tqdm

from seisnet.dataloaders import (CorrelationTable, Normalize,
                                 create_local_session)
from seisnet.utils import get_data_dir, get_repo_dir


def normalize_npz(file, fmt="mnstd"):
    norm = Normalize(fmt)
    sample = {"X": file["X"], "y": file["y"], 
              "pIdx" : file["metadata"][()]["p_phase_index"],
              "sIdx" : file["metadata"][()]["s_phase_index"]}
    norm_sample = norm(sample)
    centered_zcmp = norm_sample["X"][2,2900:3100]
    return centered_zcmp


def process_single_row(df_row, fmt="mnstd", verbose=False):
    session = create_local_session()
    try:
        sid = df_row[0]#df_row.source_id
        sid_pth = os.path.join(get_data_dir(), "train_npz", f"{sid}.npz")
        if not os.path.exists(sid_pth):
            return None
        swvfm = normalize_npz(np.load(sid_pth, allow_pickle=True),fmt)

        neighbor = df_row[1]#df_row.neighbor_id
        # Check that the CC has not already been completed in the table
        existing = session.query(CorrelationTable).filter_by(waveform_id_1=sid, waveform_id_2=neighbor).first()
        rev_exists = session.query(CorrelationTable).filter_by(waveform_id_1=neighbor, waveform_id_2=sid).first()
        if existing or rev_exists:
            return None

        # Check that the neighbor waveform exists
        nid_pth = os.path.join(get_data_dir(), "train_npz", f"{neighbor}.npz")
        if not os.path.exists(nid_pth):
            return None
        
        try:
            nwvfm = normalize_npz(np.load(nid_pth,allow_pickle=True),fmt)
            corr = pearsonr(swvfm, nwvfm).statistic
            entry = CorrelationTable(sid, neighbor, corr)
            entry.insert(session, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"[{sid}-{neighbor}] Error: {e}")
            return None

    finally:
        session.close()
    return sid


def cross_corr_parallel(sparse_df, max_workers=None, fmt="mnstd", batch_size=100_000):
    """
    Cross-correlate the waveforms
    """
    if not max_workers:
        ncpus = cpu_count()
        max_workers = int(np.floor(ncpus * 0.9))
    
    logger.info(f"Total number of cpus available is {max_workers:.0f}")

    results = []
    df_iter = sparse_df.iter_rows()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for batch in batched_iterator(df_iter, batch_size=batch_size):
            futures = { executor.submit(process_single_row, row, fmt, False): row for row in batch }
            for future in tqdm(as_completed(futures), total=len(futures), position=0, 
                               desc="Multi-proc. cross correlations ...."):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.info(f"Error in future: {e}")
    return results


def batched_iterator(iterator, batch_size):
    """Yield successive batches from an iterator."""
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch

@click.command()
@click.option("-d","--data_path", nargs=1, type=click.Path(exists=True, readable=True), 
              default=f"{get_repo_dir()}/sparse2k_deduped.parquet", help="Path to the sparse waveforms dataframe.")
@click.option("-fmt","--norm_fmt", nargs=1, type=click.STRING, default="mnstd", 
              help="Waveform normalization format. Valid options are [`mnstd`,`mnmx`]. Defaults to mnstd.")
@click.option("-b","--batch_size", nargs=1, type=click.INT, default=100_000, 
              help="Number of rows to process in each batch. Defaults to 100_000.")
def cross_corr_cli(data_path, norm_fmt, batch_size):
    os.system("clear")
    logger.info("Loading sparse dataframe....")
    sparse_df = pl.read_parquet(data_path)

    logger.info("Dataframe Loaded. Calculating cross-correlations in parallel ...")
    out = cross_corr_parallel(sparse_df, fmt=norm_fmt, batch_size=batch_size)

    logger.info("Cross-correlations completed.")
    return True