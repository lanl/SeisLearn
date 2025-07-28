import gc
import os
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)
from multiprocessing import cpu_count
from pathlib import Path

import click
import duckdb
import numpy as np
import pandas as pd
import polars as pl
from fastparquet import write
from loguru import logger
from obspy.geodetics import degrees2kilometers, locations2degrees
from pickle_blosc import pickle, unpickle
from sqlalchemy import create_engine
from tqdm import tqdm

from seisnet.utils import (get_data_dir, get_repo_dir,
                           load_picks_without_ridgecrest)


def dist_km(p,q):
    """
    Calculate distance between two points to distance in km
    """
    deg = locations2degrees(p[0], p[1], q[:,0], q[:,1])
    dkm = degrees2kilometers(deg)
    return np.around(dkm, 2)


def sparse_sample_serial(point_ids, points, radius):
    """ 
    Sparse sample points from dense 2D point cloud

    Arguments:
        points: dim=(N, 2) where 2 is for x and y (use lat, lon => x, y)
        radius: Any two points will be separated by at least this distance
    """
    sources = []
    neighbors = []
    
    for i, p in tqdm(enumerate(points), position=0, desc="Serial sparse sample...", total=len(points)):
        key = point_ids[i]
        dists = dist_km(p, points)
        # Filter the station-station distance < radius, point is not equal to curr. point, and event-station dist. is < radius
        mask = (dists < radius) & (np.arange(len(points)) != i) & (points[:,2]<radius)
        select = point_ids[mask]
        if len(select)>0:
            sources.extend([key]*len(select))
            neighbors.extend(select)
    return sources, neighbors


def sparse_worker(i, point_ids, points, radius):
    """
    Multi-threading worker.
    """
    key = point_ids[i]
    p = points[i]
    dists = dist_km(p, points)
    mask = (dists < radius) & (np.arange(len(points)) != i) & (points[:, 2] < radius)
    select = point_ids[mask]
    print(f"{key}: {select}")
    if len(select) > 0:
        return key, select
    return None


def remove_duplicated_reverse_pairs(df):
    # Normalize each pair to ensure (A, B) and (B, A) are treated as equal
    normalized_df = df.select([
        pl.when(pl.col("source_id") < pl.col("neighbor_id"))
        .then(pl.col("source_id"))
        .otherwise(pl.col("neighbor_id"))
        .alias("id1"),

        pl.when(pl.col("source_id") < pl.col("neighbor_id"))
        .then(pl.col("neighbor_id"))
        .otherwise(pl.col("source_id"))
        .alias("id2")
    ])

    # Drop duplicates on the normalized columns
    unique_pairs = normalized_df.unique()

    # Rename back to your original column names
    final_df = unique_pairs.rename({
        "id1": "source_id",
        "id2": "neighbor_id"
    })

    return final_df


def sparse_sample_parallel(point_ids, points, radius, workers=None):
    """
    Sparse sampling in parallel
    """
    sparse = []
    sources = []
    neighbors = []
    outfile = f"{get_repo_dir()}/sparse{radius}k.parquet"
    
    if not workers:
        ncpus = cpu_count()
        workers = int(np.floor(ncpus * 0.75)) * 2

    # Multi-processing was significantly slower than multi-threading 
    # since it has to copy all the arrays into memory for each process
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(sparse_worker, i, point_ids, points, radius) for i in range(len(points))]
        for f in tqdm(as_completed(futures), total=len(futures), position=0, desc="Parallel thread sparse sample..."):
            result = f.result()
            if result:
                k, v = result
                sources.extend([k]*len(v))
                neighbors.extend(v)
                sparse.append(k)
            
            if len(sparse) > 0 and len(sparse) % 10000 == 0: # Save checkpoint
                df = pl.DataFrame({"source_id": sources, "neighbor_id": neighbors})
                if os.path.exists(outfile):
                    existing = pl.read_parquet(outfile)
                    df = pl.concat([existing, df])
                
                df.write_parquet(outfile,row_group_size=1_000_000)
                sources = []
                neighbors=[]
                del existing
                del df
                gc.collect()
                
                # Pandas is slower 
                # df = pd.DataFrame(data={"source_id": sources, "neighbor_id": neighbors})
                # df.to_parquet(f"{get_repo_dir()}/sparse{radius}k.parquet", index=False)

    return sources, neighbors


@click.command()
@click.option("-r","--radius", nargs=1, type=click.INT, default=2, 
              help="Radius of interest to cross-correlate waveforms for sparse dataset. Defaults to 2.")
@click.option("-prc","--processing", nargs=1, type=click.STRING, default="parallel", 
              help="Extract the data in parallel or serially. Valid options are [`serial`,`parallel`]. Defaults to parallel.")
def extract_sparse_picks_cli(radius:int,processing):
    """
    Extract the sparse phase pick ids into a dict and save it as a pickle file.
    To load a saved file, you can run `sparse = unpickle(f"{get_repo_dir()}/sparse2k.pkl")`
    """
    os.system("clear")
    logger.info("Loading North California phase picks....")

    data = load_picks_without_ridgecrest(f"{get_data_dir()}/metadata/picks.csv")
    data["data_ids"] = data["event_id"] + "--" + data["station_id"]
    # db_path = f"sqlite:///{Path(get_repo_dir())}//sparse_picks.db"
    # engine = create_engine(db_path, echo=False)

    assert processing in ["serial","parallel"], "Invalid processing options"

    logger.info("Starting sparse sampling of phase picks....")

    outfile = f"{get_repo_dir()}/sparse{radius}k.parquet"

    if os.path.exists(outfile):
        logger.info("Checkpoint detected, loading data .....")
        chkpt = pl.read_parquet(outfile)
        existing = chkpt["source_id"].unique().to_list() 
        # Recalculate only the last source_id
        data = data[~data.data_ids.isin(existing)].reset_index(drop=True)
        del existing
        gc.collect()
        logger.info("Checkpoint loaded")
    else:
        chkpt = None
        

    points = data[["latitude","longitude","distance_km"]].to_numpy()
    data_ids = data["data_ids"].to_numpy()

    if processing=="serial":
        sources, neighbors = sparse_sample_serial(data_ids, points, radius)
    elif processing=="parallel":
        sources, neighbors = sparse_sample_parallel(data_ids, points, radius)

    # Save the final data
    df = pl.DataFrame({"source_id": sources, "neighbor_id": neighbors})
    if os.path.exists(outfile):
        existing = pl.read_parquet(outfile)
        df = pl.concat([existing, df])
    df.write_parquet(outfile,row_group_size=1_000_000)
    df.write_csv(f"{get_repo_dir()}/sparse{radius}k.csv")

    logger.info("File saved, removing duplicates ....")
    
    # Polars is fast for reading and writing files but it runs on 
    # RAM and runs out of memory for large operations. Duckdb uses disk.
    dedup_path = outfile.replace(".parquet","_dedup.parquet")
    query = f"""
    COPY (
        SELECT DISTINCT
            LEAST(source_id, neighbor_id) AS source_id,
            GREATEST(source_id, neighbor_id) AS neighbor_id
        FROM '{outfile}'
        ORDER BY source_id
    )
    TO '{dedup_path}'
    (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 10_000_000);
    """

    conn = duckdb.connect()
    conn.execute(query)
    # df = pd.DataFrame(data={"source_id": sources, "neighbor_id": neighbors}) # pandas

    logger.info("Sparse sampling completed!!!")
    return True