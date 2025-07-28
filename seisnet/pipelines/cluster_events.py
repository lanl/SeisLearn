import os

import click
import numpy as np
import polars as pl
from loguru import logger
from scipy.sparse import coo_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, normalize

from seisnet.dataloaders import create_local_session
from seisnet.utils import get_data_dir, get_repo_dir


@click.command()
@click.option("-c","--clusters", nargs=1, type=click.INT, default=5, 
              help="Number of clusters. Defaults to 5.")
def cluster_events_cli(clusters):
    os.system("clear")

    logger.info("Loading the cross-correlation database...")
    # Load the cross-correlation dataframe
    engine, _ = create_local_session(return_engine=True)
    q = "SELECT * FROM correlations;"
    dfcc = pl.read_database(q,engine)
    dfcc = dfcc.rename({"waveform_id_1": "source_id", "waveform_id_2": "neighbor_id"})
    dfcc = dfcc[["source_id","neighbor_id","correlation"]]
    dfcc = dfcc.filter(pl.col("correlation").is_not_null())


    logger.info("Loading the phyical properties...")
    # Load the physical properties 
    dfev = pl.read_csv(f"{get_data_dir()}/metadata/picks.csv")
    dfev = dfev[["event_id","station_id","azimuth", "back_azimuth", "distance_km", "elevation_m",
                "depth_km", "local_depth_m", "latitude", "longitude", "snr", "takeoff_angle"]]
    dfev = dfev.with_columns(
        pl.col("snr")
        .str.strip_chars("[ ]")
        .str.replace_all(r"\s+", ",")
        .str.split(",")
        .list.eval(pl.element().cast(pl.Float64)).list.mean()
        .alias("snr")
    )

    # Extract the unique cc pairs
    source_unique = dfcc.select("source_id").unique().with_row_index(name="source_idx")
    neighbor_unique = dfcc.select("neighbor_id").unique().with_row_index(name="neighbor_idx")

    # Join to get `source_idx`
    df_mapped = dfcc.join(source_unique, on="source_id", how="left")

    # Join to get `neighbor_idx`
    df_mapped = df_mapped.join(neighbor_unique, on="neighbor_id", how="left")

    # Create the sparse cross-correlation matrix
    rows = df_mapped["source_idx"].to_numpy()
    cols = df_mapped["neighbor_idx"].to_numpy()
    data = df_mapped["correlation"].to_numpy()
    sparse_corr = coo_matrix((data, (rows, cols)),
                            shape=(source_unique.height, neighbor_unique.height))
    
    # Neighbors dataframe
    dfn = neighbor_unique.with_columns(
        pl.col("neighbor_id")
        .str.split_exact("--", 1)
        .struct.rename_fields(["event_id", "station_id"])
        .alias("fields")
    ).unnest("fields")

    # Extract the physical properties columns
    X_phy_feat =  dfn.join(dfev, on=["event_id","station_id"], how="left")\
        .drop(["neighbor_idx","neighbor_id","event_id","station_id"])
    
    logger.info("Compressing dimensions for computation")
    # Scale the physical features
    scaler = StandardScaler()
    X_feat_scl = scaler.fit_transform(X_phy_feat)

    # Reduce dimensionality with sparse-aware SVD
    svd = TruncatedSVD(n_components=100)
    X_corr_red = svd.fit_transform(sparse_corr.transpose())
    X_corr_norm = normalize(X_corr_red)        # (Optional) Normalize the features

    # Combine the physical and cross correlation features together
    X_combo = np.hstack([X_corr_norm, X_feat_scl,])

    logger.info("Clustering.....")
    kmeans = KMeans(n_clusters=clusters, random_state=42)
    labels = kmeans.fit_predict(X_combo)

    out_df = neighbor_unique.with_columns(
        pl.Series("labels",labels)
    ).rename({"neighbor_id":"waveform_name"}).drop(["neighbor_idx"])
    
    out_df.write_parquet(f"{get_repo_dir()}/grp_wvfms.parquet",row_group_size=1_000_000)
    logger.info("Clustering completed!!!")
    return True