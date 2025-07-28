import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
from scipy.spatial import cKDTree
from shapely.geometry import Polygon

from seisnet.utils.misc import get_data_dir, timed_lru_cache


def get_ridgecrest_bounds()->gpd.GeoDataFrame:
    """
    Create a boundary polygon around the ridgecrest ttest event epicenters
    """
    ridgecrest_lons = [-118.32, -116.88, -116.88, -118.32, -118.32]
    ridgecrest_lats = [34.99, 34.99, 36.51, 36.51, 34.99]
    poly_geom = Polygon(zip(ridgecrest_lons,ridgecrest_lats))
    ridge_poly = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[poly_geom])
    return ridge_poly


@timed_lru_cache(1800)
def load_north_california_picks():
    """
    Data is always cached for 30 minutes
    """
    picks = pd.read_csv(f"{get_data_dir()}/metadata/picks.csv", dtype={"location":str})
    return picks


@timed_lru_cache(1800)
def load_picks_without_ridgecrest(path)->gpd.GeoDataFrame:
    """
    path - file path to the NC picks csv
    """
    # Load the ridgecrest boundary polygon
    rc_poly = get_ridgecrest_bounds()

    picks = pd.read_csv(path, dtype={"location":str})
    geom = gpd.points_from_xy(picks.longitude.to_numpy(),picks.latitude.to_numpy())
    picks_gdf = gpd.GeoDataFrame(picks, geometry=geom, crs="4326")
    # Remove all the ridgecrest stations
    rc_stations = ["B916","B917","B918","B921","CA01","CA02","CA03","CA04","CA05","CA06",
                   "CCC","CLC","DAW","DTP","JRC2","LRL","MPM","QSM","SLA","SRT","TOW2",
                   "WBM","WBS","WCS2","WMF","WNM","WOR","WRC2","WRV2","WVP2"]
    filt_picks = picks_gdf[
        (~picks_gdf.intersects(rc_poly.loc[0,"geometry"], align=True)) & (picks_gdf.phase_type=="['P' 'S']")
        & (~picks_gdf.station.isin(rc_stations))
    ]
    return filt_picks.reset_index(drop=True)


@timed_lru_cache(1800)
def load_events_without_ridgecrest(path)->gpd.GeoDataFrame:
    """
    path - file path to the NC events csv
    """
    # Load the ridgecrest boundary polygon
    rc_poly = get_ridgecrest_bounds()

    events = pd.read_csv(path, dtype={"location":str})
    geom = gpd.points_from_xy(events.longitude.to_numpy(),events.latitude.to_numpy())
    evt_gdf = gpd.GeoDataFrame(events, geometry=geom, crs="4326")
    filt_evts = evt_gdf[~evt_gdf.intersects(rc_poly.loc[0,"geometry"], align=True)]
    return filt_evts.reset_index(drop=True)


def filter_nc_evts_by_distance(sep_dist_m:int=1000, return_df=False, seed=42):
    """
    Given a separation distance, find all epicenters separated by that distance
    in the north california catalog
    """
    # Load the events catalog and convert to a projected coordinate distance
    dfm = load_events_without_ridgecrest(f"{get_data_dir()}/metadata/events.csv")
    df = dfm.to_crs("3310")
    df["xx"] = df.geometry.x
    df["yy"] = df.geometry.y
    df["zz"] = df.depth_km * 1000

    # Create the kDTree
    df = df[["event_id","xx","yy","zz"]]
    coords = df[["xx","yy","zz"]].values
    rng = np.random.default_rng(seed)
    coords = rng.permutation(coords)
    tree = cKDTree(coords)

    # Filter isolated points that don't intersect any neighbors
    counts = tree.query_ball_point(coords, r=sep_dist_m, return_length=True)
    isolated_idx = np.where(counts == 1)[0]
    isolated_df = df.iloc[isolated_idx]

    if return_df:
        return dfm.loc[isolated_idx,["event_id","longitude","latitude","geometry"]]
    else:
        return isolated_df.event_id.to_list()


def load_picks_within_radius(npz_path:str, picks_path:str, distance:int, seed:int)->list:
    """
    Parameters:
        npz_path(str) - path to the npz training files directory
        picks_path(str) - file path to the NC picks csv to create a dataframe
        distance(int) - distance filter to determine minimum epicenter separation distance
        seed(int) - random seed for uncertainty
    """

    evIds = filter_nc_evts_by_distance(distance, seed=seed)
    dfp = load_picks_without_ridgecrest(picks_path)
    picks = pl.from_pandas(dfp[["event_id","station_id"]])
    picks = picks.filter(pl.col("event_id").is_in(evIds))
    
    picks = picks.with_columns(
        path = npz_path + "/" + pl.col("event_id") + "--" + pl.col("station_id") + ".npz"
    )
    file_list = picks["path"].to_list()

    return file_list