import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from seisnet.utils import (get_data_dir, get_repo_dir, get_ridgecrest_bounds,
                           load_north_california_picks)


def plot_catalog_distribution_without_ridgecrest():
    # Load the data
    rc_poly = get_ridgecrest_bounds()
    picks = load_north_california_picks()
    geom = gpd.points_from_xy(picks.longitude.to_numpy(),picks.latitude.to_numpy())
    picks_gdf = gpd.GeoDataFrame(picks, geometry=geom, crs="4326")
    filt_picks = picks_gdf[
        (~picks_gdf.intersects(rc_poly.loc[0,"geometry"], align=True)) & 
        (picks_gdf.phase_type=="['P' 'S']")
    ]

    # Create the map plot
    fig,ax = plt.subplots(1,2,figsize=(8,5))
    picks_gdf.plot(ax=ax[0], markersize=2, marker="o")
    rc_poly.plot(ax=ax[0], facecolor="None")
    ax[0].set_title("NCEDC")
    filt_picks.plot(ax=ax[1], markersize=2, marker="o")
    rc_poly.plot(ax=ax[1], facecolor="None")
    ax[1].set_title("NCEDC (minus RidgeCrest)")
    plt.savefig(f"{get_repo_dir()}/figures/catalog_without_ridgecrest.png",
                dpi=200,edgecolor="none",bbox_inches="tight")
    plt.close()
    print("Created plot")

    return True


def plot_ridgescrest_test_catalog():
    ridgecrest_evts = pd.read_csv(f"{get_data_dir()}/AML/ridgecrest_events.csv")
    rc_poly = get_ridgecrest_bounds()

    fig,ax = plt.subplots()
    rc_poly.plot(ax=ax, facecolor="None")
    ax.scatter(ridgecrest_evts.lon,ridgecrest_evts.lat, s=0.1)
    plt.savefig(f"{get_repo_dir()}/figures/Test_Ridgecrest_ev_cat.png",
                dpi=200,edgecolor="none",bbox_inches="tight")
    plt.close()
    print("Created plot")

    return True


def plot_picks_distributions():
    picks = load_north_california_picks()
    pick_data = picks[["p_phase_index","s_phase_index"]].to_numpy()

    fig,ax = plt.subplots(figsize=(5,4))
    ax.boxplot(pick_data)
    ax.set_xticks([y + 1 for y in range(pick_data.shape[1])], labels=['P', 'S'])
    ax.set_title("Picks distribution")
    plt.savefig(f"{get_repo_dir()}/figures/NC_events_picks distribution.png",
                dpi=200,edgecolor="none",bbox_inches="tight")
    plt.close()
    print("Created plot")

    return True
