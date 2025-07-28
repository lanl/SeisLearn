import numpy as np
import pandas as pd
import pygmt
import xarray as xr
from scipy.ndimage import gaussian_filter

from seisnet.utils import (get_data_dir, get_repo_dir,
                           load_picks_without_ridgecrest)


def plot_all_datasets():
    """
    Plot all the north california events and the test dataset points
    """
    picks = load_picks_without_ridgecrest(f"{get_data_dir()}/metadata/picks.csv")
    coords = picks[["longitude","latitude"]].to_numpy()
    px,py = coords[:,0],coords[:,1]

    heatmap, xedges, yedges = np.histogram2d(px, py, bins=100)
    heatmap = gaussian_filter(heatmap, sigma=2)
    heatmap = np.where(heatmap>50,heatmap,np.nan)

    grid = xr.DataArray(heatmap.T,
        coords=[yedges[:-1], xedges[:-1]],
        dims=["lat", "lon"], name="z"
    )

    region = [-157, -110, 18, 46]

    topo = pygmt.datasets.load_earth_relief(resolution="30s", region=region)

    fig = pygmt.Figure()
    # define figure configuration
    pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain")

    # --------------- plotting the original Data Elevation Model -----------
    prj = "M18c"
    pygmt.makecpt(cmap="geo", series=[-6200, 8000, 10])
    fig.grdimage(region=region,grid=topo, projection=prj, frame=["lbrt", "xaf", "yaf"], cmap=True, transparency=15)
    fig.coast(region=region,projection=prj,shorelines="1/1p,gray50",borders=["1/1p,black","2/1p,black"],
            dcw=["US.CA+p0.75p,black","US.AZ+p0.75p,black"])
    pygmt.makecpt(cmap="haxby", series=[100, 9000, 10], reverse=True, truncate=[0.15,0.85])
    fig.grdimage(region=region,projection=prj,grid=grid, nan_transparent=True, transparency=25)
    fig.plot(region=region,projection=prj,x=px[::10],y=py[::10],style="c0.025c",fill="black")

    fig.plot(x=-155.275167, y=19.407667, style="c1c", pen="2p,white")
    fig.plot(x=-117.570667, y=35.747667, style="c1c", pen="2p,white")
    fig.plot(x=-111.024833, y=44.783, style="c1c", pen="2p,white")
    fig.savefig(f"{get_repo_dir()}/figures/datasets_bmp.png")
    
    return fig


def plot_hawaii_data():
    """
    Plot the Hawaii dataset
    """
    hawaii = pd.read_csv(f"{get_data_dir()}/AML/hawaii_events.csv")

    fig = pygmt.Figure()
    rgn1 = "g"
    prj1 = "G-155.25/19.4/0.2/6c"# Orthographic projection lon0/lat0/horizon/width
    hwgrid = pygmt.datasets.load_earth_relief(resolution="01s", region=[-155.8, -155, 19.2, 20.5])
    fig.grdimage(grid=hwgrid,projection=prj1,frame="afg",cmap="geo", transparency=20)
    fig.coast(region=rgn1, projection=prj1,land="skyblue")
    fig.plot(region=rgn1, projection=prj1, x=hawaii["lon"],y=hawaii["lat"],style="c0.025c",fill="black")
    fig.savefig(f"{get_repo_dir()}/figures/hawaii_bmp.png",transparent=True)

    return fig


def plot_ridgecrest_data():
    """
    Plot the Ridgecrest dataset
    """
    ridgecrest = pd.read_csv(f"{get_data_dir()}/AML/ridgecrest_events.csv")

    fig = pygmt.Figure()
    rgn2 = "g"
    prj2 = "G-117.57/35.74/0.35/6c"# Orthographic projection lon0/lat0/horizon/width
    rcgrid = pygmt.datasets.load_earth_relief(resolution="01s", region=[-118.1, -117.1, 35.3, 36.2])
    fig.grdimage(grid=rcgrid,projection=prj2,frame="afg",cmap="geo", transparency=20)
    fig.plot(region=rgn2, projection=prj2, x=ridgecrest["lon"],y=ridgecrest["lat"],style="c0.012c",fill="black")
    fig.savefig(f"{get_repo_dir()}/figures/ridgecrest_bmp.png",transparent=True)

    return fig


def plot_yellowstone_data():
    """
    Plot the Yellowstone dataset
    """
    yellowstone = pd.read_csv(f"{get_data_dir()}/AML/yellowstone_events.csv")

    fig = pygmt.Figure()
    rgn3 = "g"
    prj3 = "G-111.02/44.78/0.12/6c"# Orthographic projection lon0/lat0/horizon/width
    ylgrid = pygmt.datasets.load_earth_relief(resolution="01s", region=[-111.25, -110.7, 44.5, 44.95])
    fig.grdimage(grid=ylgrid,projection=prj3,frame="afg",cmap="geo", transparency=20)
    fig.plot(region=rgn3, projection=prj3, x=yellowstone["lon"],y=yellowstone["lat"],style="c0.025c",fill="black")
    fig.savefig(f"{get_repo_dir()}/figures/yellowstone_bmp.png",transparent=True)

    return fig