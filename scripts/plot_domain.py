# %%
import json
import sqlite3
from pathlib import Path
from typing import TypedDict, cast
from uuid import UUID

import h5netcdf
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import shapely
import shapely.wkb

# %%
# plotting settings
DATAROOT = Path("data")
FIGROOT = Path("fig")
FIGDPI = 600
FIGSUFFIX = ".pdf"
# FIGSUFFIX = ".svg"

plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update(
    {
        "savefig.pad_inches": 0.05 / 2.54,
        "grid.linewidth": 0.25,
        "axes.titlesize": 8,
        "axes.titlepad": 0.05,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }
)

# %% data paths
GEOEM_FILE = DATAROOT.joinpath("wrf_domain.nc")

WEIGHT_FILE = DATAROOT.joinpath("river_network/river_network.gpkg")

BASIN_BOUNDARY_FILE = DATAROOT.joinpath("yj_boundary.geojson")

GAUGE_TOPO_FILE = DATAROOT.joinpath("gauge_network.json")
GAUGE_LOCATION_FILE = DATAROOT.joinpath("gauge_q_loc.csv")

# %%
with h5netcdf.File(GEOEM_FILE, "r") as f:
    lon = np.array(f.variables["lon"][:], dtype=np.float64)
    lat = np.array(f.variables["lat"][:], dtype=np.float64)
    hgt = np.array(f.variables["HGT"][:, :], dtype=np.float32)

# %%
rivernet: dict[UUID, shapely.LineString] = {}
with sqlite3.connect(WEIGHT_FILE) as conn:
    cursor = conn.cursor()
    cursor.execute("select UUID, geom from centerline")
    for rid, geom in cursor.fetchall():
        line = cast(shapely.LineString, shapely.wkb.loads(geom[40:]))
        rivernet[UUID(rid)] = line
print(f"There are {len(rivernet)} rivers in the GeoPackage file.")

# %%
with open(BASIN_BOUNDARY_FILE, "rt") as f:
    _basin_boundary = json.load(f)["features"][0]["geometry"]["coordinates"][0][0]
    basin_boundary = ([_[1] for _ in _basin_boundary], [_[0] for _ in _basin_boundary])

# %%
gauge_location = pl.read_csv(GAUGE_LOCATION_FILE, schema_overrides={"stcd": pl.Utf8})


class GaugeInfo(TypedDict):
    name: str
    latitude: float
    longitude: float
    upstream: set[str]
    downstream: set[str]
    rivers: set[UUID]


gauge_topo: dict[str, GaugeInfo] = {}
with open(GAUGE_TOPO_FILE) as f:
    _gauge_topo = json.load(f)
    for gauge, info in _gauge_topo.items():
        gauge_topo[gauge] = GaugeInfo(
            name=str(gauge_location.filter(pl.col("stcd") == gauge)[0, "gauge"]),
            latitude=float(gauge_location.filter(pl.col("stcd") == gauge)[0, "lat"]),
            longitude=float(gauge_location.filter(pl.col("stcd") == gauge)[0, "lon"]),
            upstream=set(info["upstream"]),
            downstream=set(info["downstream"]),
            rivers={UUID(rid) for rid in info["river"]},
        )
print(
    f"there are {sum(len(info['rivers']) for info in gauge_topo.values())} rivers in the gauge network."
)
# %%
coast = []
with open(DATAROOT.joinpath("coastline.geojsonl")) as f:
    for line in f:
        lines = json.loads(line)["geometry"]["coordinates"]
        for ll in lines:
            clat = []
            clon = []
            for lll in ll:
                clat.append(lll[1])
                clon.append(lll[0])
            coast.append((clat, clon))

# %%
gaugeshortname = {
    "90604500": "NX",
    "90603000": "YC",
    "90602000": "NGS",
    "90601000": "LZ",
    "90802500": "LS",
    "90901200": "GZ",
}
# gaugeshortname = {"Nuxia": "奴下", "Yangcun": "杨村", "Nugesha":"努各沙", "Lazi":"拉孜"}[gaugename]

# %%
# plotting
fig = plt.figure(figsize=(8 / 2.54, 6 / 2.54), dpi=FIGDPI, layout="constrained")
ax = fig.add_subplot(111)
ax.plot(basin_boundary[1], basin_boundary[0], color="k", linewidth=0.5)
cf = ax.contourf(lon, lat, hgt)
for gaugecode, gaugeinfo in gauge_topo.items():
    match gaugeinfo["name"]:
        case "Lazi":
            color = "r"
            textva = "bottom"
        case "Nugesha":
            color = "y"
            textva = "bottom"
        case "Yangcun":
            color = "b"
            textva = "top"
        case "Lasa":
            color = "r"
            textva = "bottom"
        case "Gengzhang":
            color = "b"
            textva = "bottom"
        case _:
            color = "w"
            textva = "top"
    for rid in gaugeinfo["rivers"]:
        if rid in rivernet:
            line = rivernet[rid]
            x, y = line.xy
            ax.plot(x, y, color=color, linewidth=0.15)
    ax.plot(gaugeinfo["longitude"], gaugeinfo["latitude"], "ro", markersize=1)
    ax.text(
        gaugeinfo["longitude"],
        gaugeinfo["latitude"],
        gaugeshortname[gaugecode],
        fontsize=7,
        ha="right",
        va=textva,
    )

cb = plt.colorbar(
    ax=ax, mappable=cf, orientation="horizontal", aspect=40, pad=0.02, shrink=0.96
)
cb.set_label("Elevation (m)", fontsize=8)
ax.set_aspect("equal")
ax.set_xticks([80, 85, 90, 95, 100])
ax.set_xticklabels(["80°E", "85°E", "90°E", "95°E", "100°E"])
ax.set_yticks([24, 26, 28, 30, 32])
ax.set_yticklabels(["24°N", "26°N", "28°N", "30°N", "32°N"])

lat_min, lat_max = lat.min(), lat.max()
lon_min, lon_max = lon.min(), lon.max()

subax = fig.add_axes(
    rect=(0.15, 0.30, 0.20, 0.20),
    frameon=True,
    aspect=1,
)
for clat, clon in coast:
    subax.plot(clon, clat, color="k", linestyle="--", linewidth=0.5)
subax.plot(
    [lon_min, lon_max, lon_max, lon_min, lon_min],
    [lat_min, lat_min, lat_max, lat_max, lat_min],
    color="k",
    linewidth=0.5,
)
subax.set_xlim(50, 130)
subax.set_ylim(0, 50)
subax.set_xticks([])
subax.set_yticks([])

fig.savefig(
    FIGROOT.joinpath("domain").with_suffix(FIGSUFFIX).as_posix(), bbox_inches="tight"
)


# %%
