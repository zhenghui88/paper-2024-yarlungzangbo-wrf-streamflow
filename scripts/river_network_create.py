import math
from pathlib import Path
from uuid import UUID

import h5netcdf
import h5py
import numpy as np
import polars as pl
from numpy.typing import NDArray

EARTH_RADIUS = 6371.001e3
EARTH_RADIUS2 = EARTH_RADIUS**2
DLAT = 1.0 / 60 / 20
DLON = 1.0 / 60 / 20
DLAT_RAD = math.radians(DLAT)
DLON_RAD = math.radians(DLON)

DATAROOT = Path("data")
RIVER_GEOM_FILE: Path = DATAROOT.joinpath("src", "yaluzangbu.h5")  # input

# GEOEM_FILE = DATAROOT.joinpath("wrf_domain.nc")  # input
# RIVNET_FILE = DATAROOT.joinpath("rivnet.parquet")  # output
# WEIGHT_FILE = DATAROOT.joinpath("rivnet_weight_domain.parquet")  # output

GEOEM_FILE = DATAROOT.joinpath("wrf_experiment.nc")  # input
RIVNET_FILE = DATAROOT.joinpath("rivnet.parquet")  # output
WEIGHT_FILE = DATAROOT.joinpath("rivnet_weight_experiment.parquet")  # output


with h5netcdf.File(GEOEM_FILE, "r") as f:
    lon = np.array(f.variables["lon"][:], dtype=np.float64)
    lat = np.array(f.variables["lat"][:], dtype=np.float64)


def read_centerline(riverfile: Path):
    centerline: dict[UUID, tuple[list[float], list[float]]] = {}
    downstream: dict[UUID, UUID | None] = {}
    with h5py.File(riverfile, "r") as f:
        latgrid = np.array(f["grid/latitude"][:], dtype=np.float64)  # type: ignore
        longrid = np.array(f["grid/longitude"][:], dtype=np.float64)  # type: ignore
        for uuidbyte, geom, didbyte in zip(
            f["reach/UUID"][:],  # type: ignore
            f["reach/centerline"][:],  # type: ignore
            f["reach/downstream"][:],  # type: ignore
        ):
            uuid = UUID(bytes=uuidbyte.tobytes())
            downstream[uuid] = None if didbyte.tobytes() == b"\x00" * 16 else UUID(bytes=didbyte.tobytes())
            lats: list[float] = []
            lons: list[float] = []
            for ilat, ilon in geom:
                lats.append(float(latgrid[ilat]))
                lons.append(float(longrid[ilon]))
            centerline[uuid] = (lats, lons)
    return centerline, downstream


def read_catchment(riverfile: Path, lat: NDArray[np.floating], lon: NDArray[np.floating]):
    catchment: dict[UUID, tuple[list[tuple[int, int]], list[float]]] = {}
    ilatlonarea: dict[tuple[int, int], float] = {}
    with h5py.File(riverfile, "r") as f:
        latgrid = np.array(f["grid/latitude"][:], dtype=np.float64)  # type: ignore
        longrid = np.array(f["grid/longitude"][:], dtype=np.float64)  # type: ignore
        finelat2index: dict[float, int] = {}
        finelon2index: dict[float, int] = {}
        for uuidbyte, inner in zip(f["catchment/UUID"][:], f["catchment/inner"][:]):  # type: ignore
            uuid = UUID(bytes=uuidbyte.tobytes())
            inner_set = {(int(x[0]), int(x[1])) for x in inner}
            ilatlonarea.clear()
            for latidx, lonidx in inner_set:
                finelat = float(latgrid[latidx])
                finelon = float(longrid[lonidx])
                if finelat in finelat2index:
                    ilat = finelat2index[finelat]
                else:
                    ilat = int(np.argmin(np.abs(lat - finelat)))
                    finelat2index[finelat] = ilat
                if finelon in finelon2index:
                    ilon = finelon2index[finelon]
                else:
                    ilon = int(np.argmin(np.abs(lon - finelon)))
                    finelon2index[finelon] = ilon
                ilatlonarea[(ilat, ilon)] = (
                    ilatlonarea.get((ilat, ilon), 0)
                    + EARTH_RADIUS2 * math.cos(math.radians(finelat)) * DLAT_RAD * DLON_RAD
                )
            catchment[uuid] = ([], [])
            for ij in sorted(ilatlonarea.keys()):
                catchment[uuid][0].append(ij)
                catchment[uuid][1].append(ilatlonarea[ij])
    return catchment


def find_subnet(downstream: dict[UUID, UUID | None], outlet: UUID) -> dict[UUID, UUID | None]:
    subnet = {}

    upstreams = find_upstream(downstream)

    todo = [outlet]
    while todo:
        cid = todo.pop()
        subnet[cid] = downstream[cid]
        todo.extend(upstreams[cid])
    subnet[outlet] = None
    return subnet


def find_upstream(downstream: dict[UUID, UUID | None]) -> dict[UUID, list[UUID]]:
    upstream: dict[UUID, list[UUID]] = {k: [] for k in downstream.keys()}
    for uid, did in downstream.items():
        if did is not None and did in upstream:
            upstream[did].append(uid)
    return upstream


def calculate_zheng_order_reversed(
    downstream: dict[UUID, UUID | None], upstream: dict[UUID, list[UUID]] | None = None
) -> dict[UUID, int]:
    order: dict[UUID, int] = {}
    upstreams: dict[UUID, list[UUID]] = upstream if upstream is not None else find_upstream(downstream)
    # the downmost rivers have order 1
    current = set(k for k, v in downstream.items() if (v is None or v not in upstreams))
    for k in current:
        order[k] = 1
    # the rest of the rivers
    prev, current = current, set()
    while len(order) < len(downstream):
        current.clear()
        for cid in prev:
            uids = upstreams[cid]
            for uid in uids:
                current.add(uid)
        for k in current:
            did = downstream[k]
            if did is not None:
                order[k] = order[did] + 1
        prev, current = current, prev

    # reverse the order
    max_order = max(order.values())
    for k, v in order.items():
        order[k] = max_order - v + 1
    return order


def spherical_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1rad = math.radians(lat1)
    lon1rad = math.radians(lon1)
    lat2rad = math.radians(lat2)
    lon2rad = math.radians(lon2)
    dot = math.cos(lat1rad) * math.cos(lat2rad) * math.cos(lon2rad - lon1rad) + math.sin(lat1rad) * math.sin(lat2rad)
    return EARTH_RADIUS * math.acos(dot)


centerline, downstream = read_centerline(RIVER_GEOM_FILE)
catchment = read_catchment(RIVER_GEOM_FILE, lat, lon)

length: dict[UUID, float] = {}
for rid, (lats, lons) in centerline.items():
    did = downstream[rid]
    if did is not None:
        lat1 = centerline[did][0][-1]
        lon1 = centerline[did][1][-1]
        length[rid] = spherical_distance(lats[0], lons[0], lat1, lon1)
    else:
        length[rid] = 0.0
    for i in range(len(lats) - 1):
        length[rid] += spherical_distance(lats[i], lons[i], lats[i + 1], lons[i + 1])


# gauges
gauge = pl.read_csv(DATAROOT.joinpath("gauge_q_loc.csv"))

# Nuxia
nx = find_subnet(downstream, UUID(gauge.filter(pl.col("gauge") == "Nuxia").select(pl.col("reach")).item()))

# Yangcun
yc = set(
    find_subnet(downstream, UUID(gauge.filter(pl.col("gauge") == "Yangcun").select(pl.col("reach")).item())).keys()
)

# Nugesha
ngs = set(
    find_subnet(downstream, UUID(gauge.filter(pl.col("gauge") == "Nugesha").select(pl.col("reach")).item())).keys()
)

# Lazi
lz = set(find_subnet(downstream, UUID(gauge.filter(pl.col("gauge") == "Lazi").select(pl.col("reach")).item())).keys())

zhengorder = calculate_zheng_order_reversed(nx)

# sorted river ids
riverids = sorted(nx.keys(), key=lambda k: (zhengorder[k], k.int))

col_uuid: list[str] = []
col_index: list[int] = []
col_order: list[int] = []
col_toindex: list[int | None] = []
col_subnet: list[str] = []
col_lat: list[list[float]] = []
col_lon: list[list[float]] = []
col_length: list[float] = []
col_latidx: list[list[int]] = []
col_lonidx: list[list[int]] = []
col_area: list[list[float]] = []

riverid2index = {k: i for i, k in enumerate(riverids)}

for rid in riverids:
    col_uuid.append(rid.urn)
    col_index.append(riverid2index[rid])
    col_order.append(zhengorder[rid])
    did = nx[rid]
    if rid in lz:
        col_subnet.append("Lazi")
    elif rid in ngs:
        col_subnet.append("Nugesha")
    elif rid in yc:
        col_subnet.append("Yangcun")
    else:
        col_subnet.append("Nuxia")
    col_toindex.append(None if did is None else riverid2index[did])
    lats, lons = centerline[rid]
    col_lat.append(lats)
    col_lon.append(lons)
    col_length.append(length[rid])
    latlonidx = catchment[rid][0]
    col_latidx.append([i for i, _ in latlonidx])
    col_lonidx.append([j for _, j in latlonidx])
    col_area.append(catchment[rid][1])

pl.DataFrame(
    [
        pl.Series("uuid", col_uuid, dtype=pl.String),
        pl.Series("index", col_index, dtype=pl.UInt64),
        pl.Series("order", col_order, dtype=pl.UInt32),
        pl.Series("to", col_toindex, dtype=pl.UInt64),
        pl.Series("length", col_length, dtype=pl.Float32),
        pl.Series("gauge", col_subnet, dtype=pl.String),
        pl.Series("lat", col_lat, dtype=pl.List(pl.Float64)),
        pl.Series("lon", col_lon, dtype=pl.List(pl.Float64)),
    ]
).write_parquet(RIVNET_FILE)

pl.DataFrame(
    [
        pl.Series("uuid", col_uuid, dtype=pl.String),
        pl.Series("gauge", col_subnet, dtype=pl.String),
        pl.Series("latidx", col_latidx, dtype=pl.List(pl.UInt64)),
        pl.Series("lonidx", col_lonidx, dtype=pl.List(pl.UInt64)),
        pl.Series("area", col_area, dtype=pl.List(pl.Float64)),
    ]
).write_parquet(WEIGHT_FILE)
