#!/usr/bin/env python3
"""
Generate a mask of streamflow gauge catchments on a given grid.

Each grid cell is flagged with the integer gauge code of the catchment polygon
in which the cell center falls. Catchment geometries are read from a SQLite
database, gauge-to-catchment mappings from a JSON, and grid coordinates from a
NetCDF file. The result is written as a NetCDF with the same grid dimensions.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import cast
from uuid import UUID

import h5netcdf
import numpy as np
from shapely import wkb
from shapely.geometry import Point, Polygon

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_gauge_catchments(gauge_json_path: Path, geopackage_path: Path):
    """Load gauge definitions and corresponding catchment polygons."""
    # Read gauge info JSON
    with gauge_json_path.open("r") as f:
        gauge_info = json.load(f)

    # Read SQlite DB
    db: dict[UUID, Polygon] = {}
    with sqlite3.connect(geopackage_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT uuid, geom FROM watershed;")
        for row in cursor.fetchall():
            rid = UUID(row[0])
            blob = row[1]
            geom = cast(Polygon, wkb.loads(blob[40:]))
            db[rid] = geom

    gauge_polygons: dict[str, list[Polygon]] = {}
    for gauge_code, info in gauge_info.items():
        polys: list[Polygon] = []
        for rid in info.get("river", []):
            if UUID(rid) not in db:
                raise KeyError(f"No catchment found in DB for uuid {rid}")
            polys.append(db[UUID(rid)])
        gauge_polygons[str(gauge_code)] = polys
    logging.info(f"Loaded {len(gauge_polygons)} gauges with catchment polygons")
    return gauge_polygons


def load_grid(grid_nc_path: Path):
    """Load or build a 2D latitude and longitude grid from NetCDF input."""
    with h5netcdf.File(str(grid_nc_path), "r") as nc:
        lat = nc.variables["lat"][:]
        lon = nc.variables["lon"][:]

    # If lat/lon are 1D, build 2D meshgrid
    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
    elif lat.ndim == 2 and lon.ndim == 2:
        lat2d, lon2d = lat, lon
    else:
        raise ValueError(
            f"Unsupported lat/lon dimensions: lat.ndim={lat.ndim}, lon.ndim={lon.ndim}"
        )
    lat2d = cast(
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
        np.asarray(lat2d, dtype=np.float64),
    )
    lon2d = cast(
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
        np.asarray(lon2d, dtype=np.float64),
    )
    logging.info(f"Loaded grid with shape {lat2d.shape}")
    return lat2d, lon2d


def build_mask(
    lat2d: np.ndarray, lon2d: np.ndarray, gauge_polygons: dict[str, list[Polygon]]
):
    """Generate an integer mask array for each gauge catchment."""
    ny, nx = lat2d.shape
    gauges = list(gauge_polygons.keys())
    gauge2int = {g: i + 1 for i, g in enumerate(gauges)}  # start from 1

    mask = np.zeros((ny, nx), dtype=np.int32)

    # Iterate all grid cells
    for i in range(ny):
        if i % max(1, ny // 10) == 0:
            logging.info(f"Building mask: processing row {i + 1}/{ny}")
        for j in range(nx):
            if mask[i, j] != 0:
                continue  # already assigned
            pt = Point(float(lon2d[i, j]), float(lat2d[i, j]))
            # Test containment for each gauge
            for code, polys in gauge_polygons.items():
                for poly in polys:
                    if poly.contains(pt):
                        mask[i, j] = gauge2int[code]
                        break
                if mask[i, j] != 0:
                    break
    return gauges, mask


def write_mask_netcdf(
    out_path: Path,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    gauges: list[str],
    mask: np.ndarray,
):
    """Write latitude, longitude, and mask arrays to NetCDF output using h5netcdf."""
    ny, nx = lat2d.shape
    logging.info(f"Writing mask NetCDF to {out_path}, grid size {ny}x{nx}")
    with h5netcdf.File(str(out_path), "w") as ds:
        # dimensions
        ds.dimensions.update({"lat": ny, "lon": nx})

        # coordinate variables
        if np.allclose(lat2d[:, 0], lat2d[:, 1]):
            lat_var = ds.create_variable(
                "lat",
                ("lat",),
                np.float64,
                data=lat2d[:, 0],
                compression="gzip",
            )
        else:
            lat_var = ds.create_variable(
                "lat",
                ("lat", "lon"),
                np.float64,
                data=lat2d,
                compression="gzip",
            )
        lat_var.attrs.update(
            {
                "standard_name": np.bytes_("latitude", "ascii"),
                "units": np.bytes_("degrees_north", "ascii"),
                "axis": np.bytes_("Y", "ascii"),
            }
        )
        if np.allclose(lon2d[0, :], lon2d[1, :]):
            lon_var = ds.create_variable(
                "lon",
                ("lon",),
                np.float64,
                data=lon2d[0, :],
                compression="gzip",
            )
        else:
            lon_var = ds.create_variable(
                "lon",
                ("lat", "lon"),
                np.float64,
                data=lon2d,
                compression="gzip",
            )
        lon_var.attrs.update(
            {
                "standard_name": np.bytes_("longitude", "ascii"),
                "units": np.bytes_("degrees_east", "ascii"),
                "axis": np.bytes_("X", "ascii"),
            }
        )

        # mask variable
        mask_var = ds.create_variable(
            "gauge",
            ("lat", "lon"),
            np.int32,
            data=mask,
            fillvalue=0,
            compression="gzip",
        )
        mask_var.attrs.update(
            {
                "long_name": np.bytes_("Streamflow gauge catchment mask", "ascii"),
                "description": np.bytes_(
                    "The grid cells are flagged with the integer gauge code of the catchment "
                    "polygon in which the cell center falls. 0 indicates no catchment.",
                    "ascii",
                ),
                "flag_values": list(range(1, len(gauges) + 1)),
                "flag_meanings": np.bytes_(" ".join(gauges), "ascii"),
            }
        )


def parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description="Create a catchment mask for streamflow gauges on a grid."
    )
    p.add_argument(
        "gauge_json",
        type=Path,
        help="Path to JSON file mapping gauge codes to river UUID lists.",
    )
    p.add_argument(
        "catchment_db",
        type=Path,
        help="Path to SQLite database containing catchment geometries.",
    )
    p.add_argument(
        "grid_nc",
        type=Path,
        help="Path to NetCDF file with 'lat' and 'lon' variables for the grid.",
    )
    p.add_argument("out_nc", type=Path, help="Path for output NetCDF mask file.")
    return p.parse_args()


def main():
    args = parse_args()
    logging.info("Starting catchment mask creation")
    gauge_polys = load_gauge_catchments(args.gauge_json, args.catchment_db)
    lat2d, lon2d = load_grid(args.grid_nc)
    gauges, mask = build_mask(lat2d, lon2d, gauge_polys)
    write_mask_netcdf(args.out_nc, lat2d, lon2d, gauges, mask)
    logging.info("Finished catchment mask creation")


if __name__ == "__main__":
    main()
