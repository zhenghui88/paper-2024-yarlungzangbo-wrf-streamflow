#!/usr/bin/env python3
"""
lateralflow_create.py

Generate an ensemble of lateral flow (qlat) time series from a collected WRF ensemble NetCDF
and a catchment weight file. Outputs one Parquet file per ensemble member, named qlat_xx.parquet.

Each output Parquet has two columns:
  - id: UUID of catchment (from weight file)
  - qlat: list of floats, the time series of lateral flow for that catchment & ensemble member
"""

import argparse
import concurrent.futures
import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterator, cast

import h5netcdf
import numpy as np
import pyarrow
import pyarrow.feather
import pyarrow.parquet

logger = logging.getLogger(__name__)
RHO_WATER = 1000.0  # kg m-3


def mrro_time_ensemble_iterator(
    wrfout_path: Path, ensemble: int
) -> Iterator[tuple[datetime, np.ndarray[tuple[int, int], np.dtype[np.float32]]]]:
    """
    Iterate over time steps in a WRF-collect NetCDF.
    Yield (datetime, [mrro_2d for each ensemble member]).
    """
    with h5netcdf.File(wrfout_path, "r") as f:
        tv = f.variables["time"]
        units = str(tv.attrs["units"])
        # parse units like "seconds since YYYY-MM-DDThh:mm:ss"
        base, _, epoch_str = units.lower().partition(" since ")
        assert base.strip() == "seconds", f"unexpected time units {units}"
        epoch = datetime.fromisoformat(epoch_str.strip())
        if epoch.tzinfo is None:
            epoch = epoch.replace(tzinfo=UTC)
        assert "time" in f.dimensions, "no time dimension in WRF file"
        assert f.dimensions["time"].size > 1, "time dimension size <= 1"
        e_idx = [int(x) for x in f.variables["ensemble"][:]].index(ensemble)
        for t_idx in range(f.dimensions["time"].size):
            t_sec = tv[t_idx]
            t_dt = epoch + timedelta(seconds=float(t_sec))
            logger.debug(f"Processing time step: {t_dt}")
            mrro = f.variables["mrro"][e_idx, t_idx, :, :].astype(np.float32)
            mrro = cast(np.ndarray[tuple[int, int], np.dtype[np.float32]], mrro)
            yield t_dt, mrro


def get_ensemble_ids(wrfout_path: Path) -> list[int]:
    """Retrieve ensemble member IDs from WRF output file."""
    with h5netcdf.File(str(wrfout_path), "r") as f:
        return [int(x) for x in f.variables["ensemble"][:]]


def load_weights(weight_file: Path):
    """Load catchment weights from Feather file."""
    wt = pyarrow.feather.read_table(
        weight_file, columns=["id", "area", "lat_index", "lon_index"]
    )
    ids = wt["id"].to_pylist()
    areas_list = [np.array(x, dtype=np.float64) for x in wt["area"].to_pylist()]
    latinds_list = [np.array(x, dtype=np.int_) for x in wt["lat_index"].to_pylist()]
    loninds_list = [np.array(x, dtype=np.int_) for x in wt["lon_index"].to_pylist()]
    return ids, areas_list, latinds_list, loninds_list


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate lateral flow ensemble (qlat) from WRF output and catchment weights"
    )
    p.add_argument(
        "output_root",
        type=Path,
        help="Directory to write Parquet outputs (will be created if needed)",
    )
    p.add_argument(
        "wrfout_file",
        type=Path,
        help="Path to collected WRF ensemble NetCDF file (from wrfout_collect.py)",
    )
    p.add_argument(
        "weight_file",
        type=Path,
        help="Path to Feather file with columns id, area, lat_index, lon_index",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # logging already configured at module import

    # validate paths
    if not args.wrfout_file.exists():
        logger.error(f"WRF output file not found: {args.wrfout_file}")
        sys.exit(1)
    if not args.weight_file.exists():
        logger.error(f"Weight file not found: {args.weight_file}")
        sys.exit(1)

    # prepare output directory
    args.output_root.mkdir(parents=True, exist_ok=True)

    # Discover ensemble IDs
    ens_values = get_ensemble_ids(args.wrfout_file)

    # Load catchment weights
    ids, areas_list, latinds_list, loninds_list = load_weights(args.weight_file)

    # Function to process one ensemble member: collect qlat time series and write Parquet
    def process_member(ensemble: int):
        # accumulate timestamps and qlat per time step
        ts_list: list[datetime] = []
        qlat_list: list[list[float]] = []
        for dt, mrro in mrro_time_ensemble_iterator(args.wrfout_file, ensemble):
            print(dt)
            ts_list.append(dt)
            cqlat = [
                float(np.dot(a, mrro[li, lj])) / RHO_WATER
                for a, li, lj in zip(areas_list, latinds_list, loninds_list)
            ]
            qlat_list.append(cqlat)

        # build columns: id plus each time step
        names = ["id"]
        data = [pyarrow.array([x.bytes for x in ids], type=pyarrow.uuid())]
        for t_val, q_vals in zip(ts_list, qlat_list):
            names.append(t_val.isoformat(sep="T"))
            data.append(pyarrow.array(q_vals, type=pyarrow.float32()))

        table = pyarrow.table(data, names=names)
        out_file = args.output_root / f"qlat_{ensemble:02d}.parquet"
        pyarrow.parquet.write_table(table, out_file)
        logger.info(f"Wrote ensemble member {ensemble:02d} to {out_file}")

    # Run processing in parallel over all ensemble members
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(process_member, ens_values)


if __name__ == "__main__":
    main()
