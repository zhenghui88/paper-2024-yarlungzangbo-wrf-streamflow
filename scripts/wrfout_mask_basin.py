#!/usr/bin/env python3
"""
CLI tool to append a 'basin' mask variable to a NetCDF file based on
a catchment weight file in Feather format.

The 'basin' variable is 2D (y_dim, x_dim), with 1 indicating grid cells
inside the basin and 0 otherwise.
"""

import argparse
import itertools
import sys
from pathlib import Path

import h5netcdf
import numpy as np
import pyarrow.feather as feather


def parse_args():
    parser = argparse.ArgumentParser(
        description="Append a 'basin' mask to a NetCDF file."
    )
    parser.add_argument(
        "target",
        type=Path,
        help="Path to the NetCDF file to modify (opened in append mode).",
    )
    parser.add_argument(
        "weight_file",
        type=Path,
        help=(
            "Path to the Feather file containing 'lat_index' and 'lon_index' "
            "columns for basin cells."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    nc_path = args.target
    wt_path = args.weight_file

    if not nc_path.exists():
        print(f"ERROR: NetCDF file '{nc_path}' does not exist.", file=sys.stderr)
        sys.exit(1)
    if not wt_path.exists():
        print(f"ERROR: Weight file '{wt_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Read the weight table
    table = feather.read_table(wt_path)
    if "lat_index" not in table.column_names or "lon_index" not in table.column_names:
        print(
            "ERROR: Weight file must contain 'lat_index' and 'lon_index' columns.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Open dataset in append mode
    with h5netcdf.File(nc_path, "a") as ds:
        ny = ds.dimensions["lat"].size
        nx = ds.dimensions["lon"].size

        # Initialize mask array with zeros
        basin_mask = np.zeros((ny, nx), dtype=np.int8)

        # Set mask entries for each catchment cell
        for xi, yi in zip(
            itertools.chain.from_iterable(table["lon_index"]),
            itertools.chain.from_iterable(table["lat_index"]),
        ):
            basin_mask[int(yi), int(xi)] = 1

        # Create a DataArray for the basin mask
        basin_da = ds.create_variable(
            name="basin",
            dimensions=("lat", "lon"),
            dtype="i1",
            data=basin_mask,
            compression="gzip",
        )
        basin_da.attrs.update(
            {
                "long_name": np.bytes_("basin_mask", "ascii"),
                "flag_values": np.array([0, 1], dtype="i1"),
                "flag_meanings": np.bytes_("outside_basin inside_basin", "ascii"),
            }
        )

        print(f"Appended 'basin' mask to {nc_path}")


if __name__ == "__main__":
    main()
