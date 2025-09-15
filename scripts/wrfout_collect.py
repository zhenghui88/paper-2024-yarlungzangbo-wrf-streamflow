#!/usr/bin/env python3

"""
wrfout_collect.py

Collect hourly gridded data from a series of WRF ensemble netCDF files,
and write the data to a new netCDF file. The output file structure uses
the DIMINFO/VARINFO/FILEINFO dictionaries.

Usage:
    python wrfout_collect.py --wrfout-files wrfout1.nc wrfout2.nc ... --output wrf_experiment.nc
"""

from collections.abc import Iterator, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import h5netcdf
import numpy as np

# Spatial slices
latslice = slice(162, 287)  # 125 points
lonslice = slice(92, 487)  # 395 points


# Reference epoch for time variable
EPOCH = datetime(2000, 1, 1, tzinfo=UTC)

# Global file metadata
FILEINFO: dict[str, Any] = {
    "title": "Gridded data",
    "source": "WRF version 4.3.3",
    "author": "ZHENG, Hui",
    "contact": "zhenghui@tea.ac.cn",
    "history": f"Created on {datetime.now(UTC).isoformat(sep='T')}",
    "institution": "Institute of Atmospheric Physics, CAS",
    "comment": (
        f"range of latitude index in the original domain: [{latslice.start}-{latslice.stop - 1}]\n"
        f"range of longitude index in the original domain: [{lonslice.start}-{lonslice.stop - 1}]"
    ),
}

# Dimension definitions
DIMINFO: dict[str, int | None] = {
    "ensemble": None,
    "time": None,
    "lat": latslice.stop - latslice.start,
    "lon": lonslice.stop - lonslice.start,
    "soil_layer": 4,
    "bnd": 2,
}

# Variable definitions and metadata
VARINFO: dict[str, dict[str, Any]] = {
    "ensemble": {
        "dtype": "i4",
        "dims": ("ensemble",),
    },
    "time": {
        "units": f"seconds since {EPOCH.isoformat(sep='T')}",
        "dtype": "i8",
        "dims": ("time",),
        "bounds": "time_bnds",
        "axis": "T",
    },
    "time_bnds": {
        "units": f"seconds since {EPOCH.isoformat(sep='T')}",
        "dtype": "i8",
        "dims": ("time", "bnd"),
    },
    "lat": {
        "units": "degrees_north",
        "standard_name": "latitude",
        "dtype": "f8",
        "dims": ("lat",),
        "axis": "Y",
    },
    "lon": {
        "units": "degrees_east",
        "standard_name": "longitude",
        "dtype": "f8",
        "dims": ("lon",),
        "axis": "X",
    },
    "soil_layer": {
        "units": "m",
        "dtype": "f4",
        "dims": ("soil_layer",),
        "standard_name": "depth",
        "description": "soil layer center depth from the surface",
        "data": [0.05, 0.25, 0.7, 1.5],
        "axis": "Z",
    },
    # WRF output fields (no GPM any more)
    "pr": {
        "units": "kg m-2 s-1",
        "standard_name": "precipitation_flux",
        "description": "WRF simulated precipitation",
        "dtype": "f4",
        "dims": ("ensemble", "time", "lat", "lon"),
        "fillvalue": np.nan,
        "cell_methods": "time: mean",
    },
    "et": {
        "units": "kg m-2 s-1",
        "standard_name": "water_evapotranspiration_flux",
        "description": "WRF simulated evapotranspiration",
        "dtype": "f4",
        "dims": ("ensemble", "time", "lat", "lon"),
        "fillvalue": np.nan,
        "cell_methods": "time: mean",
    },
    "t": {
        "units": "K",
        "standard_name": "air_temperature",
        "description": "WRF simulated 2-meter air temperature",
        "dtype": "f4",
        "dims": ("ensemble", "time", "lat", "lon"),
        "fillvalue": np.nan,
        "cell_methods": "time: point",
    },
    "mrro": {
        "units": "kg m-2 s-1",
        "standard_name": "runoff_flux",
        "description": "WRF simulated runoff",
        "dtype": "f4",
        "dims": ("ensemble", "time", "lat", "lon"),
        "fillvalue": np.nan,
        "cell_methods": "time: mean",
    },
    "swe": {
        "units": "kg m-2",
        "standard_name": "surface_snow_amount",
        "description": "WRF simulated snow water equivalent",
        "dtype": "f4",
        "dims": ("ensemble", "time", "lat", "lon"),
        "fillvalue": np.nan,
        "cell_methods": "time: point",
    },
    "sm": {
        "units": "m3 m-3",
        "standard_name": "volume_fraction_of_water_in_soil",
        "description": "WRF simulated soil moisture",
        "dtype": "f4",
        "dims": ("ensemble", "time", "soil_layer", "lat", "lon"),
        "fillvalue": np.nan,
        "cell_methods": "time: point",
    },
}


def wrfout_iterator(
    wrfout_path: Path,
    lat_slice: slice | None = None,
    lon_slice: slice | None = None,
    time_start: datetime | None = None,
    time_stop: datetime | None = None,
) -> Iterator[tuple[datetime | None, datetime, Mapping[str, np.ndarray]]]:
    with h5netcdf.File(wrfout_path, "r") as f:
        tv = f.variables["time"]
        units = str(tv.attrs["units"])
        # parse units like "seconds since YYYY-MM-DDThh:mm:ss"
        base, _since, epoch_str = units.lower().partition(" since ")
        assert base.strip() == "seconds", f"unexpected time units {units}"
        epoch = datetime.fromisoformat(epoch_str.strip())
        if epoch.tzinfo is None:
            epoch = epoch.replace(tzinfo=UTC)
        assert "time" in f.dimensions, "no time dimension in WRF file"
        assert f.dimensions["time"].size > 1, "time dimension size <= 1"
        # iterate over time steps
        for ii in range(f.dimensions["time"].size):
            te = epoch + timedelta(seconds=float(tv[ii]))
            if ii == 0:
                tb = None
            else:
                tb = epoch + timedelta(seconds=float(tv[ii - 1]))
            if time_start is not None:
                if te < time_start:
                    continue
                if tb is None:
                    tb = time_start
                else:
                    tb = max(tb, time_start)
            if time_stop is not None:
                if tb is not None and tb > time_stop:
                    continue
                else:
                    if te > time_stop:
                        continue
                if te > time_stop:
                    te = time_stop

            if lat_slice is not None:
                lat_slice_ = lat_slice
            else:
                lat_slice_ = slice(None)
            if lon_slice is not None:
                lon_slice_ = lon_slice
            else:
                lon_slice_ = slice(None)
            data = {
                "pr": np.maximum(f.variables["RAINC"][ii, lat_slice_, lon_slice_], 0.0)
                + np.maximum(f.variables["RAINNC"][ii, lat_slice_, lon_slice_], 0.0),
                "mrro": np.maximum(
                    f.variables["SFROFF"][ii, lat_slice_, lon_slice_], 0.0
                )
                + np.maximum(f.variables["UDROFF"][ii, lat_slice_, lon_slice_], 0.0),
                "t": f.variables["T2"][ii, lat_slice_, lon_slice_],
                "et": np.maximum(f.variables["ECAN"][ii, lat_slice_, lon_slice_], 0.0)
                + np.maximum(f.variables["EDIR"][ii, lat_slice_, lon_slice_], 0.0)
                + np.maximum(f.variables["ETRAN"][ii, lat_slice_, lon_slice_], 0.0),
                "swe": f.variables["SNOW"][ii, lat_slice_, lon_slice_],
                "sm": f.variables["SMOIS"][ii, :, lat_slice_, lon_slice_],
            }
            yield tb, te, data


def collect_and_write(
    output_path: Path,
    wrfout_files_input: list[Path],
    time_start: datetime | None = None,
    time_stop: datetime | None = None,
):
    wrfout_files = sorted(wrfout_files_input, key=lambda p: p.stem)

    # open output file and define structure
    with h5netcdf.File(output_path, "w") as f:
        # global attributes
        for k, v in FILEINFO.items():
            f.attrs[k] = np.bytes_(v, "ascii") if isinstance(v, str) else v

        # dimensions
        f.dimensions.update(DIMINFO)

        # set unlimited dimensions
        if DIMINFO["ensemble"] is None:
            f.resize_dimension("ensemble", len(wrfout_files))

        # variables
        for name, info in VARINFO.items():
            var = f.create_variable(
                name,
                info["dims"],
                dtype=info["dtype"],
                fillvalue=info.get("fillvalue"),
                compression="gzip",
            )
            # assign metadata
            for attr, val in info.items():
                if attr in ("dims", "dtype", "fillvalue", "data"):
                    continue
                var.attrs[attr] = (
                    np.bytes_(val, "ascii") if isinstance(val, str) else val
                )
            # if static data is given
            if "data" in info:
                var[:] = np.array(info["data"], dtype=info["dtype"])

        # ensemble axis data
        f.variables["ensemble"][:] = (
            np.arange(len(wrfout_files), dtype=VARINFO["ensemble"]["dtype"]) + 1
        )

        # spatial coordinates: assume existence in the first file
        with h5netcdf.File(wrfout_files[0], "r") as ref:
            lat0 = np.array(
                ref.variables["XLAT"][latslice, (lonslice.start + lonslice.stop) // 2],
                dtype=VARINFO["lat"]["dtype"],
            )
            lon0 = np.array(
                ref.variables["XLONG"][(latslice.stop + latslice.start) // 2, lonslice],
                dtype=VARINFO["lon"]["dtype"],
            )
        f.variables["lat"][:] = lat0
        f.variables["lon"][:] = lon0

        # iterate over time and ensemble files
        wrf_data_iterators = [
            wrfout_iterator(x, latslice, lonslice, time_start, time_stop)
            for x in sorted(wrfout_files)
        ]
        for it, members in enumerate(zip(*wrf_data_iterators)):
            for imember, member in enumerate(members):
                tb, te, data = member
                if imember == 0:
                    if tb is None:  # fix me: first time step default to 1 hour
                        tb = te - timedelta(hours=1)
                    # first ensemble, need to append time steps
                    f.resize_dimension("time", it + 1)
                    f.variables["time"][it] = int((te - EPOCH).total_seconds())
                    f.variables["time_bnds"][it, :] = (
                        int((tb - EPOCH).total_seconds()),
                        int((te - EPOCH).total_seconds()),
                    )
                # write data for this ensemble and time step
                for varname, vardata in data.items():
                    var = f.variables[varname]
                    var[imember, it, ...] = vardata


def parse_datetime_arg(s: str) -> datetime:
    t = datetime.fromisoformat(s.strip())
    if t.tzinfo is None:
        t = t.replace(tzinfo=UTC)
    else:
        t = t.astimezone(UTC)
    return t


def parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Collect WRF ensemble outputs")
    p.add_argument(
        "--wrfout-files",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to the WRF ensemble netCDF files",
    )
    p.add_argument(
        "--start",
        type=parse_datetime_arg,
        default=None,
        help="Start datetime in ISO format, e.g., 2020-06-01T00:00:00",
    )
    p.add_argument(
        "--stop",
        type=parse_datetime_arg,
        default=None,
        help="Stop datetime in ISO format, e.g., 2020-07-01T00:00:00",
    )
    p.add_argument("--output", type=Path, required=True, help="Output netCDF file path")
    return p.parse_args()


def main():
    args = parse_args()
    output_file = Path(args.output)
    wrfout_files = cast(list[Path], args.wrfout_files)
    time_start = cast(datetime | None, args.start)
    time_stop = cast(datetime | None, args.stop)
    collect_and_write(output_file, wrfout_files, time_start, time_stop)


if __name__ == "__main__":
    main()
