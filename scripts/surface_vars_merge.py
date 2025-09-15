import argparse
from datetime import datetime
from pathlib import Path

import h5netcdf
import numpy as np

DATETIMEEPOCH = datetime(2000, 1, 1)


def merge_surface_vars(input_files, output_file):
    with h5netcdf.File(output_file, "w") as dst:
        for isrc, srcfile in enumerate(input_files):
            with h5netcdf.File(srcfile, "r") as src:
                if isrc == 0:
                    # Create the root group and dimensions only once
                    dst.dimensions.update(
                        {
                            "time": None,
                            "south_north": src.dimensions["south_north"].size,
                            "west_east": src.dimensions["west_east"].size,
                        }
                    )
                    # create time, XLAT, and XLONG variables
                    dst.create_variable(
                        "time",
                        ("time",),
                        dtype=np.float64,
                        compression="gzip",
                        compression_opts=9,
                    ).attrs.update(
                        {
                            np.bytes_(k, "ascii"): np.bytes_(v, "ascii")
                            for k, v in {
                                "standard_name": "time",
                                "units": f"seconds since {DATETIMEEPOCH.isoformat(sep='T')}Z",
                            }.items()
                        }
                    )
                    dst.create_variable(
                        "XLAT",
                        ("south_north", "west_east"),
                        dtype=np.float64,
                        data=src["XLAT"][:],
                        compression="gzip",
                        compression_opts=9,
                    ).attrs.update(
                        {
                            np.bytes_(k, "ascii"): np.bytes_(v, "ascii")
                            for k, v in {
                                "standard_name": "latitude",
                                "units": "degrees_north",
                            }.items()
                        }
                    )
                    dst.create_variable(
                        "XLONG",
                        ("south_north", "west_east"),
                        dtype=np.float64,
                        data=src["XLONG"][:],
                        compression="gzip",
                        compression_opts=9,
                    ).attrs.update(
                        {
                            np.bytes_(k, "ascii"): np.bytes_(v, "ascii")
                            for k, v in {
                                "standard_name": "longitude",
                                "units": "degrees_east",
                            }.items()
                        }
                    )
                    # create the rest of the variables
                    vars_info = {
                        "U10": {
                            "dims": ("time", "south_north", "west_east"),
                            "dtype": np.float32,
                            "units": "m s-1",
                            "description": "eastward wind component at 10 m",
                        },
                        "V10": {
                            "dims": ("time", "south_north", "west_east"),
                            "dtype": np.float32,
                            "units": "m s-1",
                            "description": "northward wind component at 10 m",
                        },
                        "T2": {
                            "dims": ("time", "south_north", "west_east"),
                            "dtype": np.float32,
                            "units": "K",
                            "description": "temperature at 2 m",
                        },
                        "Q2": {
                            "dims": ("time", "south_north", "west_east"),
                            "dtype": np.float32,
                            "units": "kg kg-1",
                            "description": "mixing ratio at 2 m",
                        },
                        "PSFC": {
                            "dims": ("time", "south_north", "west_east"),
                            "dtype": np.float32,
                            "units": "Pa",
                            "description": "surface pressure",
                        },
                        "SWDOWN": {
                            "dims": ("time", "south_north", "west_east"),
                            "dtype": np.float32,
                            "units": "W m-2",
                            "description": "downward shortwave radiation at the surface",
                        },
                        "GLW": {
                            "dims": ("time", "south_north", "west_east"),
                            "dtype": np.float32,
                            "units": "W m-2",
                            "description": "downward longwave radiation at the surface",
                        },
                        "RAINC": {
                            "dims": ("time", "south_north", "west_east"),
                            "dtype": np.float32,
                            "units": "kg m-2 s-1",
                            "description": "total cumulus precpitation rate",
                        },
                        "RAINNC": {
                            "dims": ("time", "south_north", "west_east"),
                            "dtype": np.float32,
                            "units": "kg m-2 s-1",
                            "description": "total grid scale precipitation rate",
                        },
                        "SNOWNC": {
                            "dims": ("time", "south_north", "west_east"),
                            "dtype": np.float32,
                            "units": "kg m-2 s-1",
                            "description": "total grid scale snow and ice rate",
                        },
                        "GRAUPELNC": {
                            "dims": ("time", "south_north", "west_east"),
                            "dtype": np.float32,
                            "units": "kg m-2 s-1",
                            "description": "total grid scale graupel rate",
                        },
                        "HAILNC": {
                            "dims": ("time", "south_north", "west_east"),
                            "dtype": np.float32,
                            "units": "kg m-2 s-1",
                            "description": "total grid scale hail rate",
                        },
                        "SFROFF": {
                            "dims": ("time", "south_north", "west_east"),
                            "dtype": np.float32,
                            "units": "kg m-2 s-1",
                            "description": "surface runoff",
                        },
                        "UDROFF": {
                            "dims": ("time", "south_north", "west_east"),
                            "dtype": np.float32,
                            "units": "kg m-2 s-1",
                            "description": "subsurface runoff",
                        },
                    }

                    for var, info in vars_info.items():
                        dst.create_variable(
                            var,
                            info["dims"],
                            dtype=info["dtype"],
                            compression="gzip",
                            compression_opts=9,
                        ).attrs.update(
                            {
                                np.bytes_(k, "ascii"): np.bytes_(v, "ascii")
                                for k, v in {
                                    "units": info["units"],
                                    "description": info["description"],
                                }.items()
                            }
                        )
                # resize the time dimension
                ibeg = dst.dimensions["time"].size
                if isrc == 0:
                    dst.resize_dimension("time", src.dimensions["time"].size)
                    isrc = 0
                else:
                    dst.resize_dimension(
                        "time",
                        dst.dimensions["time"].size + src.dimensions["time"].size - 1,
                    )
                    isrc = 1
                # copy the variables
                print(srcfile, src.dimensions["time"].size)
                dst.variables["time"][ibeg:] = src.variables["time"][isrc:]
                for var in vars_info.keys():
                    dst.variables[var][ibeg:, ...] = src.variables[var][isrc:, ...]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge surface variables from multiple HDF5 files into one."
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output NetCDF file.",
    )
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help="Input NetCDF files to merge.",
    )
    args = parser.parse_args()

    merge_surface_vars(args.inputs, args.output)
