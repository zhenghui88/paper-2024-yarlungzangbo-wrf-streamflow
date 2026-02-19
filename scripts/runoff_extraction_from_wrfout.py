import argparse
from datetime import datetime, timedelta
from pathlib import Path

import h5netcdf
import numpy as np

DATETIME_EPOCH = datetime(2000, 1, 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract total runoff (mrro) from wrfout files."
    )
    parser.add_argument(
        "srcfile", type=Path, help="Source file containing the runoff data"
    )
    parser.add_argument("outfile", type=Path, help="Output NetCDF file")
    parser.add_argument(
        "-b",
        "--start",
        type=str,
        help="Start datetime (YYYY-mm-ddTHH:MM:SS)",
    )
    parser.add_argument(
        "-e",
        "--stop",
        type=str,
        help="Stop datetime (YYYY-mm-ddTHH:MM:SS)",
    )
    return parser.parse_args()


def get_lat_lon(ncfile):
    with h5netcdf.File(ncfile, "r") as f:
        nlon = int(f.dimensions["west_east"].size)
        nlat = int(f.dimensions["south_north"].size)
        lon = f.variables["XLONG"][nlat // 2, :]
        lat = f.variables["XLAT"][:, nlon // 2]
    return np.array(lat), np.array(lon)


def create_output(outfile, lat, lon):
    with h5netcdf.File(outfile, "w", decode_vlen_strings=False) as f:
        f.dimensions = {"time": None, "lat": len(lat), "lon": len(lon)}
        f.create_variable(
            "time", ("time",), np.int64, compression="gzip", compression_opts=6
        )
        f.create_variable(
            "lat",
            ("lat",),
            np.float64,
            data=lat,
            compression="gzip",
            compression_opts=6,
        )
        f.create_variable(
            "lon",
            ("lon",),
            np.float64,
            data=lon,
            compression="gzip",
            compression_opts=6,
        )
        f.create_variable(
            "mrro",
            ("time", "lat", "lon"),
            np.float32,
            fillvalue=np.nan,
            compression="gzip",
            compression_opts=6,
        )
        f["time"].attrs.update(
            {"units": np.bytes_(f"seconds since {DATETIME_EPOCH.isoformat()}", "ascii")}
        )
        f["lat"].attrs.update({"units": np.bytes_("degrees_north", "ascii")})
        f["lon"].attrs.update({"units": np.bytes_("degrees_east", "ascii")})
        f["mrro"].attrs.update(
            {
                "long_name": np.bytes_("total_runoff", "ascii"),
                "units": np.bytes_("kg m-2 s-1", "ascii"),
            }
        )


def parse_time_ref(time_units: str):
    # Parse the time reference from the units string
    if "since" in time_units:
        time_ref_str = time_units.split("since")[1].strip()
        time_ref = datetime.fromisoformat(time_ref_str.replace("Z", "+00:00"))
    else:
        raise ValueError(f"Invalid time units format: {time_units}")
    return time_ref


def main():
    args = parse_args()
    dt_start = datetime.fromisoformat(args.start) if args.start else None
    dt_stop = datetime.fromisoformat(args.stop) if args.stop else None

    # Find lat/lon from the first available file
    lat, lon = get_lat_lon(args.srcfile)
    create_output(args.outfile, lat, lon)

    # Set the "time" dimension size in advance
    with (
        h5netcdf.File(args.outfile, "a") as fout,
        h5netcdf.File(args.srcfile, "r") as fsrc,
    ):
        time_ref = parse_time_ref(fsrc.variables["time"].attrs["units"])
        times = [time_ref + timedelta(seconds=x) for x in fsrc.variables["time"][:]]

        fout.resize_dimension("time", len(times))

        for idx, t in enumerate(times):
            if dt_start and t < dt_start:
                continue
            if dt_stop and t > dt_stop:
                continue
            fout.variables["time"][idx] = int((t - DATETIME_EPOCH).total_seconds())
            qsrf = np.maximum(fsrc.variables["SFROFF"][idx, :, :], 0.0)
            qsub = np.maximum(fsrc.variables["UDROFF"][idx, :, :], 0.0)
            mrro = qsrf + qsub
            fout.variables["mrro"][idx, :, :] = mrro


if __name__ == "__main__":
    main()
