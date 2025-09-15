import argparse
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import h5netcdf
import h5py
import numpy as np

DATETIME_EPOCH = datetime(2000, 1, 1, tzinfo=UTC)


def parse_datetime(dtstr: str) -> datetime:
    # Accepts ISO format, e.g., 2013-05-01T00:00:00 or 2013-05-01
    try:
        dt = datetime.fromisoformat(dtstr)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        else:
            dt = dt.astimezone(UTC)
        return dt
    except Exception:
        raise argparse.ArgumentTypeError(f"Invalid datetime: {dtstr}")


def read_gpm_hdf5(
    data_root: Path, dtbegin: datetime | None, dtend: datetime | None
) -> Iterator[
    tuple[
        datetime,
        datetime,
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ndarray[tuple[int, int], np.dtype[np.float32]],
    ]
]:
    """Read GPM HDF5 files from data_root between dtbegin and dtend (if provided)."""
    print(data_root)
    for filename in sorted(data_root.glob("**/*.HDF5")):
        dtstrday, dtstrs, dtstre = filename.stem.split(".")[4].split("-")
        dtb = datetime.fromisoformat(f"{dtstrday}T{dtstrs.removeprefix('S')}").replace(
            tzinfo=UTC
        )
        dte = datetime.fromisoformat(f"{dtstrday}T{dtstre.removeprefix('E')}").replace(
            tzinfo=UTC
        ) + timedelta(seconds=1)
        print(dtb, dte)
        # Handle None for dtbegin/dtend
        if dtbegin is not None and dte <= dtbegin:
            continue
        if dtend is not None and dtb >= dtend:
            continue
        with h5py.File(filename, "r") as f:
            lon = np.array(f["Grid/lon"][:], np.float64)  # type: ignore
            lat = np.array(f["Grid/lat"][:], np.float64)  # type: ignore
            precip = np.array(f["Grid/precipitation"][0, :, :], np.float32).T  # type: ignore
            yield (dtb, dte, lon, lat, precip)


def main():
    parser = argparse.ArgumentParser(description="Convert GPM HDF5 files to NetCDF.")
    parser.add_argument(
        "srcroot", type=Path, help="Root directory containing GPM HDF5 files."
    )
    parser.add_argument("desfile", type=Path, help="Output NetCDF file.")
    parser.add_argument(
        "--start",
        type=parse_datetime,
        default=None,
        help="Start datetime (inclusive), e.g., 2013-05-01T00:00:00",
    )
    parser.add_argument(
        "--stop",
        type=parse_datetime,
        default=None,
        help="Stop datetime (exclusive), e.g., 2013-05-02T00:00:00",
    )
    args = parser.parse_args()

    with h5netcdf.File(args.desfile, "w") as f:
        f.attrs.update(
            {
                "title": "GPM IMERG Precipitation Data",
                "institution": "IAP, CAS",
                "source": "GPM IMERG v7 3B-HHR data",
                "history": f"Created {datetime.now(UTC).isoformat(sep='T')}",
                "references": "https://gpm.nasa.gov/",
            }
        )
        for ii, (dtb, dte, lon, lat, precip) in enumerate(
            read_gpm_hdf5(args.srcroot, args.start, args.stop)
        ):
            if ii == 0:
                f.dimensions["time"] = None
                f.dimensions["bnd"] = 2
                f.dimensions["lat"] = len(lat)
                f.dimensions["lon"] = len(lon)

                f.create_variable(
                    "time", ("time",), dtype="f8", compression="gzip"
                ).attrs.update(
                    {
                        "units": np.bytes_(
                            f"seconds since {DATETIME_EPOCH.isoformat(sep='T')}",
                            "ascii",
                        ),
                        "standard_name": np.bytes_("time", "ascii"),
                        "bounds": np.bytes_("time_bnds", "ascii"),
                    }
                )
                f.create_variable(
                    "time_bnds", ("time", "bnd"), dtype="f8", compression="gzip"
                ).attrs.update(
                    {
                        "units": np.bytes_(
                            f"seconds since {DATETIME_EPOCH.isoformat(sep='T')}",
                            "ascii",
                        ),
                        "standard_name": np.bytes_("time", "ascii"),
                    }
                )

                f.create_variable(
                    "lat", ("lat",), dtype="f8", data=lat, compression="gzip"
                ).attrs.update(
                    {
                        "units": np.bytes_("degrees_north", "ascii"),
                        "standard_name": np.bytes_("latitude", "ascii"),
                    }
                )

                f.create_variable(
                    "lon", ("lon",), dtype="f8", data=lon, compression="gzip"
                ).attrs.update(
                    {
                        "units": np.bytes_("degrees_east", "ascii"),
                        "standard_name": np.bytes_("longitude", "ascii"),
                    }
                )

                f.create_variable(
                    "pr",
                    ("time", "lat", "lon"),
                    dtype="f4",
                    fillvalue=np.nan,
                    compression="gzip",
                ).attrs.update(
                    {
                        "units": np.bytes_("kg m-2 s-1", "ascii"),
                        "standard_name": np.bytes_("precipitation_flux", "ascii"),
                    }
                )

                pr_tmp = np.zeros((len(lat), len(lon)), np.float32)
                count = 0

            pr_tmp[:, :] += np.where(precip == -9999.9, np.nan, precip)  # type: ignore
            count += 1  # type: ignore
            if dte.minute == 0:
                print(f"Writing data for {dte}, count={count}")

                f.resize_dimension("time", f.dimensions["time"].size + 1)
                f.variables["time"][-1] = (dte - DATETIME_EPOCH).total_seconds()
                f.variables["time_bnds"][-1, :] = (
                    (dte - timedelta(hours=1) - DATETIME_EPOCH).total_seconds(),
                    (dte - DATETIME_EPOCH).total_seconds(),
                )
                f.variables["pr"][-1, :, :] = (
                    pr_tmp / count / 3600.0  # type: ignore
                )  # convert mm/hr to kg/m2/s

                pr_tmp[:, :] = 0.0  # type: ignore
                count = 0


if __name__ == "__main__":
    main()
