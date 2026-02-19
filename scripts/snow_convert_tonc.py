#!/usr/bin/env python3
"""
Convert snow occupation GeoTIFF files to a NetCDF4 file with time dimension.

This script reads multiple GeoTIFF files containing snow occupation data,
extracts the coordinate transformation, builds latitude and longitude grids,
parses timestamps from filenames, and writes the data as time series to a NetCDF4 file.
"""

from datetime import UTC, datetime
from pathlib import Path

import h5netcdf
import numpy as np
import rasterio
from rasterio.transform import xy

DATETIME_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
DATETIME_UNITS = f"seconds since {DATETIME_EPOCH.isoformat(sep='T')}"


def parse_timestamp_from_filename(filename: Path) -> datetime:
    """
    Parse timestamp from filename like '2013_001.tif'.

    Args:
        filename: Filename to parse

    Returns:
        datetime object with UTC timezone
    """
    stem = filename.stem
    try:
        dt = datetime.strptime(stem, "%Y_%j")
        return dt.replace(tzinfo=UTC)
    except ValueError as e:
        raise ValueError(f"Cannot parse date from filename {filename}: {e}")


def extract_grid_from_geotiff(srcfile: Path):
    """Extract coordinate grid from GeoTIFF file."""
    from rasterio.warp import transform

    with rasterio.open(srcfile) as src:
        height, width = src.shape
        crs = src.crs
        ll_crs = rasterio.CRS.from_epsg(4326)  # WGS84

        # Create pixel coordinate arrays
        rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

        # Convert pixel coordinates to geographic coordinates
        xs, ys = xy(src.transform, rows.ravel() + 0.5, cols.ravel() + 0.5)
        lon, lat = transform(crs, ll_crs, xs, ys)  # type: ignore
        lon = np.array(lon).reshape(height, width)
        lat = np.array(lat).reshape(height, width)

        # Check if latitude decreases from north to south and flip if needed
        if lat[0, 0] > lat[-1, 0]:
            lat = np.flipud(lat)
            lon = np.flipud(lon)

        # Check if grid is regular
        lat_is_regular = np.allclose(lat, lat[:, 0:1])
        lon_is_regular = np.allclose(lon, lon[0:1, :])
        is_regular_grid = lat_is_regular and lon_is_regular

    if is_regular_grid:
        lat_1d = lat[:, 0]
        lon_1d = lon[0, :]
        return lat_1d, lon_1d, str(crs)
    else:
        return lat, lon, str(crs)


def read_snow_data(srcfile: Path) -> np.ndarray:
    """Read and validate snow data from GeoTIFF file."""
    scale = 0.1
    with rasterio.open(srcfile) as src:
        # Read first band
        data = src.read(1)

        # Check if latitude decreases from north to south and flip if needed
        transform = src.transform
        height, width = src.shape
        rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        _xs, ys = xy(transform, rows.ravel() + 0.5, cols.ravel() + 0.5)
        lat = np.array(ys).reshape(height, width)

        if lat[0, 0] > lat[-1, 0]:
            data = np.flipud(data)
    return scale * data


def main(output_file: Path, source_folder: Path):
    """
    Convert snow GeoTIFF files to NetCDF with time dimension.

    Args:
        output_file: Path to the output NetCDF4 file
        source_folder: Path to the folder containing source GeoTIFF files
    """
    # Find all TIF files matching the pattern
    pattern = "**/*.tiff"
    tif_files = sorted(source_folder.glob("**/*.tif"))

    if not tif_files:
        raise ValueError(
            f"No TIF files found in pattern: {source_folder.joinpath(pattern)}"
        )

    print(f"Found {len(tif_files)} TIF files")

    # Parse timestamps and sort files
    file_timestamps = []
    for tif_file in tif_files:
        try:
            timestamp = parse_timestamp_from_filename(tif_file)
            file_timestamps.append((timestamp, Path(tif_file)))
        except ValueError as e:
            print(f"Warning: Skipping file due to parsing error: {e}")
            continue

    if not file_timestamps:
        raise ValueError("No valid files found after timestamp parsing")

    # Sort by timestamp
    file_timestamps.sort(key=lambda x: x[0])
    timestamps = [x[0] for x in file_timestamps]
    sorted_files = [x[1] for x in file_timestamps]

    print(
        f"Processing {len(sorted_files)} files from {timestamps[0]} to {timestamps[-1]}"
    )

    # Extract grid from first file
    first_file = sorted_files[0]
    lat, lon, crs = extract_grid_from_geotiff(first_file)

    # Write to NetCDF4 file
    with h5netcdf.File(output_file, "w") as f:
        # Create time dimension (unlimited)
        f.dimensions["time"] = None

        if lat.ndim == 1 and lon.ndim == 1:
            # Regular grid - use 1D coordinate variables
            f.dimensions["latitude"] = lat.size
            f.dimensions["longitude"] = lon.size

            # Create coordinate variables
            lat_var = f.create_variable(
                "latitude",
                ("latitude",),
                data=lat,
                dtype=np.float64,
                compression="gzip",
            )
            lon_var = f.create_variable(
                "longitude",
                ("longitude",),
                data=lon,
                dtype=np.float64,
                compression="gzip",
            )

            # Create snow data variable (without data)
            snow_var = f.create_variable(
                "snow",
                ("time", "latitude", "longitude"),
                dtype=np.float64,
                compression="gzip",
            )
        else:
            # Irregular grid - use 2D coordinate variables
            f.dimensions["y"] = lat.shape[0]
            f.dimensions["x"] = lat.shape[1]

            lat_var = f.create_variable(
                "latitude",
                ("y", "x"),
                data=lat,
                dtype=np.float64,
                compression="gzip",
            )
            lon_var = f.create_variable(
                "longitude", ("y", "x"), data=lon, dtype=np.float64, compression="gzip"
            )

            # Create snow data variable (without data)
            snow_var = f.create_variable(
                "snow",
                ("time", "y", "x"),
                dtype=np.float64,
                fillvalue=np.nan,
                compression="gzip",
            )

            print("Irregular grid detected - using 2D coordinate variables")

        # Create time variable (without data)
        time_var = f.create_variable(
            "time", ("time",), dtype=np.int64, compression="gzip"
        )

        # Add coordinate attributes
        lat_var.attrs["units"] = np.bytes_("degrees_north", "ascii")
        lat_var.attrs["long_name"] = np.bytes_("latitude", "ascii")
        lat_var.attrs["standard_name"] = np.bytes_("latitude", "ascii")
        lat_var.attrs["axis"] = np.bytes_("Y", "ascii")

        lon_var.attrs["units"] = np.bytes_("degrees_east", "ascii")
        lon_var.attrs["long_name"] = np.bytes_("longitude", "ascii")
        lon_var.attrs["standard_name"] = np.bytes_("longitude", "ascii")
        lon_var.attrs["axis"] = np.bytes_("X", "ascii")

        # Add time attributes
        time_var.attrs["units"] = np.bytes_(DATETIME_UNITS, "ascii")
        time_var.attrs["long_name"] = np.bytes_("time", "ascii")
        time_var.attrs["standard_name"] = np.bytes_("time", "ascii")
        time_var.attrs["axis"] = np.bytes_("T", "ascii")
        time_var.attrs["calendar"] = np.bytes_("gregorian", "ascii")

        # Add snow data attributes
        snow_var.attrs["units"] = np.bytes_("kg m-2", "ascii")
        snow_var.attrs["long_name"] = np.bytes_("snow water equivalent", "ascii")
        snow_var.attrs["standard_name"] = np.bytes_("surface_snow_amount", "ascii")

        # Add global attributes
        f.attrs["title"] = np.bytes_(
            "Snow water equivalent data with coordinates", "ascii"
        )
        f.attrs["source"] = np.bytes_(
            "https://doi.org/10.11888/Cryos.tpdc.302476", "ascii"
        )
        f.attrs["history"] = np.bytes_(
            f"Created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}", "ascii"
        )
        f.attrs["Conventions"] = np.bytes_("CF-1.8", "ascii")

        if crs:
            f.attrs["crs"] = np.bytes_(str(crs), "ascii")

        # Process and write files one by one
        for i, (tif_file, timestamp) in enumerate(zip(sorted_files, timestamps)):
            print(f"Processing {i + 1}/{len(sorted_files)}: {tif_file.name}")

            # Read snow data for this file
            snow_data = read_snow_data(tif_file)

            # Resize dimensions to accommodate new time step
            f.resize_dimension("time", i + 1)

            # Write time value
            time_var[i] = (timestamp - DATETIME_EPOCH).total_seconds()

            # Write snow data
            snow_var[i] = snow_data

    print(f"Successfully processed {len(sorted_files)} files")
    if lat.ndim == 1 and lon.ndim == 1:
        print(f"Grid shape: {lat.size} x {lon.size}")
    else:
        print(f"Grid shape: {lat.shape[0]} x {lat.shape[1]}")
    print(f"Time range: {len(timestamps)} time steps")
    print(f"Latitude range: {lat.min():.6f} to {lat.max():.6f}")
    print(f"Longitude range: {lon.min():.6f} to {lon.max():.6f}")
    print(f"Snow data written to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert snow water equivalent GeoTIFF files to NetCDF4 with time dimension"
    )
    parser.add_argument(
        "source_folder",
        type=Path,
        help="Path to the source folder containing GeoTIFF files",
    )
    parser.add_argument(
        "output_file",
        type=Path,
        nargs="?",
        help="Path to the output NetCDF4 file (not required with --check-only)",
    )

    args = parser.parse_args()

    # Validate source folder exists
    if not args.source_folder.exists():
        print(f"Error: Source folder {args.source_folder} does not exist")
        exit(1)

    try:
        # Validate output directory exists
        if not args.output_file.parent.exists():
            print(f"Error: Output directory {args.output_file.parent} does not exist")
            exit(1)
        # Convert files
        main(args.output_file, args.source_folder)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
