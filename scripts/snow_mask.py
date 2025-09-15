#!/usr/bin/env python3
"""
Apply mask to snow water equivalent data.

This script reads a NetCDF4 file containing snow water equivalent data
(from snow_interpolation.py) and applies a mask from another NetCDF4 file.
Where mask == 0, the snow data is set to NaN.

The mask file must have the same spatial dimensions as the data file.
"""

import argparse
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import h5netcdf
import numpy as np


def read_data_file(data_file: Path):
    """
    Read snow water equivalent data from the NetCDF file.

    Args:
        data_file: Path to the data NetCDF file

    Returns:
        tuple: (snow_data, mask_shape)
    """
    with h5netcdf.File(data_file, "r") as f:
        # Read snow data
        if "snow" in f.variables:
            snow_data = f.variables["snow"][:]
        else:
            raise ValueError("Could not find snow variable in data file")
        snow_data = cast(np.ndarray, snow_data)

    # Return data and expected mask shape (spatial dimensions only)
    mask_shape = (int(snow_data.shape[1]), int(snow_data.shape[2]))

    return snow_data, mask_shape


def read_mask_file(mask_file: Path):
    """
    Read mask data from the NetCDF file.

    Args:
        mask_file: Path to the mask NetCDF file

    Returns:
        np.ndarray: Mask data array
    """
    with h5netcdf.File(mask_file, "r") as f:
        # Read mask variable
        if "mask" in f.variables:
            mask = f.variables["mask"][:]
        else:
            raise ValueError("Could not find mask variable in mask file")
        mask = cast(np.ndarray, mask)

    return mask > 0


def apply_mask_to_file(output_file: Path, mask: np.ndarray, expected_shape: tuple):
    """
    Apply mask to snow data in the NetCDF file. Where mask == False, set snow data to NaN.

    Args:
        output_file: Path to the NetCDF file to modify
        mask: Mask array (lat, lon)
        expected_shape: Expected spatial shape (lat, lon) for validation
    """
    # Check mask dimensions
    if mask.shape != expected_shape:
        raise ValueError(
            f"Mask shape {mask.shape} does not match expected spatial shape {expected_shape}"
        )

    # Modify the file in place
    with h5netcdf.File(output_file, "a") as f:
        snow_var = f.variables["snow"]

        # Apply 2D mask to all time steps
        for t in range(snow_var.shape[0]):
            data_slice = snow_var[t, :, :]
            data_slice[~mask] = np.nan
            snow_var[t, :, :] = data_slice

        # Update history
        now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        history_str = f"Mask applied on {now_str}"
        if "history" in f.attrs:
            existing_history = str(f.attrs["history"])
            f.attrs["history"] = np.bytes_(
                f"{existing_history}; {history_str}", "ascii"
            )
        else:
            f.attrs["history"] = np.bytes_(history_str, "ascii")


def count_masked_points(output_file: Path):
    """
    Count the number of valid data points after masking.

    Args:
        output_file: Path to the output NetCDF file

    Returns:
        int: Number of valid (non-NaN) data points
    """
    with h5netcdf.File(output_file, "r") as f:
        snow_data = f.variables["snow"][:]
        return int(np.sum(~np.isnan(snow_data)))


def main():
    """Main function to handle command line arguments and orchestrate the masking."""
    parser = argparse.ArgumentParser(
        description="Apply mask to snow water equivalent data"
    )
    parser.add_argument(
        "data_file",
        type=Path,
        help="Path to the data NetCDF file (from snow_interpolation.py)",
    )
    parser.add_argument(
        "mask_file",
        type=Path,
        help="Path to the mask NetCDF file with 'mask' variable",
    )
    parser.add_argument(
        "output_file",
        type=Path,
        help="Path to the output NetCDF file",
    )

    args = parser.parse_args()

    # Validate input files exist
    if not args.data_file.exists():
        print(f"Error: Data file {args.data_file} does not exist")
        exit(1)

    if not args.mask_file.exists():
        print(f"Error: Mask file {args.mask_file} does not exist")
        exit(1)

    # Validate output directory exists
    if not args.output_file.parent.exists():
        print(f"Error: Output directory {args.output_file.parent} does not exist")
        exit(1)

    try:
        print("Reading data file...")
        snow_data, mask_shape = read_data_file(args.data_file)
        original_valid = int(np.sum(~np.isnan(snow_data)))

        print("Reading mask file...")
        mask = read_mask_file(args.mask_file)

        print("Copying data file to output...")
        shutil.copyfile(args.data_file, args.output_file)

        print("Applying mask...")
        apply_mask_to_file(args.output_file, mask, mask_shape)

        # Count remaining valid points
        masked_valid = count_masked_points(args.output_file)
        masked_points = original_valid - masked_valid

        print("Successfully applied mask to snow data")
        print(f"Output file: {args.output_file}")
        print(f"Data shape: {snow_data.shape}")
        print(f"Masked {masked_points} data points")
        print(f"Remaining valid points: {masked_valid}")

    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
