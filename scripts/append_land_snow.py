from pathlib import Path

import h5netcdf as nc
import numpy as np

LATSLICE = slice(162, 287)
LONSLICE = slice(92, 487)


def append_land_snow(
    src_file: Path, land_file: Path, snow_file: Path, dst_file: Path
) -> None:
    """
    Append land and snow data to the source file.

    Parameters:
        src_file (Path): Path to the source file.
        land_file (Path): Path to the land data file.
        snow_file (Path): Path to the snow data file.
        dest_file (Path): Path to the destination file.
    """
    with (
        nc.File(str(dst_file), "w") as dst,
        nc.File(str(src_file), "r") as src,
    ):
        # Copy global attributes from source file to destination file
        dst.attrs.update(
            {
                np.bytes_(k, "ascii"): np.bytes_(v, "ascii")
                if isinstance(v, str)
                else v
                for k, v in src.attrs.items()
                if k != "comment"
            }
        )
        dst.attrs["comment"] = np.bytes_(
            "range of latitude index in the original domain: [162-286]\nrange of longitude index in the original domain: [92-486]",
            "ascii",
        )

        # Copy dimensions from source file to destination file
        for dim_name, dim in src.dimensions.items():
            if dim.isunlimited():
                dst.dimensions[dim_name] = None
            elif dim_name == "ensemble":
                dst.dimensions[dim_name] = dim.size + 2
            else:
                dst.dimensions[dim_name] = dim.size

        # Copy variables (time, lat, lon, mask, pr, and gpm) definitions from source to destination
        for var_name in src.variables.keys():
            src_var = src.variables[var_name]
            fillvalue = getattr(src_var, "_FillValue", None)
            # Prepare attributes for the variable
            var_attrs = {
                k: v
                for k, v in {
                    np.bytes_(a, "ascii"): np.bytes_(v, "ascii")
                    if isinstance(v, str)
                    else v
                    for a, v in src_var.attrs.items()
                    if a != "_FillValue"
                }.items()
            }
            v = dst.create_variable(
                var_name,
                src_var.dimensions,
                src_var.dtype,
                fillvalue=fillvalue,
                compression="gzip",
                compression_opts=9,
            )
            for attr_key, attr_val in var_attrs.items():
                v.attrs[attr_key] = attr_val

        dst.resize_dimension("time", src.dimensions["time"].size)

        # Copy data for time, lat, lon, mask, and gpm
        for var_name in ["time", "lat", "lon", "mask", "gpm"]:
            dst.variables[var_name][:] = src.variables[var_name][:]

        mask = dst.variables["mask"][:] > 0

        # Append pr variable data
        with (
            nc.File(str(land_file), "r") as land,
            nc.File(str(snow_file), "r") as snow,
        ):
            land_data = land.variables["RAINNC"][:][:, LATSLICE, LONSLICE][
                np.newaxis, :, :, :
            ]
            land_data[
                ~np.repeat(
                    mask[np.newaxis, np.newaxis, :, :], land_data.shape[1], axis=1
                )
            ] = np.nan
            snow_data = snow.variables["RAINNC"][:][:, LATSLICE, LONSLICE][
                np.newaxis, :, :, :
            ]
            snow_data[
                ~np.repeat(
                    mask[np.newaxis, np.newaxis, :, :], snow_data.shape[1], axis=1
                )
            ] = np.nan
            data = np.concatenate(
                (src.variables["pr"][:], land_data, snow_data), axis=0
            )
            dst.variables["pr"][:] = data

        # Append mrro variable data
        with (
            nc.File(str(land_file), "r") as land,
            nc.File(str(snow_file), "r") as snow,
        ):
            land_data = (
                land.variables["SFROFF"][:][:, LATSLICE, LONSLICE]
                + land.variables["UDROFF"][:][:, LATSLICE, LONSLICE]
            )[np.newaxis, :, :, :]
            land_data[
                ~np.repeat(
                    mask[np.newaxis, np.newaxis, :, :], land_data.shape[1], axis=1
                )
            ] = np.nan
            snow_data = (
                snow.variables["SFROFF"][:][:, LATSLICE, LONSLICE]
                + snow.variables["UDROFF"][:][:, LATSLICE, LONSLICE]
            )[np.newaxis, :, :, :]
            snow_data[
                ~np.repeat(
                    mask[np.newaxis, np.newaxis, :, :], snow_data.shape[1], axis=1
                )
            ] = np.nan
            data = np.concatenate(
                (src.variables["mrro"][:], land_data, snow_data), axis=0
            )
            dst.variables["mrro"][:] = data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Append land and snow data to the source file."
    )
    parser.add_argument("src_file", type=Path, help="Path to the source file.")
    parser.add_argument("land_file", type=Path, help="Path to the land data file.")
    parser.add_argument("snow_file", type=Path, help="Path to the snow data file.")
    parser.add_argument("dst_file", type=Path, help="Path to the destination file.")

    args = parser.parse_args()

    append_land_snow(args.src_file, args.land_file, args.snow_file, args.dst_file)
