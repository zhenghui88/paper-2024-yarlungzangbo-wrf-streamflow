#!/usr/bin/env python3
"""
Parse streamflow observations for multiple stations and consolidate into a single file.

Reads station codes from a CSV file, loads corresponding JSON.zip files,
and creates a time-aligned DataFrame with one column per station.
"""

import argparse
import json
import logging
import sys
import zipfile
from datetime import timedelta
from pathlib import Path
from typing import List

import polars as pl

# Global constant for the data column to extract (makes future edits easier)
DATA_COLUMN = "q"


def read_station_list(csv_path: Path) -> List[str]:
    """Read station codes from CSV file."""
    try:
        df = pl.read_csv(
            csv_path,
            columns=[
                "stcd",
            ],
            infer_schema=False,
        )
    except Exception as e:
        raise ValueError(f"Failed to read CSV file {csv_path}: {e}")
    return df["stcd"].cast(pl.Utf8).to_list()


def read_station_data(station_code: str, source_root: Path) -> pl.DataFrame:
    """Read JSON data for a single station using the pattern from streamflow_json2parquet.py."""
    json_zip_path = source_root / f"{station_code}.json.zip"

    if not json_zip_path.exists():
        logging.warning("Data file not found: %s", json_zip_path)
        return pl.DataFrame()

    try:
        with zipfile.ZipFile(json_zip_path, "r") as zip_ref:
            json_files = [
                name for name in zip_ref.namelist() if name.lower().endswith(".json")
            ]
            if not json_files:
                raise ValueError(f"No JSON files found in ZIP archive: {json_zip_path}")
            first_json = json_files[0]
            logging.debug(
                "Reading JSON file from ZIP: %s -> %s", json_zip_path, first_json
            )
            with zip_ref.open(first_json) as json_file:
                json_data = json.load(json_file)

        # Use the pattern from streamflow_json2parquet.py
        first_entry = next(iter(json_data["data"].values()))
        df = pl.DataFrame(first_entry)

        # Convert columns as in the original script
        df = df.select(
            (
                pl.col("tm").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")
                - timedelta(hours=8)
            ).cast(pl.Datetime(time_unit="ms", time_zone="UTC")),
            pl.col(DATA_COLUMN).cast(pl.Float32),
            pl.col("z").cast(pl.Float32),
        )

        # Rename the data column to station code
        df = df.rename({DATA_COLUMN: station_code})

        logging.info("Loaded %d records for station %s", len(df), station_code)
        return df

    except Exception as e:
        logging.error("Failed to read data for station %s: %s", station_code, e)
        return pl.DataFrame()


def consolidate_station_data(
    station_codes: List[str], source_root: Path
) -> pl.DataFrame:
    """Read and consolidate data from multiple stations."""
    dataframes = []

    for station_code in station_codes:
        df = read_station_data(station_code, source_root)
        if not df.is_empty():
            # Keep only tm and the station column (drop z for consolidation)
            df = df.select(["tm", station_code])
            dataframes.append(df)

    if not dataframes:
        logging.warning("No data loaded from any station")
        return pl.DataFrame()

    # Join all DataFrames on the 'tm' column
    consolidated = dataframes[0]
    for df in dataframes[1:]:
        consolidated = consolidated.join(df, on="tm", how="full", coalesce=True)

    # Sort by time
    consolidated = consolidated.sort("tm")

    logging.info(
        "Consolidated data: %d rows, %d columns",
        len(consolidated),
        len(consolidated.columns),
    )
    return consolidated


def write_output(df: pl.DataFrame, output_path: Path, force: bool = False):
    """Write DataFrame to output file (format determined by extension)."""
    if output_path.exists() and not force:
        raise FileExistsError(
            f"Output file {output_path} exists. Use --force to overwrite."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".parquet":
        df.write_parquet(output_path)
    elif output_path.suffix.lower() == ".csv":
        df.write_csv(output_path)
    else:
        # Default to parquet
        df.write_parquet(output_path)

    logging.info("Wrote %d rows to %s", len(df), output_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse streamflow observations for multiple stations"
    )
    parser.add_argument(
        "station_list",
        type=Path,
        help="CSV file containing station codes (must have 'stcd' column)",
    )
    parser.add_argument(
        "source_root",
        type=Path,
        help="Root directory containing station data files (<station_code>.json.zip)",
    )
    parser.add_argument(
        "destination",
        type=Path,
        help="Output file path (format determined by extension)",
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="Overwrite output if exists"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Validate input paths
    if not args.station_list.exists():
        logging.error("Station list file does not exist: %s", args.station_list)
        sys.exit(1)

    if not args.source_root.exists():
        logging.error("Source root directory does not exist: %s", args.source_root)
        sys.exit(1)

    try:
        # Read station codes
        logging.info("Reading station list from %s", args.station_list)
        station_codes = read_station_list(args.station_list)
        logging.info("Found %d stations to process", len(station_codes))

        # Consolidate data
        logging.info("Reading and consolidating station data from %s", args.source_root)
        consolidated_df = consolidate_station_data(station_codes, args.source_root)

        if consolidated_df.is_empty():
            logging.warning("No data to write")
            sys.exit(1)

        # Write output
        logging.info("Writing consolidated data to %s", args.destination)
        write_output(consolidated_df, args.destination, force=args.force)

        logging.info("Done.")

    except Exception as e:
        logging.exception("Script failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
