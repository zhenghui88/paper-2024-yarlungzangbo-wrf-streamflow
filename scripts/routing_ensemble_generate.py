#!/usr/bin/env python3

"""
Generate an ensemble of river routing parameter files by:
  1) Copying gauge network JSON and gauge measurement CSV once to an output directory.
  2) Copying a base river parameter Parquet file N times, naming each copy with a
     zero-padded index appended to its stem.
  3) For each copy, perturbing the 'celerity' values based on gauge-specific
     log-normal distributions defined by mean_lncelerity and std_lncelerity.
"""

import argparse
import json
import logging
import math
import random
import shutil
from pathlib import Path
from typing import cast
from uuid import UUID

import pyarrow
import pyarrow.feather


def load_gauge_stats(csv_path: Path) -> dict[str, tuple[float, float]]:
    """
    Read the gauge measurement CSV and return a dict mapping
    gauge ID string to (mean_lncelerity, std_lncelerity).
    """
    gauge_index = 0
    mean_index = 1
    std_index = 2
    stats: dict[str, tuple[float, float]] = {}
    with open(csv_path, "rt") as f:
        header = [x.strip().lower() for x in f.readline().split(",")]
        gauge_index = header.index("gauge")
        mean_index = header.index("mean_lncelerity")
        std_index = header.index("std_lncelerity")
        for line in f:
            parts = line.split(",")
            if len(parts) <= max(gauge_index, mean_index, std_index):
                raise ValueError(f"Invalid line in CSV: {line}")
            gauge = parts[gauge_index].strip()
            mean_ln = float(parts[mean_index].strip())
            std_ln = float(parts[std_index].strip())
            stats[gauge] = (mean_ln, std_ln)
    return stats


def load_gauge_network(json_path: Path) -> dict[str, set[UUID]]:
    """
    Read the gauge network JSON and return its dict. Each key is a gauge ID string,
    and its value is expected to contain a "river" list of UUID strings.
    """
    network: dict[str, set[UUID]] = {}

    with json_path.open("r") as f:
        data = json.load(f)
        for gauge, info in data.items():
            rivers = info.get("river", [])
            river_uuids = {UUID(r) for r in rivers}
            network[str(gauge)] = river_uuids

    return network


def perturb_and_write(
    parameter_path: Path,
    gauge_stats: dict[str, tuple[float, float]],
    gauge_network: dict[str, set[UUID]],
) -> None:
    """
    Load a river parameter Parquet file into a DataFrame, perturb its 'celerity'
    column for each gauge by drawing one log-normal random sample per gauge, then
    overwrite the Parquet file with updated values.
    """
    table = pyarrow.feather.read_table(parameter_path)
    colnames = list(table.column_names)
    ids = table["id"].to_pylist()
    celerities = table["celerity"].to_pylist()

    sample = random.normalvariate()
    logging.info(f"Random sample for perturbation: {sample}")
    for gauge, (mean_ln, std_ln) in gauge_stats.items():
        ln_sample = sample * std_ln + mean_ln
        celerity_val = math.exp(ln_sample)
        rivers = gauge_network.get(gauge, cast(set[UUID], {}))
        if not rivers:
            logging.debug(f"No rivers linked to gauge {gauge}")
            continue

        for i, r in enumerate(ids):
            if r in rivers:
                celerities[i] = celerity_val

    new_table = pyarrow.table(
        [
            table[col]
            if col != "celerity"
            else pyarrow.array(celerities, type=pyarrow.float32())
            for col in colnames
        ],
        names=colnames,
    )
    pyarrow.feather.write_feather(new_table, parameter_path)
    logging.info(f"Written perturbed Parquet: {parameter_path}")


def generate_ensemble(
    base_parquet: Path,
    gauge_json: Path,
    gauge_csv: Path,
    members: int,
    output_dir: Path,
) -> None:
    """
    Generate an ensemble of 'members' river parameter files by copying and
    perturbing the base_parquet. Copies gauge files once and writes N
    perturbed Parquet files into output_dir.
    """
    # Validate inputs
    for p in (base_parquet, gauge_json, gauge_csv):
        if not p.is_file():
            logging.error(f"Required file not found: {p}")
            raise FileNotFoundError(p)
    if members < 1:
        logging.error("Number of ensemble members must be >= 1")
        raise ValueError("Invalid ensemble size")

    # Prepare output directory
    if not output_dir.is_dir():
        logging.error(f"Output directory does not exist: {output_dir}")
        raise NotADirectoryError(output_dir)

    # Load gauge stats and network
    gauge_stats = load_gauge_stats(gauge_csv)
    gauge_network_map = load_gauge_network(gauge_json)

    # Determine naming
    width = len(str(members))
    stem = base_parquet.stem
    suffix = base_parquet.suffix

    # Generate ensemble members
    for i in range(1, members + 1):
        idx = f"{i:0{width}d}"
        out_name = f"{stem}_{idx}{suffix}"
        out_path = output_dir / out_name
        shutil.copyfile(base_parquet, out_path)
        logging.info(f"[{idx}] Created copy {out_name}")

        perturb_and_write(out_path, gauge_stats, gauge_network_map)


def main():
    parser = argparse.ArgumentParser(
        description="Generate an ensemble of routing parameter Parquet files"
    )
    parser.add_argument(
        "river_parameter",
        type=Path,
        help="Base river parameter Parquet file (with 'id' and 'celerity' columns)",
    )
    parser.add_argument("gauge_network", type=Path, help="Gauge network JSON file")
    parser.add_argument(
        "gauge_measurement",
        type=Path,
        help="Gauge measurement CSV file (gauge,mean_lncelerity,std_lncelerity)",
    )
    parser.add_argument(
        "number", type=int, help="Number of ensemble members to generate"
    )
    parser.add_argument(
        "output", type=Path, help="Output directory to write ensemble files"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    try:
        generate_ensemble(
            args.river_parameter,
            args.gauge_network,
            args.gauge_measurement,
            args.number,
            args.output,
        )
    except Exception as exc:
        logging.error(f"Ensemble generation failed: {exc}")
        exit(1)


if __name__ == "__main__":
    main()
