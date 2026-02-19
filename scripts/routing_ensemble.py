import concurrent.futures
import itertools
import json
import logging
from datetime import timedelta
from pathlib import Path
from uuid import UUID

from routing import MuskingumData, collect_output, write_output
from routing_common import Simulation


def read_riverids(filepath: Path):
    riverids: set[UUID] = set()
    with open(filepath, "rt") as f:
        data = json.load(f)
        for info in data.values():
            riverids.add(UUID(info["at"]))  # type:ignore
    return riverids


def routing_single(
    river_setup: Path,
    river_parameter: Path,
    init_file: Path,
    timestep: timedelta,
    qlat_file: Path,
    riverids: set[UUID],
    qout_file: Path,
    model: str = "muskingum",
):
    """
    Run a single river routing simulation.

    Returns a generator of (datetime, {river_id: flow}) tuples.
    """
    logging.info(
        f"Running routing for {river_parameter} and {qlat_file}, output to {qout_file}"
    )
    muskingum_data = MuskingumData(
        river_setup, river_parameter, init_file, qlat_file, timestep, model
    )
    simulation = Simulation(
        muskingum_data.topo,
        muskingum_data.init,
        muskingum_data.length,
        muskingum_data.weight,
        muskingum_data.celerity,
        muskingum_data.timestep,
        iter(muskingum_data.qlat),
        None,
        muskingum_data.model,
    )
    qout = collect_output(riverids, simulation.run())
    write_output(qout_file, qout)


def routing_ensemble(
    river_setup: Path,
    river_parameter_root: Path,
    init_file: Path,
    timestep: timedelta,
    qlat_root: Path,
    gauge_network: Path,
    qout_root: Path,
    max_proc: int,
    model: str = "muskingum",
):
    """
    Run river routing simulations with perturbed parameters.

    The parameter files are assumed to be named with a <river_parameter_root>/river_parameter_xxx.<suffix> where xxx is a number.
    The lateral flow input files are assumed to be named with a <qlat_root>/qlat_yyy.parquet where yyy is a number.
    The output files are named with a <qout_root>/qout_yyy_xxx.parquet where yyy and xxx are the same as in the input files.
    """
    riverids = read_riverids(gauge_network)
    qlat_files = sorted(qlat_root.glob("qlat_*.parquet"))
    parameter_files = sorted(river_parameter_root.glob("river_parameter_*.arrow"))
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_proc) as executor:
        futures = []
        for qlat_file, parameter_file in itertools.product(qlat_files, parameter_files):
            qout_file = (
                qout_root
                / f"qout_{qlat_file.stem.split('_')[-1]}_{parameter_file.stem.split('_')[-1]}.parquet"
            )

            futures.append(
                executor.submit(
                    routing_single,
                    river_setup,
                    parameter_file,
                    init_file,
                    timestep,
                    qlat_file,
                    riverids,
                    qout_file,
                    model,
                )
            )
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Simulation generated an exception: {e}")


if __name__ == "__main__":
    import argparse
    import logging

    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s %(levelname)s: %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Muskingum river routing with segmental optimization"
    )
    parser.add_argument(
        "river_setup", type=Path, help="Path to river setup file (Arrow)"
    )
    parser.add_argument(
        "river_parameter_root",
        type=Path,
        help="Root path to river network parameter file (Arrow)",
    )
    parser.add_argument(
        "init", type=Path, help="Path to river initial condition file (Arrow)"
    )
    parser.add_argument(
        "step",
        type=lambda x: timedelta(seconds=float(x)),
        help="Maximum time step in seconds",
    )
    parser.add_argument(
        "qlat_root",
        type=Path,
        help="Root path to lateral flow input file (Parquet)",
    )
    parser.add_argument("qout_root", type=Path, help="Case root path")
    parser.add_argument("gauge_network", type=Path, help="Gauge topology file (Json)")
    parser.add_argument(
        "--max_proc", type=int, default=4, help="Maximum number of processes"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["muskingum", "pass_through"],
        default="muskingum",
        help="River routing model to use",
    )
    args = parser.parse_args()

    routing_ensemble(
        args.river_setup,
        args.river_parameter_root,
        args.init,
        args.step,
        args.qlat_root,
        args.gauge_network,
        args.qout_root,
        args.max_proc,
        args.model,
    )
