import functools
import logging
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pyarrow
import pyarrow.feather
import pyarrow.parquet
from routing_common import Simulation, collect_output


class MuskingumData:
    def __init__(
        self,
        river_setup: Path,
        river_parameter: Path,
        init_file: Path,
        qlat_file: Path,
        timestep: timedelta,
        model: str = "muskingum",
    ):
        self.topo = river_setup_read(river_setup)
        self.length, self.weight, self.celerity = muskingum_parameter_read(
            river_parameter
        )
        self.init = initial_condition_read(init_file)
        self.qlat = read_lateral_flow(qlat_file)
        self.timestep = timestep
        self.model = model


def river_setup_read(filepath: Path) -> dict[UUID, UUID | None]:
    table = pyarrow.feather.read_table(filepath)
    ids: list[UUID] = table["id"].to_pylist()
    tos: list[UUID | None] = table["to"].to_pylist()
    topo = {rid: rto for rid, rto in zip(ids, tos)}
    return topo


def muskingum_parameter_read(
    filepath: Path,
) -> tuple[dict[UUID, float], dict[UUID, float], dict[UUID, float]]:
    table = pyarrow.feather.read_table(filepath)
    ids: list[UUID] = table["id"].to_pylist()
    lengths: list[float] = table["length"].to_pylist()
    weights: list[float] = table["x"].to_pylist()
    celerities: list[float] = table["celerity"].to_pylist()
    length = {k: v for k, v in zip(ids, lengths)}
    weight = {k: v for k, v in zip(ids, weights)}
    celerity = {k: v for k, v in zip(ids, celerities)}
    return length, weight, celerity


def initial_condition_read(
    filepath: Path,
) -> tuple[datetime, dict[UUID, float]]:
    init_dt = datetime.fromisoformat(filepath.stem)
    if init_dt.tzinfo is None:
        init_dt = init_dt.replace(tzinfo=UTC)
    else:
        init_dt = init_dt.astimezone(tz=UTC)
    try:
        table = pyarrow.feather.read_table(filepath)
        ids: list[UUID] = table["id"].to_pylist()
        qinit: list[float] = table["q"].to_pylist()
    except FileNotFoundError:
        # if initial condition file not found, assume zero initial condition
        ids = []
        qinit = []
    return init_dt, {k: v for k, v in zip(ids, qinit)}


@functools.lru_cache(maxsize=1)
def read_lateral_flow(qlat_file) -> list[tuple[datetime, dict[UUID, float]]]:
    ret: list[tuple[datetime, dict[UUID, float]]] = []
    table = pyarrow.parquet.read_table(qlat_file)
    ids: list[UUID] = table["id"].to_pylist()
    for colname in table.column_names[1:]:
        dt = datetime.fromisoformat(colname)
        qlat_values: list[float | None] = table[colname].to_pylist()
        data = {k: v for k, v in zip(ids, qlat_values) if v is not None}
        ret.append((dt, data))
    return ret


def lateral_flow_iterator(
    qlat_file,
) -> Iterator[tuple[datetime, dict[UUID, float]]]:
    alldata = read_lateral_flow(qlat_file)
    for dt, data in alldata:
        yield dt, data


def write_output(
    filepath: Path,
    srcdata: tuple[list[datetime], dict[UUID, list[float]]],
):
    filepath.unlink(missing_ok=True)
    dts, data = srcdata
    table = pyarrow.table(
        [
            pyarrow.array(dts, type=pyarrow.timestamp("ms", tz="UTC")),
        ],
        names=[
            "datetime",
        ],
    )
    for rid, rdata in data.items():
        table = table.append_column(
            str(rid), pyarrow.array(rdata, type=pyarrow.float32())
        )
    pyarrow.parquet.write_table(table, filepath)


def routing_main(
    river_setup: Path,
    river_parameter: Path,
    init_file: Path,
    step: timedelta,
    qlat_file: Path,
    qout_file: Path,
    model: str = "muskingum",
):
    muskingumdata = MuskingumData(
        river_setup, river_parameter, init_file, qlat_file, step, model
    )

    simulation = Simulation(
        muskingumdata.topo,
        muskingumdata.init,
        muskingumdata.length,
        muskingumdata.weight,
        muskingumdata.celerity,
        muskingumdata.timestep,
        iter(muskingumdata.qlat),
        None,
        muskingumdata.model,
    )
    qout_data = collect_output(set(simulation.id), simulation.run())
    if not qout_file.parent.is_dir():
        qout_file.parent.mkdir(parents=True, exist_ok=True)
    write_output(qout_file, qout_data)


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
        "river_parameter",
        type=Path,
        help="Path to river network parameter file (Arrow)",
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
        "qlat",
        type=Path,
        help="Path to lateral flow input file (Parquet)",
    )
    parser.add_argument("qout", type=Path, help="Output file path (Parquet)")
    parser.add_argument(
        "--model",
        type=str,
        choices=["muskingum", "pass_through"],
        default="muskingum",
        help="River routing model to use",
    )
    args = parser.parse_args()

    routing_main(
        args.river_setup,
        args.river_parameter,
        args.init,
        args.step,
        args.qlat,
        args.qout,
        args.model,
    )
