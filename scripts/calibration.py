import logging
import math
from collections.abc import Iterator, Set
from datetime import datetime, timedelta
from pathlib import Path
from uuid import UUID

import pyarrow
import pyarrow.parquet
from gauge_network import (
    GaugeNetwork,
    gauge_newtwork_load,
)
from routing import MuskingumData
from routing_common import Simulation

CELERITY_GUESSES: list[float] = [0.05 + x * 0.05 for x in range(220)]


def correlation_coefficient(
    obs: tuple[list[datetime], list[float | None]],
    sim: tuple[list[datetime], list[float]],
):
    """Correlation coefficient as skill score"""
    # find the common time steps
    obs_dict: dict[datetime, float] = {
        t: v for t, v in zip(*obs) if v is not None and math.isfinite(v)
    }
    sim_dict: dict[datetime, float] = {t: v for t, v in zip(*sim) if math.isfinite(v)}
    common_times = sorted(set(obs_dict.keys()) & set(sim_dict.keys()))

    cc = float("nan")
    if len(common_times) >= 2:
        n = len(common_times)
        obs_vals = [obs_dict[t] for t in common_times]
        sim_vals = [sim_dict[t] for t in common_times]
        obs_mean = sum(obs_vals) / n
        sim_mean = sum(sim_vals) / n
        obs_anom = [v - obs_mean for v in obs_vals]
        sim_anom = [v - sim_mean for v in sim_vals]
        obs_var = math.sumprod(obs_anom, obs_anom)
        sim_var = math.sumprod(sim_anom, sim_anom)
        cov = math.sumprod(obs_anom, sim_anom)
        if obs_var > 0 and sim_var > 0:
            cc = cov / math.sqrt(obs_var * sim_var)
        else:
            cc = float("nan")
    return cc


def collect_simulation(
    sim: Iterator[tuple[datetime, dict[UUID, float]]],
    rivers: Set[UUID],
    *,
    ensure: UUID | None = None,
):
    assert ensure is None or ensure in rivers, "ensure river not in rivers"
    dts: list[datetime] = []
    riverdata: dict[UUID, list[float]] = {r: [] for r in rivers}
    for dt, qout in sim:
        assert ensure is None or ensure in qout, "ensure river not in output"
        dts.append(dt)
        for rid in rivers:
            riverdata[rid].append(qout.get(rid, 0.0))
    return dts, riverdata


def run_optimization(
    gaugecode: str,
    muskingumdata: MuskingumData,
    gaugenet: GaugeNetwork,
    qout_yet: tuple[list[datetime], dict[UUID, list[float]]],
    gaugeobs: tuple[list[datetime], list[float | None]],
) -> tuple[
    list[float],
    float,
    float,
    tuple[list[datetime], dict[UUID, list[float]]],
]:
    best_celerity: float = float("nan")
    best_score: float = float("-inf")
    best_qout: tuple[list[datetime], dict[UUID, list[float]]] = ([], {})

    # river network topography
    gauge2river: dict[str, UUID] = {k: v.at for k, v in gaugenet.items()}
    riverids: set[UUID] = set(gaugenet[gaugecode].rivers)
    topo = {
        k: v if v in riverids else None
        for k, v in muskingumdata.topo.items()
        if k in riverids
    }

    # upstream flow
    qdown: list[tuple[datetime, dict[UUID, float]]] = []
    for k in gaugenet[gaugecode].upstream:
        kk = gauge2river[k]
        assert kk in qout_yet[1], f"upstream gauge ({k}, {kk}) not simulated yet"
    if len(gaugenet[gaugecode].upstream) > 0:
        for i, dt in enumerate(qout_yet[0]):
            qdown_data: dict[UUID, float] = {}
            for upgauge in gaugenet[gaugecode].upstream:
                uprid = gauge2river[upgauge]
                rid = muskingumdata.topo[uprid]
                assert rid is not None, f"upstream river ({uprid}) has no downstream"
                qdown_data[rid] = qout_yet[1][uprid][i]
            qdown.append((dt, qdown_data))

    scores: list[float] = []
    for celerity_guess in CELERITY_GUESSES:
        celerity = {k: celerity_guess for k in topo.keys()}
        simulation = Simulation(
            topo,
            muskingumdata.init,
            muskingumdata.length,
            muskingumdata.weight,
            celerity,
            muskingumdata.timestep,
            iter(muskingumdata.qlat),
            iter(qdown) if len(qdown) > 0 else None,
            muskingumdata.model,
        )
        qout = collect_simulation(
            simulation.run(), riverids, ensure=gauge2river[gaugecode]
        )
        score = correlation_coefficient(
            gaugeobs, (qout[0], qout[1][gauge2river[gaugecode]])
        )
        scores.append(score)
        if score > best_score:
            logging.info(
                f"Gauge {gaugecode}: celerity {celerity_guess:.2f} with score {score:.4f}, new best"
            )
            best_score = score
            best_celerity = celerity_guess
            best_qout = qout
        else:
            logging.info(
                f"Gauge {gaugecode}: celerity {celerity_guess:.2f} with score {score:.4f}, not better than best {best_score:.4f}"
            )
    return scores, best_celerity, best_score, best_qout


def optimization(
    muskingumdata: MuskingumData,
    gaugenet: GaugeNetwork,
    gaugeobs: tuple[list[datetime], dict[str, list[float | None]]],
    parameter_path: Path,
    simulation_path: Path,
):
    gauge_list = sort_gauge_network(gaugenet)
    all_scores: dict[str, list[float]] = {}
    best_scores: dict[str, float] = {}
    best_guess: dict[str, float] = {}
    qout_yet: tuple[list[datetime], dict[UUID, list[float]]] = ([], {})
    for gauge in gauge_list:
        logging.info(f"Optimizing gauge {gauge}")
        scores, celerity, score, qout = run_optimization(
            gauge, muskingumdata, gaugenet, qout_yet, (gaugeobs[0], gaugeobs[1][gauge])
        )
        best_scores[gauge] = score
        best_guess[gauge] = celerity
        all_scores[gauge] = scores
        qout_yet = (qout[0], qout_yet[1] | qout[1])
    result = {
        gauge: {
            "best_celerity": best_guess[gauge],
            "best_score": best_scores[gauge],
            "celerities": CELERITY_GUESSES,
            "scores": all_scores[gauge],
        }
        for gauge in gauge_list
    }
    with open(parameter_path, "wt") as f:
        import json

        json.dump(result, f, indent=2)
    table = pyarrow.table(
        [
            pyarrow.array(qout_yet[0], type=pyarrow.timestamp("ms", tz="UTC")),
        ],
        names=[
            "datetime",
        ],
    )
    for k, v in qout_yet[1].items():
        table = table.append_column(str(k), pyarrow.array(v, type=pyarrow.float32()))
    pyarrow.parquet.write_table(table, simulation_path)


def sort_gauge_network(gaugenet: GaugeNetwork) -> list[str]:
    zheng_order: dict[str, int] = {}

    # find the most downstream gauges
    current: set[str] = set()
    for k, v in gaugenet.items():
        if len(v.downstream) == 0 or all(d not in gaugenet for d in v.downstream):
            current.add(k)
    for k in current:
        zheng_order[k] = 0

    down: set[str] = set()
    # the rest
    while len(zheng_order) < len(gaugenet):
        current, down = down, current
        if len(down) == 0:
            raise ValueError("Cycle detected in gauge network")
        for did in down:
            for gid in gaugenet[did].upstream:
                zheng_order[gid] = zheng_order[did] - 1
                current.add(gid)
    return [k for k, _ in sorted(zheng_order.items(), key=lambda item: item[1])]


def gauge_observation_read(
    filepath: Path,
) -> tuple[list[datetime], dict[str, list[float | None]]]:
    import pyarrow.parquet

    table = pyarrow.parquet.read_table(filepath)
    colnames = list(table.column_names)
    dts = table["tm"].to_pylist()
    data = {k: table[k].to_pylist() for k in colnames if k != "tm"}
    return dts, data


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Muskingum river routing with segmental optimization"
    )
    parser = argparse.ArgumentParser(
        description="Muskingum river routing with segmental optimization"
    )
    _ = parser.add_argument(
        "river_setup", type=Path, help="Path to river setup file (Arrow)"
    )
    _ = parser.add_argument(
        "river_parameter",
        type=Path,
        help="Path to river network parameter file (Arrow)",
    )
    _ = parser.add_argument(
        "init", type=Path, help="Path to river initial condition file (Arrow)"
    )
    _ = parser.add_argument(
        "step",
        type=lambda x: timedelta(seconds=float(x)),
        help="Maximum time step in seconds",
    )
    _ = parser.add_argument(
        "qlat",
        type=Path,
        help="Path to lateral flow input file (Parquet)",
    )
    _ = parser.add_argument(
        "--model",
        type=str,
        choices=["muskingum", "pass_through"],
        default="muskingum",
        help="River routing model to use",
    )
    _ = parser.add_argument(
        "gauge_network", type=Path, help="Gauge topology file (Json)"
    )
    _ = parser.add_argument(
        "gauge_observation", type=Path, help="Gauge observation file (Parquet)"
    )
    _ = parser.add_argument(
        "optimized_simulation",
        type=Path,
        help="Output optimized simulation file (Parquet)",
    )
    _ = parser.add_argument(
        "optimized_parameter", type=Path, help="Output optimized parameter file (json)"
    )
    args = parser.parse_args()
    muskingumdata = MuskingumData(
        args.river_setup,
        args.river_parameter,
        args.init,
        args.qlat,
        args.step,
        args.model,
    )
    gauge_network = gauge_newtwork_load(args.gauge_network)
    gauge_observation = gauge_observation_read(args.gauge_observation)
    optimization(
        muskingumdata,
        gauge_network,
        gauge_observation,
        args.optimized_parameter,
        args.optimized_simulation,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )
    main()
