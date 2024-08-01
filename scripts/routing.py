import concurrent.futures
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numba
import numpy as np
import polars as pl
from numpy.typing import NDArray


@numba.njit
def muskingum(
    dt: float,
    dl: float,
    qin: float,
    qp: float,
    qinp: float,
    qlat: float,
    c: float,
    w: float,
):
    k = 0.5 * c * dt / dl
    aa = 1 - w + k
    bb = k - w
    cc = 1 - w - k
    dd = w + k
    ee = 2 * k
    q = (bb * qin + cc * qp + dd * qinp + ee * qlat) / aa
    return q


@numba.njit
def routing_step(
    to: NDArray[np.int_],
    length: NDArray[np.floating],
    dt: float,
    q: NDArray[np.floating],
    qp: NDArray[np.floating],
    qin: NDArray[np.floating],
    qinp: NDArray[np.floating],
    qlat: NDArray[np.floating],
    c: NDArray[np.floating],
    w: float,
):
    qin[:] = 0.0
    for ii, it in enumerate(to):
        q[ii] = muskingum(dt, length[ii], qin[ii], qp[ii], qinp[ii], qlat[ii], c[ii], w)
        if it >= 0:
            qin[it] = qin[it] + q[ii]


def routing(
    dt_ratio: int,
    rivnet: pl.DataFrame,
    qlat: pl.DataFrame,
    c: dict[str, float],
    w: float = 0.3,
):
    qout = np.zeros((qlat.height, rivnet.height), dtype=np.float32)

    q = np.zeros((rivnet.height), dtype=np.float32)
    qp = np.zeros_like(q)
    qin = np.zeros_like(q)
    qinp = np.zeros_like(q)

    to = np.where(np.isnan(rivnet[:, "to"].to_numpy()), -1, rivnet[:, "to"].to_numpy()).astype(np.int32)
    length = np.array(rivnet[:, "length"], dtype=np.float32)

    celerity = np.full_like(q, 3.0, dtype=np.float32)
    for k, v in c.items():
        celerity[rivnet[:, "gauge"] == k] = v

    for i in range(1, qlat.height):
        dt = float((qlat[i, "tm"] - qlat[i - 1, "tm"]).total_seconds())
        ql = qlat[i, 1:].to_numpy().squeeze()
        dt_step = dt / dt_ratio
        for _ in range(dt_ratio):
            routing_step(to, length, dt_step, q, qp, qin, qinp, ql, celerity, w)
            q, qp = qp, q
            qin, qinp = qinp, qin
        qout[i, :] = q
    qout = qlat.select(pl.col("tm")).hstack(
        pl.from_numpy(
            qout.astype(np.float32),
            schema=[(i, pl.Float32) for i in rivnet["uuid"]],
            orient="row",
        )
    )
    return qout


def run_case(rivnet: pl.DataFrame, qlat: pl.DataFrame, c: dict[str, float], outfile: Path):
    w = 0.3
    qout = routing(12, rivnet, qlat, c, w)
    qout.write_parquet(outfile)


def run_case_worker(args):
    run_case(*args)


@dataclass
class Skill:
    nse: float
    kge: float
    alpha: float
    beta: float
    rho: float


def calculate_skill(qobs, qout, start, stop):
    assert qobs.width == 2, "qobs must have two columns"
    assert qout.width == 2, "qout must have two columns"
    qobsout = (
        qobs.filter((pl.col("tm") >= start) & (pl.col("tm") <= stop)).join(qout, on="tm", how="inner").drop_nulls()
    )
    beta = qobsout[:, 2].mean() / qobsout[:, 1].mean()
    alpha = qobsout[:, 2].std() / qobsout[:, 1].std()
    rho = qobsout[:, 1:].corr()[0, 1]
    kge = 1.0 - float(np.sqrt((1 - rho) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
    nse = 1 - float((qobsout[:, 2] - qobsout[:, 1]).pow(2).sum() / (qobsout[:, 1] - qobsout[:, 1].mean()).pow(2).sum())
    return Skill(nse, kge, alpha, beta, rho)


if __name__ == "__main__":
    DATAROOT = Path("../data")
    DATETIME_START = datetime(2013, 6, 20)
    DATETIME_END = datetime(2013, 10, 1)

    WEIGHT_FILE = DATAROOT.joinpath("rivnet.parquet")
    rivnet = pl.read_parquet(WEIGHT_FILE, columns=["uuid", "to", "length", "gauge"])

    GAUGE_FILE = DATAROOT.joinpath("gauge_q_loc.csv")
    gaugeinfo = pl.read_csv(GAUGE_FILE)

    QOBS_FILE = DATAROOT.joinpath("gauge_q_obs.parquet")
    qobs = pl.read_parquet(QOBS_FILE)

    nexp = 5

    gauges = ["Lazi", "Nugesha", "Yangcun", "Nuxia"]
    celerity = pl.DataFrame([pl.Series(g, [np.nan for _ in range(nexp)], pl.Float32) for g in gauges])
    celerity_guesses = [float(x) for x in np.arange(0.1, 6.1, 0.1)]

    for gauge in gauges:
        args = []
        qoutroot = DATAROOT.joinpath("qout", f"qout_{gauge.lower()}")
        qoutroot.mkdir(parents=True, exist_ok=True)
        for exp in range(5):
            qlat = pl.read_parquet(DATAROOT.joinpath("qlat", f"qlat_{exp:02d}.parquet"))
            for c in celerity_guesses:
                desfile = qoutroot.joinpath(f"exp{exp}_{c:.2f}.parquet")
                args.append(
                    (
                        rivnet,
                        qlat,
                        {
                            "Lazi": celerity[exp, "Lazi"] if np.isfinite(celerity[exp, "Lazi"]) else c,
                            "Nugesha": celerity[exp, "Nugesha"] if np.isfinite(celerity[exp, "Nugesha"]) else c,
                            "Yangcun": celerity[exp, "Yangcun"] if np.isfinite(celerity[exp, "Yangcun"]) else c,
                            "Nuxia": celerity[exp, "Nuxia"] if np.isfinite(celerity[exp, "Nuxia"]) else c,
                        },
                        desfile,
                    )
                )
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(run_case_worker, args)
            for _ in results:
                pass

        gaugereach = str(gaugeinfo.filter(pl.col("gauge") == gauge).select(pl.col("reach")).item()).lower()
        kges: dict[int, tuple[list[float], list[Skill]]] = {}
        for filepath in sorted(qoutroot.glob("exp*.parquet")):
            exp = int(filepath.stem.split("_")[0].removeprefix("exp"))
            cc = float(filepath.stem.split("_")[-1])
            qout = pl.read_parquet(filepath)
            skill = calculate_skill(qobs["tm", gauge], qout["tm", gaugereach], DATETIME_START, DATETIME_END)
            if exp not in kges:
                kges[exp] = ([], [])
            kges[exp][0].append(cc)
            kges[exp][1].append(skill)
        for exp in sorted(kges.keys()):
            max_rho = float("-inf")
            max_rho_celerity = 0
            for cc, ss in zip(kges[exp][0], kges[exp][1]):
                if ss.rho > max_rho:
                    max_rho = ss.rho
                    max_rho_celerity = cc
            print(exp, max_rho, max_rho_celerity)
            celerity[exp, gauge] = max_rho_celerity
        print(celerity)
    celerity.write_csv(DATAROOT.joinpath("celerity_optimal.csv"), float_precision=2)
