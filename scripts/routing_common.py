import itertools
import logging
import math
from collections.abc import Iterator, Mapping, Set
from datetime import datetime, timedelta
from uuid import UUID

import numba
import numpy as np


@numba.njit
def muskingum_river(
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
    aa = 1.0 - w + k
    bb = k - w
    cc = 1.0 - w - k
    dd = w + k
    ee = 2.0 * k
    q = (bb * qin + cc * qp + dd * qinp + ee * qlat) / aa
    return q


@numba.njit
def muskingum_step(
    to: np.ndarray[tuple[int], np.dtype[np.int_]],
    length: np.ndarray[tuple[int], np.dtype[np.floating]],
    dt: float,
    q: np.ndarray[tuple[int], np.dtype[np.floating]],
    qp: np.ndarray[tuple[int], np.dtype[np.floating]],
    qin: np.ndarray[tuple[int], np.dtype[np.floating]],
    qinp: np.ndarray[tuple[int], np.dtype[np.floating]],
    qlat: np.ndarray[tuple[int], np.dtype[np.floating]],
    c: np.ndarray[tuple[int], np.dtype[np.floating]],
    w: np.ndarray[tuple[int], np.dtype[np.floating]],
):
    for ii, it in enumerate(to):
        q[ii] = muskingum_river(
            dt, length[ii], qin[ii], qp[ii], qinp[ii], qlat[ii], c[ii], w[ii]
        )
        if it >= 0:
            qin[it] = qin[it] + q[ii]


@numba.njit
def pass_through_step(
    to: np.ndarray[tuple[int], np.dtype[np.int_]],
    q: np.ndarray[tuple[int], np.dtype[np.floating]],
    qin: np.ndarray[tuple[int], np.dtype[np.floating]],
    qlat: np.ndarray[tuple[int], np.dtype[np.floating]],
):
    for ii, it in enumerate(to):
        q[ii] = qin[ii] + qlat[ii]
        if it >= 0:
            qin[it] = qin[it] + q[ii]


class Simulation:
    def __init__(
        self,
        topo: Mapping[UUID, UUID | None],
        init: tuple[datetime, Mapping[UUID, float]],
        length: Mapping[UUID, float],
        weight: Mapping[UUID, float],
        celerity: Mapping[UUID, float],
        timestep_max: timedelta,
        qlat: Iterator[tuple[datetime, Mapping[UUID, float]]],
        qdown: Iterator[tuple[datetime, Mapping[UUID, float]]] | None,
        model: str = "muskingum",
    ):
        topo_subnet = {
            k: v if (v is None or v in topo) else None for k, v in topo.items()
        }
        self.id: list[UUID] = self.sort_river_network(topo_subnet)
        self.id2index: dict[UUID, int] = {uid: i for i, uid in enumerate(self.id)}
        self.to: np.ndarray[tuple[int], np.dtype[np.int64]] = np.array(
            [
                self.id2index[to_uid] if to_uid is not None else -1
                for to_uid in (topo_subnet[uid] for uid in self.id)
            ],
            dtype=np.int64,
        )
        self.length: np.ndarray[tuple[int], np.dtype[np.float32]] = np.array(
            [length[uid] for uid in self.id], dtype=np.float32
        )
        self.weight: np.ndarray[tuple[int], np.dtype[np.float32]] = np.array(
            [weight[uid] for uid in self.id], dtype=np.float32
        )
        self.celerity: np.ndarray[tuple[int], np.dtype[np.float32]] = np.array(
            [celerity[uid] for uid in self.id], dtype=np.float32
        )

        self.timestep_max: timedelta = timestep_max
        if qdown is None:
            logging.debug("No downstream flow provided, assuming zero")
            iter1, iter2 = itertools.tee(qlat)
            self.qlat_iter = iter1

            def qdown_gen() -> Iterator[tuple[datetime, Mapping[UUID, float]]]:
                for dt, _ in iter2:
                    yield dt, {}

            self.qdown_iter = qdown_gen()
        else:
            self.qlat_iter: Iterator[tuple[datetime, Mapping[UUID, float]]] = qlat
            self.qdown_iter: Iterator[tuple[datetime, Mapping[UUID, float]]] = qdown

        self.dt = init[0]
        self.q: np.ndarray[tuple[int], np.dtype[np.float32]] = np.array(
            [init[1].get(rid, 0.0) for rid in self.id], dtype=np.float32
        )
        self.qp: np.ndarray[tuple[int], np.dtype[np.float32]] = np.zeros_like(self.q)
        self.qin: np.ndarray[tuple[int], np.dtype[np.float32]] = np.zeros_like(self.q)
        self.qinp: np.ndarray[tuple[int], np.dtype[np.float32]] = np.zeros_like(self.q)
        self.qlat: np.ndarray[tuple[int], np.dtype[np.float32]] = np.zeros_like(self.q)

        self.model = model

    def run(
        self,
    ) -> Iterator[tuple[datetime, dict[UUID, float]]]:
        logging.debug("Starting simulation")
        for (dt_qlat, qlat), (dt_qdown, qdown) in zip(self.qlat_iter, self.qdown_iter):
            while dt_qlat <= self.dt:
                dt_qlat, qlat = next(self.qlat_iter)
            while dt_qdown <= self.dt:
                dt_qdown, qdown = next(self.qdown_iter)
            assert dt_qlat == dt_qdown, (
                "Time steps of qlat, qdown, and model times do not match"
            )
            cdt = dt_qlat
            self.qlat[:] = np.array(
                [qlat.get(uid, 0.0) for uid in self.id], dtype=np.float32
            )
            dt = cdt - self.dt
            nstep = math.ceil(dt / self.timestep_max)
            dt_step = dt / nstep
            for i in range(nstep):
                self.q, self.qp = self.qp, self.q
                self.qin, self.qinp = self.qinp, self.qin
                self.qin[:] = np.array(
                    [qdown.get(uid, 0.0) for uid in self.id], dtype=np.float32
                )
                logging.debug(
                    f"model step from {self.dt + i * dt_step} to {self.dt + (i + 1) * dt_step}, qlat_dt = {dt_qlat}, qdown_dt = {dt_qdown}"
                )
                match self.model:
                    case "pass_through":
                        pass_through_step(self.to, self.q, self.qin, self.qlat)
                    case "muskingum":
                        muskingum_step(
                            self.to,
                            self.length,
                            dt_step.total_seconds(),
                            self.q,
                            self.qp,
                            self.qin,
                            self.qinp,
                            self.qlat,
                            self.celerity,
                            self.weight,
                        )
            self.dt = cdt
            yield cdt, {uid: float(self.qp[i]) for i, uid in enumerate(self.id)}

    @staticmethod
    def sort_river_network(to: dict[UUID, UUID | None]) -> list[UUID]:
        """
        Sort river segments so that upstream segments come before downstream segments.
        """
        zheng_order: dict[UUID, int] = {}
        ups: dict[UUID, set[UUID]] = {k: set() for k in to}
        for k, v in to.items():
            if v is not None and v in ups:
                ups[v].add(k)

        # find the most downstream rivers
        current: set[UUID] = set()
        for k, v in to.items():
            if v is None or v not in to:
                current.add(k)
        for k in current:
            zheng_order[k] = 0

        down: set[UUID] = set()
        # the rest
        while len(zheng_order) < len(to):
            current, down = down, current
            if len(down) == 0:
                raise ValueError("Cycle detected in river network")
            for did in down:
                for rid in ups[did]:
                    zheng_order[rid] = zheng_order[did] - 1
                    current.add(rid)

        return [k for k, _ in sorted(zheng_order.items(), key=lambda item: item[1])]


def collect_output(
    rivers: Set[UUID],
    srcdata: Iterator[tuple[datetime, dict[UUID, float]]],
) -> tuple[list[datetime], dict[UUID, list[float]]]:
    dts: list[datetime] = []
    data: dict[UUID, list[float]] = {rid: [] for rid in rivers}
    for dt, qout in srcdata:
        dts.append(dt)
        for rid in rivers:
            data[rid].append(qout.get(rid, 0.0))
    return dts, data
