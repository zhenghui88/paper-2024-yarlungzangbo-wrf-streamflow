#!/usr/bin/env python3
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar
from uuid import UUID

T = TypeVar("T")

type River = UUID
type Gauge = str

# downstream river of each river, None if no downstream river
type RiverNetworkDown = dict[River, River | None]

# upstream rivers of each river, empty set if no upstream river
type RiverNetworkUp = dict[River, set[River]]


@dataclass
class GaugeInfo:
    at: River
    upstream: set[Gauge]
    downstream: set[Gauge]
    rivers: set[River]


type GaugeNetwork = dict[Gauge, GaugeInfo]


def gauge_network_save(gaugenet: GaugeNetwork, filepath: Path):
    import json

    data = {
        str(k): {
            "at": str(v.at),
            "upstream": [str(vv) for vv in v.upstream],
            "downstream": [str(vv) for vv in v.downstream],
            "river": [str(vv) for vv in v.rivers],
        }
        for k, v in gaugenet.items()
    }

    with open(filepath, "wt") as f:
        json.dump(data, f, indent=2)


def gauge_newtwork_load(filepath: Path) -> GaugeNetwork:
    import json

    with open(filepath, "rt") as f:
        data = json.load(f)

    gaugenet: GaugeNetwork = {}
    for k, v in data.items():
        gaugenet[k] = GaugeInfo(
            at=UUID(v["at"]),
            upstream=set(v["upstream"]),
            downstream=set(v["downstream"]),
            rivers=set(UUID(r) for r in v["river"]),
        )
    return gaugenet


def build_gauge_network(
    gauges: dict[River, Gauge],
    gauge_downstream_gauges: dict[Gauge, Gauge | None],
    gauge_upstream_reaches: dict[Gauge, set[River]],
) -> GaugeNetwork:
    assert set(gauge_downstream_gauges.keys()) == set(gauge_upstream_reaches.keys()), (
        "gauges do not match"
    )
    gauge_at = {v: k for k, v in gauges.items()}

    gauge_downstream = {
        g: {}
        if v is None
        else {
            v,
        }
        for g, v in gauge_downstream_gauges.items()
    }

    gauge_upstream = {g: set() for g in gauge_downstream_gauges.keys()}
    for gauge, ds in gauge_downstream.items():
        for d in ds:
            gauge_upstream[d].add(gauge)

    gauge_river = {g: r.copy() for g, r in gauge_upstream_reaches.items()}
    for gauge, grs in gauge_river.items():
        for ugauge in gauge_upstream[gauge]:
            grs -= gauge_upstream_reaches[ugauge]

    return {
        g: GaugeInfo(
            gauge_at[g], gauge_upstream[g], gauge_downstream[g], gauge_river[g]
        )
        for g in gauge_downstream_gauges.keys()
    }


def gauge_find_downstream_gauges(
    gauges: dict[River, Gauge], rivernetwork_down: RiverNetworkDown
) -> dict[Gauge, Gauge | None]:
    downstreams: dict[Gauge, Gauge | None] = {g: None for g in gauges.values()}
    for start, gauge in gauges.items():
        current = rivernetwork_down[start]
        while current in rivernetwork_down:
            if current in gauges:
                downstreams[gauge] = gauges[current]
                break
            current = rivernetwork_down[current]
    return downstreams


def gauge_find_upstream_rivers(
    gauges: dict[River, Gauge], rivernetwork_up: RiverNetworkUp
) -> dict[Gauge, set[River]]:
    from collections import defaultdict

    upstreams: dict[Gauge, set[River]] = defaultdict(set)
    for start, code in gauges.items():
        current: set[River] = {start} if start in rivernetwork_up else set()
        upper: set[River] = set()
        while current:
            upper.clear()
            # find upper of current
            for rid in current:
                upstreams[code].add(rid)
                for uid in rivernetwork_up[rid]:
                    if uid in rivernetwork_up:
                        upper.add(uid)
            current, upper = upper, current
    return upstreams


def save_gauge_upstreams(gaugenet: dict[T, T | None], filepath: Path):
    import json

    data = {str(k): [str(r) for r in v] for k, v in gaugenet.items()}

    with open(filepath, "wt") as f:
        json.dump(data, f, indent=2)


def river_network_read(filepath: Path) -> RiverNetworkDown:
    """
    read river network from a SQLite database file

    The database is expected to have a table named 'centerline' with columns:
    - uuid: TEXT, the unique identifier of the river segment
    - downstream: TEXT, the unique identifier of the downstream river segment (or NULL if none)

    Returns a dictionary mapping each river segment's UUID to its downstream segment's UUID (or None).
    """
    import sqlite3

    rivernet: RiverNetworkDown = {}

    with sqlite3.connect(filepath) as conn:
        cursor = conn.cursor()
        cursor = cursor.execute("SELECT uuid, downstream FROM centerline")
        for rid_str, did_str in cursor.fetchall():
            rid = UUID(rid_str)
            did = UUID(did_str) if did_str else None
            rivernet[rid] = did
    return rivernet


def river_network_build_up(rivernet: RiverNetworkDown) -> RiverNetworkUp:
    """
    Find the adjacent upstream rivers for each river in the river network.
    """
    upstreams: RiverNetworkUp = {river: set() for river in rivernet.keys()}
    for river in rivernet:
        down = rivernet[river]
        if down is not None:
            upstreams[down].add(river)
    return upstreams


def gauge_river_mapping_read(filepath: Path) -> dict[River, Gauge]:
    """
    Read gauge information from a CSV file.

    The CSV file is expected to have columns:
    - reach: the unique identifier of the river gauge
    - stcd: the code of the river gauge
    """
    import csv

    gauges: dict[River, Gauge] = {}

    with open(filepath, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            gauges[UUID(row["river"])] = str(row["stcd"])
    return gauges


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="""Build the gauge network from river network and gauge locations.""",
    )
    parser.add_argument(
        "rivernetwork",
        type=Path,
        help="A SQLite database file with a table 'centerline' containing text columns 'uuid' and 'downstream', which are string representation of UUIDs.",
    )
    parser.add_argument(
        "gauges",
        type=Path,
        help="A CSV file with columns 'river' (UUID) and 'stcd' (string).",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="JSON file to save the gauge network. A map from gauge code (stcd) to its upstream and downstream gauges and the set of river segments (river UUID) upstream of the gauge and downstream of the upstream gauges.",
    )
    args = parser.parse_args()

    rivernetworkdown = river_network_read(args.rivernetwork)
    gauges = gauge_river_mapping_read(args.gauges)
    rivernetworkup = river_network_build_up(rivernetworkdown)
    gauges_downstream_gauges = gauge_find_downstream_gauges(gauges, rivernetworkdown)
    gauge_upstream_rivers = gauge_find_upstream_rivers(gauges, rivernetworkup)
    gauge_network = build_gauge_network(
        gauges, gauges_downstream_gauges, gauge_upstream_rivers
    )
    gauge_network_save(gauge_network, args.output.with_suffix(".json"))


if __name__ == "__main__":
    main()
