# %%
import json
import string
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypedDict
from uuid import UUID

import matplotlib as mpl
import matplotlib.dates as mpldates
import matplotlib.pyplot as plt
import matplotlib.typing as mpltyping
import numpy as np
import polars as pl
import scipy.stats as stats

# %%
# plotting settings
DATAROOT = Path("data")
FIGROOT = Path("fig")
FIGDPI = 600
FIGSUFFIX = ".pdf"


plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update(
    {
        "savefig.pad_inches": 0.05 / 2.54,
        "grid.linewidth": 0.25,
        "axes.titlesize": 8,
        "axes.titlepad": 0.05,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "text.usetex": False,
    }
)

# %%
DATETIME_START = datetime(2013, 6, 20, tzinfo=UTC)
DATETIME_STOP = datetime(2013, 10, 1, tzinfo=UTC) + timedelta(seconds=1)

EXPS = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13"]
SELECTED_EXPS = EXPS.copy()
SELECTED_EXPS.remove("10")

NPARAMETERS = 100

GAUGE_TOPO_FILE = DATAROOT.joinpath("gauge_network.json")
GAUGE_LOCATION_FILE = DATAROOT.joinpath("gauge_q_loc.csv")

QOBS_FILE = DATAROOT.joinpath("gauge_q_obs.parquet")

QSIM_ROOT = DATAROOT.joinpath("qout_ensemble")

# %%
# gauge information

gauge_location = pl.read_csv(GAUGE_LOCATION_FILE, schema_overrides={"stcd": pl.Utf8})


class GaugeInfo(TypedDict):
    name: str
    latitude: float
    longitude: float
    at: UUID
    upstream: set[str]
    downstream: set[str]
    rivers: set[UUID]


gaugenetwork: dict[str, GaugeInfo] = {}
with open(GAUGE_TOPO_FILE) as f:
    _gauge_topo = json.load(f)
    for gauge, info in _gauge_topo.items():
        gaugenetwork[gauge] = GaugeInfo(
            name=str(gauge_location.filter(pl.col("stcd") == gauge)[0, "gauge"]),
            latitude=float(gauge_location.filter(pl.col("stcd") == gauge)[0, "lat"]),
            longitude=float(gauge_location.filter(pl.col("stcd") == gauge)[0, "lon"]),
            at=UUID(info["at"]),
            upstream=set(info["upstream"]),
            downstream=set(info["downstream"]),
            rivers={UUID(rid) for rid in info["river"]},
        )
print(
    f"there are {sum(len(info['rivers']) for info in gaugenetwork.values())} rivers in total."
)

# %%
gaugeshortname = {
    "90604500": "NX",
    "90603000": "YC",
    "90602000": "NGS",
    "90601000": "LZ",
    "90802500": "LS",
    "90901200": "GZ",
}
gaugefullname = {
    "90604500": "Nuxia",
    "90603000": "Yangcun",
    "90602000": "Nugesha",
    "90601000": "Lazi",
    "90802500": "Lhasa",
    "90901200": "Gengzhang",
}
# gaugeshortname = {"Nuxia": "奴下", "Yangcun": "杨村", "Nugesha":"努各沙", "Lazi":"拉孜"}[gaugename]
gaugelist = ["90601000", "90602000", "90802500", "90603000", "90901200", "90604500"]

# %%
# read observed streamflow
qobs = pl.read_parquet(QOBS_FILE).filter(
    (pl.col("tm") >= DATETIME_START) & (pl.col("tm") <= DATETIME_STOP)
)
nobs = 0
for col in gaugelist:
    nobs += qobs.select(pl.col(col)).drop_nulls().shape[0]
print(f"there are {nobs} records of observed streamflow.")

# %%
# read simulated streamflow
qsims: dict[str, dict[str, pl.DataFrame]] = {gauge: {} for gauge in gaugelist}
for exp in EXPS:
    for iparam in range(1, NPARAMETERS + 1):
        qsim_file = QSIM_ROOT.joinpath(f"qout_{exp}_{iparam:03d}.parquet")
        qsim = pl.read_parquet(qsim_file).filter(
            (pl.col("datetime") >= DATETIME_START)
            & (pl.col("datetime") <= DATETIME_STOP)
        )
        for gauge in gaugelist:
            rid = str(gaugenetwork[gauge]["at"])
            data = qsim.select(
                pl.col("datetime").alias("tm"), pl.col(rid).alias(f"{iparam:03d}")
            )
            if exp in qsims[gauge]:
                newdata = data.join(
                    qsims[gauge][exp], on="tm", how="full", coalesce=True
                )
                qsims[gauge][exp] = newdata
            else:
                qsims[gauge][exp] = data


# %%
QSIM_PASS_ROOT = DATAROOT.joinpath("qout_passthrough")
qsim_pass: dict[str, dict[str, pl.DataFrame]] = {gauge: {} for gauge in gaugelist}
for exp in EXPS:
    qsim_file = QSIM_PASS_ROOT.joinpath(f"qout_{exp}.parquet")
    qsim = pl.read_parquet(qsim_file).filter(
        (pl.col("datetime") >= DATETIME_START) & (pl.col("datetime") <= DATETIME_STOP)
    )
    for gauge in gaugelist:
        rid = str(gaugenetwork[gauge]["at"])
        data = qsim.select(pl.col("datetime").alias("tm"), pl.col(rid).alias("pass"))
        qsim_pass[gauge][exp] = data

# %%
linecolors: dict[str, mpltyping.ColorType] = {
    c: mpl.color_sequences["tab20"][i] for i, c in enumerate(EXPS)
}
# linecolors: dict[str, str] = {
#     "01": "C0",
#     "02": "C1",
#     "03": "C2",
#     "04": "C3",
#     "05": "C4",
#     "06": "C5",
#     "07": "C6",
#     "08": "C7",
#     "09": "C8",
#     "10": "tab:gray",
#     "11": "C9",
#     "12": "tab:olive",
#     "13": "tab:cyan",
# }
linestyles: dict[str, str] = {
    "01": "-",
    "02": "-",
    "03": "-",
    "04": "-",
    "05": "-",
    "06": "-",
    "07": "-",
    "08": "-",
    "09": "-",
    "10": "-",
    "11": "-",
    "12": "-",
    "13": "-",
}

# %%
# plot streamflow time series
fig = plt.figure(figsize=(14 / 2.54, 14 / 2.54), dpi=FIGDPI, layout="constrained")
axs = fig.subplots(3, 2, sharex=True, squeeze=False)

ylims: dict[str, tuple[float, float]] = {
    "90604500": (0, 15000),
    "90603000": (0, 11000),
    "90602000": (0, 7500),
    "90601000": (0, 3000),
    "90802500": (0, 3000),
    "90901200": (0, 3500),
}

for igauge, gauge in enumerate(gaugelist):
    ax = axs.flatten()[igauge]
    ax.plot(qobs["tm"], qobs[gauge], "k.", markersize=1, label="Obs")
    for iexp, exp in enumerate(SELECTED_EXPS):
        ax.fill_between(
            qsims[gauge][exp]["tm"],
            qsims[gauge][exp].select(pl.all().exclude("tm")).min_horizontal(),
            qsims[gauge][exp].select(pl.all().exclude("tm")).max_horizontal(),
            color=linecolors[exp],
            linewidth=0,
            alpha=0.3,
        )
        ax.plot(
            qsims[gauge][exp]["tm"],
            qsims[gauge][exp].select(pl.all().exclude("tm")).mean_horizontal(),
            color=linecolors[exp],
            linewidth=1,
            linestyle=linestyles[exp],
            label=f"E{exp}",
        )
    ax.text(
        0.02,
        0.98,
        f"{string.ascii_lowercase[igauge]}) {gaugeshortname[gauge]}",
        ha="left",
        va="top",
        transform=ax.transAxes,
        fontsize=12,
    )
    # ax.title(
    #     f"{string.ascii_lowercase[igauge]}) {gaugefullname[gauge]}",
    # )
    ax.set_xlim(DATETIME_START, DATETIME_STOP)
    ax.xaxis.set_ticks(
        [
            datetime(2013, 6, 20),
            datetime(2013, 7, 1),
            datetime(2013, 7, 15),
            datetime(2013, 8, 1),
            datetime(2013, 8, 15),
            datetime(2013, 9, 1),
            datetime(2013, 9, 15),
            datetime(2013, 10, 1),
        ]
    )
    ax.xaxis.set_major_formatter(mpldates.DateFormatter("%m-%d"))
    ax.xaxis.set_tick_params(rotation=45)
    ax.set_ylim(ylims[gauge])
axs.flatten()[-2].legend(
    loc="upper right",
    frameon=False,
    fontsize=8,
    ncol=4,
    borderaxespad=0.1,
    borderpad=0.05,
    handlelength=0.8,
    labelspacing=0.1,
    handletextpad=0.1,
    columnspacing=0.2,
)
axs[1][0].set_ylabel("Streamflow (m³ s⁻¹)")

plt.savefig(
    FIGROOT.joinpath("streamflow_simulation").with_suffix(FIGSUFFIX),
    dpi=FIGDPI,
    bbox_inches="tight",
)


# %%
class Skill(TypedDict):
    kge: float
    g: tuple[float, float, float]
    alpha: float
    beta: float
    rho: float
    nse: float


def calculate_skill(obs, sim):
    beta = sim.mean() / obs.mean()
    alpha = sim.std() / obs.std()
    rho = np.corrcoef(obs, sim)[0, 1]
    s = (1 - rho) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2
    kge = 1.0 - float(np.sqrt(s))
    g = (
        float((alpha - 1) ** 2 / s),
        float((beta - 1) ** 2 / s),
        float((rho - 1) ** 2 / s),
    )
    nse = 1 - float(np.sum(np.pow(sim - obs, 2)) / np.sum(np.pow(obs - obs.mean(), 2)))
    return Skill(kge=kge, g=g, alpha=alpha, beta=beta, rho=rho, nse=nse)


# %%
qsim_nse: dict[str, dict[str, list[float]]] = {gauge: {} for gauge in gaugelist}
qsim_kge: dict[str, dict[str, list[float]]] = {gauge: {} for gauge in gaugelist}
qsim_kge_g: dict[str, dict[str, list[tuple[float, float, float]]]] = {
    gauge: {} for gauge in gaugelist
}
qsim_rho: dict[str, dict[str, list[float]]] = {gauge: {} for gauge in gaugelist}

for gauge in gaugelist:
    obs = qobs.select(pl.col("tm"), pl.col(gauge).alias("obs"))
    for exp in EXPS:
        for iparam in range(1, NPARAMETERS + 1):
            sim = qsims[gauge][exp].select(
                pl.col("tm"), pl.col(f"{iparam:03d}").alias("sim")
            )
            obssim = obs.join(sim, on="tm", how="inner").drop_nulls()
            skill = calculate_skill(obssim["obs"].to_numpy(), obssim["sim"].to_numpy())
            qsim_kge[gauge].setdefault(exp, []).append(skill["kge"])
            qsim_rho[gauge].setdefault(exp, []).append(skill["rho"])
            qsim_nse[gauge].setdefault(exp, []).append(skill["nse"])
            qsim_kge_g[gauge].setdefault(exp, []).append(skill["g"])

# %%
qpass_nse: dict[str, dict[str, float]] = {gauge: {} for gauge in gaugelist}
qpass_kge: dict[str, dict[str, float]] = {gauge: {} for gauge in gaugelist}
qpass_kge_g: dict[str, dict[str, tuple[float, float, float]]] = {
    gauge: {} for gauge in gaugelist
}
qpass_rho: dict[str, dict[str, float]] = {gauge: {} for gauge in gaugelist}

for gauge in gaugelist:
    obs = qobs.select(pl.col("tm"), pl.col(gauge).alias("obs"))
    for exp in EXPS:
        sim = qsim_pass[gauge][exp].select(pl.col("tm"), pl.col("pass").alias("sim"))
        obssim = obs.join(sim, on="tm", how="inner").drop_nulls()
        skill = calculate_skill(obssim["obs"].to_numpy(), obssim["sim"].to_numpy())
        qpass_kge[gauge][exp] = skill["kge"]
        qpass_rho[gauge][exp] = skill["rho"]
        qpass_nse[gauge][exp] = skill["nse"]
        qpass_kge_g[gauge][exp] = skill["g"]

# %%
# save skills
with open(DATAROOT.joinpath("skills/streamflow_kge.json"), "wt") as f:
    json.dump(qsim_kge, f, indent=4)
with open(DATAROOT.joinpath("skills/streamflow_rho.json"), "wt") as f:
    json.dump(qsim_rho, f, indent=4)
with open(DATAROOT.joinpath("skills/streamflow_kge_g.json"), "wt") as f:
    json.dump(qsim_kge_g, f, indent=4)
with open(DATAROOT.joinpath("skills/passthrough_kge.json"), "wt") as f:
    json.dump(qpass_kge, f, indent=4)
with open(DATAROOT.joinpath("skills/passthrough_rho.json"), "wt") as f:
    json.dump(qpass_rho, f, indent=4)
with open(DATAROOT.joinpath("skills/passthrough_kge_g.json"), "wt") as f:
    json.dump(qpass_kge_g, f, indent=4)

# %%
qsim_rho_pvalue = np.full(
    (len(gaugelist), len(SELECTED_EXPS), len(SELECTED_EXPS)), np.nan
)
qsim_rho_rank = np.full((len(gaugelist), len(SELECTED_EXPS)), 0)
qsim_rho_order: dict[str, list[str]] = {gauge: [] for gauge in gaugelist}
for igauge, gauge in enumerate(gaugelist):
    qsim_rho_rank[igauge, :] = np.argsort(
        [np.median(qsim_rho[gauge][exp]) for exp in SELECTED_EXPS]
    )[::-1]
    qsim_rho_order[gauge] = [SELECTED_EXPS[i] for i in qsim_rho_rank[igauge, :]]
    for iexp, expi in enumerate(qsim_rho_order[gauge]):
        for jexp, expj in enumerate(qsim_rho_order[gauge]):
            x = qsim_rho[gauge][expi]
            y = qsim_rho[gauge][expj]
            res = stats.ttest_rel(x, y)
            qsim_rho_pvalue[igauge, iexp, jexp] = res.pvalue

# %%
qsim_kge_pvalue = np.full(
    (len(gaugelist), len(SELECTED_EXPS), len(SELECTED_EXPS)), np.nan
)
qsim_kge_rank = np.full((len(gaugelist), len(SELECTED_EXPS)), 0)
qsim_kge_order: dict[str, list[str]] = {gauge: [] for gauge in gaugelist}
for igauge, gauge in enumerate(gaugelist):
    qsim_kge_rank[igauge, :] = np.argsort(
        [np.median(qsim_kge[gauge][exp]) for exp in SELECTED_EXPS]
    )[::-1]
    qsim_kge_order[gauge] = [SELECTED_EXPS[i] for i in qsim_kge_rank[igauge, :]]
    for iexp, expi in enumerate(qsim_kge_order[gauge]):
        for jexp, expj in enumerate(qsim_kge_order[gauge]):
            x = qsim_kge[gauge][expi]
            y = qsim_kge[gauge][expj]
            res = stats.ttest_rel(x, y)
            qsim_kge_pvalue[igauge, iexp, jexp] = res.pvalue


# %%
fig = plt.figure(figsize=(10 / 2.54, 12 / 2.54), dpi=FIGDPI, layout="constrained")
axs = fig.subplots(
    3,
    2,
    squeeze=False,
)

for igauge, gauge in enumerate(gaugelist):
    ax = axs.flatten()[igauge]
    im = ax.pcolormesh(
        qsim_rho_pvalue[igauge, ::-1, :] < 0.05,
        cmap="Greys",
        vmin=0,
        vmax=1,
        edgecolors="k",
        linewidth=0.5,
        alpha=0.5,
    )
    labels = [f"E{e}" for e in qsim_rho_order[gauge]]
    ax.set_xticks(np.arange(len(SELECTED_EXPS)) + 0.5, labels=labels)
    ax.set_yticks(np.arange(len(SELECTED_EXPS)) + 0.5, labels=labels[::-1])
    ax.xaxis.set_tick_params(rotation=45)
    ax.set_title(f"{string.ascii_lowercase[igauge]}) {gaugeshortname[gauge]}")

plt.savefig(
    FIGROOT.joinpath("rho_difference_significance").with_suffix(FIGSUFFIX),
    dpi=FIGDPI,
    bbox_inches="tight",
)

# %%
fig = plt.figure(figsize=(10 / 2.54, 12 / 2.54), dpi=FIGDPI, layout="constrained")
axs = fig.subplots(
    3,
    2,
)

for igauge, gauge in enumerate(gaugelist):
    ax = axs.flatten()[igauge]
    im = ax.pcolormesh(
        qsim_kge_pvalue[igauge, ::-1, :] < 0.05,
        cmap="Greys",
        vmin=0,
        vmax=1,
        edgecolors="k",
        linewidth=0.5,
        alpha=0.5,
    )
    labels = [f"E{e}" for e in qsim_kge_order[gauge]]
    ax.set_xticks(np.arange(len(SELECTED_EXPS)) + 0.5, labels=labels)
    ax.set_yticks(np.arange(len(SELECTED_EXPS)) + 0.5, labels=labels[::-1])
    ax.xaxis.set_tick_params(rotation=45)
    ax.set_title(f"{string.ascii_lowercase[igauge]}) {gaugeshortname[gauge]}")
plt.savefig(
    FIGROOT.joinpath("kge_difference_significance").with_suffix(FIGSUFFIX),
    dpi=FIGDPI,
    bbox_inches="tight",
)


# %%
fig = plt.figure(figsize=(14 / 2.54, 12 / 2.54), dpi=FIGDPI, layout="constrained")
axs = fig.subplots(3, 2, sharey=True, squeeze=False)

flierprops = dict(
    marker=".", markerfacecolor="black", markersize=2, markeredgecolor="none"
)
medianprops = dict(linestyle="-", linewidth=1, color="k")


for igauge, gauge in enumerate(gaugelist):
    srhos = np.full((len(SELECTED_EXPS), NPARAMETERS), np.nan)
    for iexp, exp in enumerate(qsim_rho_order[gauge]):
        srhos[iexp, :] = qsim_rho[gauge][exp]
    ax = axs.flatten()[igauge]
    ax.boxplot(
        srhos.T,
        medianprops=medianprops,
        flierprops=flierprops,
    )
    ax.set_xticks(range(1, len(SELECTED_EXPS) + 1))
    ax.set_xticklabels([f"E{e}" for e in qsim_rho_order[gauge]])
    ax.xaxis.set_tick_params(rotation=45)
    ax.text(
        0.02,
        0.02,
        f"{string.ascii_lowercase[igauge]}) {gaugeshortname[gauge]}",
        ha="left",
        va="bottom",
        transform=ax.transAxes,
        fontsize=12,
    )
axs[-2][0].set_ylabel("Correlation Coefficient")

plt.savefig(
    FIGROOT.joinpath("streamflow_rho").with_suffix(FIGSUFFIX),
    dpi=FIGDPI,
    bbox_inches="tight",
)

# %%
fig = plt.figure(figsize=(14 / 2.54, 12 / 2.54), dpi=FIGDPI, layout="constrained")
axs = fig.subplots(3, 2, squeeze=False)

flierprops = dict(
    marker=".", markerfacecolor="black", markersize=2, markeredgecolor="none"
)
medianprops = dict(linestyle="-", linewidth=1, color="k")

for igauge, gauge in enumerate(gaugelist):
    ax = axs.flatten()[igauge]
    skge = np.full((len(SELECTED_EXPS), NPARAMETERS), np.nan)
    for iexp, exp in enumerate(qsim_kge_order[gauge]):
        skge[iexp, :] = qsim_kge[gauge][exp]
    ax.boxplot(
        skge.T,
        medianprops=medianprops,
        flierprops=flierprops,
    )
    ax.set_xticks(range(1, len(SELECTED_EXPS) + 1))
    ax.set_xticklabels([f"E{e}" for e in qsim_kge_order[gauge]])
    ax.xaxis.set_tick_params(rotation=45)
    ax.text(
        0.02,
        0.02,
        f"{string.ascii_lowercase[igauge]}) {gaugeshortname[gauge]}",
        ha="left",
        va="bottom",
        transform=ax.transAxes,
        fontsize=12,
    )
axs[-2][0].set_ylabel("Kling-Gupta Efficiency")

plt.savefig(
    FIGROOT.joinpath("streamflow_kge").with_suffix(FIGSUFFIX),
    dpi=FIGDPI,
    bbox_inches="tight",
)

# %%
# plot streamflow time series
fig = plt.figure(figsize=(14 / 2.54, 14 / 2.54), dpi=FIGDPI, layout="constrained")
axs = fig.subplots(3, 2, sharex=True, squeeze=False)

ylims: dict[str, tuple[float, float]] = {
    "90604500": (0, 40000),
    "90603000": (0, 30000),
    "90602000": (0, 20000),
    "90601000": (0, 10000),
    "90802500": (0, 10000),
    "90901200": (0, 10000),
}

for igauge, gauge in enumerate(gaugelist):
    ax = axs.flatten()[igauge]
    ax.plot(qobs["tm"], qobs[gauge], "k.", markersize=1, label="Obs")
    for iexp, exp in enumerate(SELECTED_EXPS):
        ax.plot(
            qsim_pass[gauge][exp]["tm"],
            qsim_pass[gauge][exp].select(pl.all().exclude("tm")).mean_horizontal(),
            color=linecolors[exp],
            linewidth=0.5,
            linestyle=linestyles[exp],
            alpha=0.5,
            label=f"E{exp}",
        )
    ax.text(
        0.02,
        0.98,
        f"{string.ascii_lowercase[igauge]}) {gaugeshortname[gauge]}",
        ha="left",
        va="top",
        transform=ax.transAxes,
        fontsize=12,
    )
    ax.set_xlim(DATETIME_START, DATETIME_STOP)
    ax.xaxis.set_ticks(
        [
            datetime(2013, 6, 20),
            datetime(2013, 7, 1),
            datetime(2013, 7, 15),
            datetime(2013, 8, 1),
            datetime(2013, 8, 15),
            datetime(2013, 9, 1),
            datetime(2013, 9, 15),
            datetime(2013, 10, 1),
        ]
    )
    ax.xaxis.set_major_formatter(mpldates.DateFormatter("%m-%d"))
    ax.xaxis.set_tick_params(rotation=45)
    ax.set_ylim(ylims[gauge])
axs.flatten()[-2].legend(
    loc="upper right",
    frameon=False,
    fontsize=8,
    ncol=4,
    borderaxespad=0.1,
    borderpad=0.05,
    handlelength=0.8,
    labelspacing=0.1,
    handletextpad=0.1,
    columnspacing=0.2,
)
axs[1][0].set_ylabel("Area-weighted Aggregation of Runoff (m³ s⁻¹)")

plt.savefig(
    FIGROOT.joinpath("streamflow_passthrough").with_suffix(FIGSUFFIX),
    dpi=FIGDPI,
    bbox_inches="tight",
)

# %%
