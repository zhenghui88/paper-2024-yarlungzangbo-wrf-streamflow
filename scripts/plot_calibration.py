# %%
import json
import string
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypedDict
from uuid import UUID

import matplotlib as mpl
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
# plt.style.use("default")
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
# time and data settings
DATETIME_START = datetime(2013, 6, 20, tzinfo=UTC)
DATETIME_STOP = datetime(2013, 10, 1, tzinfo=UTC) + timedelta(seconds=1)
EXPS = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13"]

GAUGE_TOPO_FILE = DATAROOT.joinpath("gauge_network.json")
GAUGE_LOCATION_FILE = DATAROOT.joinpath("gauge_q_loc.csv")

QOUTROOT = DATAROOT.joinpath("qout_calibration")

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
# gaugeshortname = {"Nuxia": "奴下", "Yangcun": "杨村", "Nugesha":"努各沙", "Lazi":"拉孜"}[gaugename]
gaugelist = ["90601000", "90602000", "90802500", "90603000", "90901200", "90604500"]


# %%
# read calibration results
class CalibrationInfo(TypedDict):
    best_celerity: float
    best_score: float
    celerity: list[float]
    score: list[float]


calibration: dict[str, dict[str, CalibrationInfo]] = defaultdict(dict)

for iexp, exp in enumerate(EXPS):
    calibfile = QOUTROOT.joinpath(f"qout_{exp}_parameter.json")
    with open(calibfile) as f:
        _calibration = json.load(f)
        for gauge, info in _calibration.items():
            calibration[gauge][exp] = CalibrationInfo(
                best_celerity=float(info["best_celerity"]),
                best_score=float(info["best_score"]),
                celerity=[float(c) for c in info["celerities"]],
                score=[float(s) for s in info["scores"]],
            )

# %%
# find out the outliers in the optimal correlation coefficient
best_celerity = np.full((len(gaugelist), len(EXPS)), np.nan)
best_cc = np.full((len(gaugelist), len(EXPS)), np.nan)
for igauge, gauge in enumerate(gaugelist):
    calibinfo = calibration[gauge]
    for iexp, exp in enumerate(EXPS):
        best_celerity[igauge, iexp] = calibinfo[exp]["best_celerity"]
        best_cc[igauge, iexp] = calibinfo[exp]["best_score"]

cc_pvalue: dict[str, float] = {}
cc_tvalue: dict[str, float] = {}
for iexp, exp in enumerate(EXPS):
    cc_group1 = np.array(best_cc[:, iexp].flatten())
    cc_group2 = np.concatenate(
        (best_cc[:, :iexp].flatten(), best_cc[:, (iexp + 1) :].flatten())
    )
    t_value, p_value = stats.ttest_ind(cc_group1, cc_group2, equal_var=False)
    cc_tvalue[exp] = float(t_value)  # type: ignore
    cc_pvalue[exp] = float(p_value)  # type: ignore
exp_outlier: set[str] = set()
for exp in EXPS:
    if cc_tvalue[exp] < 0 and cc_pvalue[exp] < 0.05:
        exp_outlier.add(exp)
print(f"Experiments {exp_outlier} have significantly lower correlation coefficients.")

# show the distribution of the optimal correlation coefficients
fig = plt.figure(figsize=(7 / 2.54, 5 / 2.54), dpi=FIGDPI, layout="constrained")
axs = fig.subplots(1, 1, sharex=True, sharey=True, squeeze=False)
ax = axs[0][0]

plt.violinplot(best_cc, showmeans=True)
plt.ylabel("Correlation Coefficient (-)")
plt.xticks(
    ticks=np.arange(1, len(EXPS) + 1),
    labels=[f"E{exp}" for exp in EXPS],
    rotation=45,
)
for iexp, exp in enumerate(EXPS):
    y = 0.5
    if cc_pvalue[exp] < 0.05 and cc_tvalue[exp] < 0:
        ax.text(
            iexp + 1,
            y,
            "*",
            ha="center",
            va="center",
            fontsize=8,
            color="red",
        )
plt.ylim(0.5, 0.9)
plt.xlim(0.5, len(EXPS) + 0.5)

fig.savefig(
    FIGROOT.joinpath("optimal_cc_outlier").with_suffix(FIGSUFFIX).as_posix(),
    bbox_inches="tight",
)


# %%
# fit the log-normal distribution to the optimal celerities
SELECTED_EXPS = list(EXPS)
for exp in exp_outlier:
    SELECTED_EXPS.remove(exp)
lncelerity_mean: dict[str, float] = {}
lncelerity_std: dict[str, float] = {}
for gauge in gaugelist:
    celerities = [calibration[gauge][exp]["best_celerity"] for exp in SELECTED_EXPS]
    lncelerity_mean[gauge] = float(np.nanmean(np.log(celerities)).item())
    lncelerity_std[gauge] = float(
        np.nanstd(np.log(celerities) - lncelerity_mean[gauge]).item()
    )

CELERITY_MEASURE_FILE = DATAROOT.joinpath(
    "qout_calibration", "celerity_measurement.csv"
)
with open(CELERITY_MEASURE_FILE, "wt") as f:
    f.write("gauge,mean_lncelerity,std_lncelerity\n")
    for igauge, gauge in enumerate(gaugelist):
        f.write(f"{gauge},{lncelerity_mean[gauge]:.6f},{lncelerity_std[gauge]:.6f}\n")

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
fig = plt.figure(figsize=(14 / 2.54, 14 / 2.54), dpi=FIGDPI, layout="constrained")
axs = fig.subplots(3, 2, sharex=True, sharey=True, squeeze=False)


for igauge, gauge in enumerate(gaugelist):
    ax = axs.flatten()[igauge]
    calibinfo = calibration[gauge]
    for iexp, exp in enumerate(EXPS):
        ax.plot(
            calibinfo[exp]["celerity"],
            calibinfo[exp]["score"],
            linewidth=1,
            label=f"E{exp}",
            linestyle=linestyles[exp],
            color=linecolors[exp],
        )
    for iexp, exp in enumerate(EXPS):
        ax.scatter(
            calibinfo[exp]["best_celerity"],
            calibinfo[exp]["best_score"],
            color="k",
            marker="+",
            linewidth=1,
            s=10,
            zorder=2,
        )
    ax.set_xscale("log")
    ax.plot(
        [np.exp(lncelerity_mean[gauge]), np.exp(lncelerity_mean[gauge])],
        [-0.5, 0.9],
        color="k",
    )
    ax.fill_betweenx(
        [-0.5, 0.9],
        [np.exp(lncelerity_mean[gauge] - lncelerity_std[gauge])],
        [np.exp(lncelerity_mean[gauge] + lncelerity_std[gauge])],
        color="gray",
        linewidth=0,
        alpha=0.5,
    )
    ax.text(
        np.exp(lncelerity_mean[gauge]),
        -0.4,
        f"{np.exp(lncelerity_mean[gauge]):.2f}",
        fontsize=8,
        ha="left",
        va="bottom",
    )
    ax.text(
        np.exp(lncelerity_mean[gauge] - lncelerity_std[gauge]),
        -0.5,
        f"{np.exp(lncelerity_mean[gauge] - lncelerity_std[gauge]):.2f}",
        fontsize=8,
        ha="right",
        va="bottom",
        color="gray",
    )
    ax.text(
        np.exp(lncelerity_mean[gauge] + lncelerity_std[gauge]),
        -0.5,
        f"{np.exp(lncelerity_mean[gauge] + lncelerity_std[gauge]):.2f}",
        fontsize=8,
        ha="left",
        va="bottom",
        color="gray",
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
    ax.set_ylim(-0.5, 0.9)
    ax.set_xlim(0.05, 6.2)
axs.flatten()[-3].legend(
    loc="lower left",
    frameon=False,
    fontsize=8,
    ncol=5,
    borderaxespad=0.1,
    borderpad=0.05,
    handlelength=0.8,
    labelspacing=0.1,
    handletextpad=0.1,
    columnspacing=0.2,
)
axs[-1][0].set_xlabel("Celerity (m s⁻¹)")
axs[-1][1].set_xlabel("Celerity (m s⁻¹)")
axs[1][0].set_ylabel("Correlation Coefficient (-)")

plt.savefig(
    FIGROOT.joinpath("celerity_estimation").with_suffix(FIGSUFFIX),
    dpi=FIGDPI,
    bbox_inches="tight",
)
