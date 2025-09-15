# %%
import json
import string
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypedDict
from uuid import UUID

import matplotlib.dates as mpldates
import matplotlib.pyplot as plt
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
# read data
with open(DATAROOT.joinpath("skills/streamflow_rho.json"), "r") as f:
    qsim_rho = json.load(f)
with open(DATAROOT.joinpath("skills/streamflow_kge.json"), "r") as f:
    qsim_kge = json.load(f)
with open(DATAROOT.joinpath("skills/passthrough_kge.json"), "r") as f:
    qpass_kge = json.load(f)
with open(DATAROOT.joinpath("skills/passthrough_rho.json"), "r") as f:
    qpass_rho = json.load(f)
with open(DATAROOT.joinpath("skills/swe_spatial_correlation.json"), "r") as f:
    swe_srho = json.load(f)
with open(DATAROOT.joinpath("skills/swe_temporal_correlation.json"), "r") as f:
    swe_rho = json.load(f)
with open(DATAROOT.joinpath("skills/pr_kge.json"), "r") as f:
    pr_kge = json.load(f)
with open(DATAROOT.joinpath("skills/pr_spatial_correlation.json"), "r") as f:
    pr_srho = json.load(f)
with open(DATAROOT.joinpath("skills/pr_temporal_correlation.json"), "r") as f:
    pr_rho = json.load(f)

# %%
qsim_rho_median = {
    gauge: {e: float(np.median(v)) for e, v in exps.items()}
    for gauge, exps in qsim_rho.items()
}
qsim_kge_median = {
    gauge: {e: float(np.median(v)) for e, v in exps.items()}
    for gauge, exps in qsim_kge.items()
}
qsim_rho_best = {
    gauge: {e: float(np.max(v)) for e, v in exps.items()}
    for gauge, exps in qsim_rho.items()
}
qsim_kge_best = {
    gauge: {e: float(np.max(v)) for e, v in exps.items()}
    for gauge, exps in qsim_kge.items()
}

# %%
rank_kge_info = {
    gauge: {
        "streamflow_best": {
            "statistic": float("nan"),
            "pvalue": float("nan"),
        },
        "runoff": {
            "statistic": float("nan"),
            "pvalue": float("nan"),
        },
        "precipitation": {
            "statistic": float("nan"),
            "pvalue": float("nan"),
        },
    }
    for gauge in gaugelist
}

fig = plt.figure(figsize=(10 / 2.54, 12 / 2.54), dpi=FIGDPI, layout="constrained")
axs = fig.subplots(3, 2, sharex=True, sharey=True, squeeze=False)


for igauge, gauge in enumerate(gaugelist):
    ax = axs.flatten()[igauge]
    ax.plot(
        [0.5, len(SELECTED_EXPS) + 0.5],
        [0.5, len(SELECTED_EXPS) + 0.5],
        "k--",
        lw=0.5,
    )

    x = [qsim_kge_median[gauge][e] for e in SELECTED_EXPS]
    y = [qsim_kge_best[gauge][e] for e in SELECTED_EXPS]
    res = stats.spearmanr(x, y)
    rank_kge_info[gauge]["streamflow_best"]["statistic"] = res.statistic
    rank_kge_info[gauge]["streamflow_best"]["pvalue"] = res.pvalue
    xrank = len(SELECTED_EXPS) + 1 - stats.rankdata(x, method="ordinal")
    yrank = len(SELECTED_EXPS) + 1 - stats.rankdata(y, method="ordinal")
    for ii, (xx, yy, ee) in enumerate(zip(xrank, yrank, SELECTED_EXPS)):
        print(gauge, ee, xx, x[ii])
        ax.plot(
            xx,
            yy,
            "o",
            markersize=3,
            color=plt.color_sequences["tab20"][ii],
            label=f"E{ee}",
        )

    y = [qpass_kge[gauge][e] for e in SELECTED_EXPS]
    res = stats.spearmanr(x, y)
    rank_kge_info[gauge]["runoff"]["statistic"] = res.statistic
    rank_kge_info[gauge]["runoff"]["pvalue"] = res.pvalue
    xrank = len(SELECTED_EXPS) + 1 - stats.rankdata(x, method="ordinal")
    yrank = len(SELECTED_EXPS) + 1 - stats.rankdata(y, method="ordinal")
    for ii, (xx, yy, ee) in enumerate(zip(xrank, yrank, SELECTED_EXPS)):
        ax.plot(xx, yy, "P", color=plt.color_sequences["tab20"][ii])

    y = [pr_kge[gauge][e] for e in SELECTED_EXPS]
    res = stats.spearmanr(x, y)
    rank_kge_info[gauge]["precipitation"]["statistic"] = res.statistic
    rank_kge_info[gauge]["precipitation"]["pvalue"] = res.pvalue

    xrank = len(SELECTED_EXPS) + 1 - stats.rankdata(x, method="ordinal")
    yrank = len(SELECTED_EXPS) + 1 - stats.rankdata(y, method="ordinal")
    for ii, (xx, yy, ee) in enumerate(zip(xrank, yrank, SELECTED_EXPS)):
        ax.plot(xx, yy, "^", color=plt.color_sequences["tab20"][ii])

    ax.set_title(
        f"{string.ascii_lowercase[igauge]}) {gaugeshortname[gauge]}",
        fontsize=12,
    )

    ax.set_xlim([0.5, len(SELECTED_EXPS) + 0.5])
    ax.set_ylim([0.5, len(SELECTED_EXPS) + 0.5])
    ax.set_xticks(range(1, len(SELECTED_EXPS) + 1))
    ax.set_yticks(range(1, len(SELECTED_EXPS) + 1))
axs.flatten()[0].legend(
    loc="upper left",
    frameon=False,
    fontsize=8,
    ncol=2,
    borderaxespad=0.1,
    borderpad=0.05,
    handlelength=0.8,
    labelspacing=0.02,
    handletextpad=0.1,
    columnspacing=0.2,
)
axs[-1][0].set_xlabel("Rank by median Q KGE")
axs[-1][1].set_xlabel("Rank by median Q KGE")
axs[1][0].set_ylabel("Rank")

plt.savefig(
    FIGROOT.joinpath("rank_kge_relationship").with_suffix(FIGSUFFIX),
    dpi=FIGDPI,
    bbox_inches="tight",
)
print(json.dumps(rank_kge_info, indent=4, ensure_ascii=False))

# %%

rank_rho_info = {
    gauge: {
        "streamflow_best": {
            "statistic": float("nan"),
            "pvalue": float("nan"),
        },
        "runoff": {
            "statistic": float("nan"),
            "pvalue": float("nan"),
        },
        "precipitation": {
            "statistic": float("nan"),
            "pvalue": float("nan"),
        },
        "precipitations_spatial": {
            "statistic": float("nan"),
            "pvalue": float("nan"),
        },
        "swe": {
            "statistic": float("nan"),
            "pvalue": float("nan"),
        },
        "swe_spatial": {
            "statistic": float("nan"),
            "pvalue": float("nan"),
        },
    }
    for gauge in gaugelist
}

fig = plt.figure(figsize=(10 / 2.54, 12 / 2.54), dpi=FIGDPI, layout="constrained")
axs = fig.subplots(3, 2, sharex=True, sharey=True, squeeze=False)


for igauge, gauge in enumerate(gaugelist):
    ax = axs.flatten()[igauge]
    ax.plot(
        [0.5, len(SELECTED_EXPS) + 0.5],
        [0.5, len(SELECTED_EXPS) + 0.5],
        "k--",
        lw=0.5,
    )

    x = [qsim_rho_median[gauge][e] for e in SELECTED_EXPS]
    y = [qsim_rho_best[gauge][e] for e in SELECTED_EXPS]
    res = stats.spearmanr(x, y)
    rank_rho_info[gauge]["streamflow_best"]["statistic"] = res.statistic
    rank_rho_info[gauge]["streamflow_best"]["pvalue"] = res.pvalue
    xrank = len(SELECTED_EXPS) + 1 - stats.rankdata(x, method="ordinal")
    yrank = len(SELECTED_EXPS) + 1 - stats.rankdata(y, method="ordinal")
    for ii, (xx, yy, ee) in enumerate(zip(xrank, yrank, SELECTED_EXPS)):
        ax.plot(
            xx,
            yy,
            "o",
            color=plt.color_sequences["tab20"][ii],
            label=f"E{ee}",
            markersize=3,
        )

    y = [qpass_rho[gauge][e] for e in SELECTED_EXPS]
    res = stats.spearmanr(x, y)
    rank_rho_info[gauge]["runoff"]["statistic"] = res.statistic
    rank_rho_info[gauge]["runoff"]["pvalue"] = res.pvalue
    xrank = len(SELECTED_EXPS) + 1 - stats.rankdata(x, method="ordinal")
    yrank = len(SELECTED_EXPS) + 1 - stats.rankdata(y, method="ordinal")
    for ii, (xx, yy, ee) in enumerate(zip(xrank, yrank, SELECTED_EXPS)):
        ax.plot(xx, yy, "P", color=plt.color_sequences["tab20"][ii])

    y = [pr_rho[gauge][e] for e in SELECTED_EXPS]
    res = stats.spearmanr(x, y)
    rank_rho_info[gauge]["precipitation"]["statistic"] = res.statistic
    rank_rho_info[gauge]["precipitation"]["pvalue"] = res.pvalue
    xrank = len(SELECTED_EXPS) + 1 - stats.rankdata(x, method="ordinal")
    yrank = len(SELECTED_EXPS) + 1 - stats.rankdata(y, method="ordinal")
    for ii, (xx, yy, ee) in enumerate(zip(xrank, yrank, SELECTED_EXPS)):
        ax.plot(xx, yy, "s", color=plt.color_sequences["tab20"][ii])

    y = [pr_srho[gauge][e] for e in SELECTED_EXPS]
    res = stats.spearmanr(x, y)
    rank_rho_info[gauge]["precipitations_spatial"]["statistic"] = res.statistic
    rank_rho_info[gauge]["precipitations_spatial"]["pvalue"] = res.pvalue
    xrank = len(SELECTED_EXPS) + 1 - stats.rankdata(x, method="ordinal")
    yrank = len(SELECTED_EXPS) + 1 - stats.rankdata(y, method="ordinal")
    for ii, (xx, yy, ee) in enumerate(zip(xrank, yrank, SELECTED_EXPS)):
        ax.plot(xx, yy, "^", color=plt.color_sequences["tab20"][ii])

    # y = [swe_rho[gauge][e] for e in SELECTED_EXPS]
    # res = stats.spearmanr(x, y)
    # rank_rho_info[gauge]["swe"]["statistic"] = res.statistic
    # rank_rho_info[gauge]["swe"]["pvalue"] = res.pvalue
    # ax.plot(
    #     stats.rankdata(x, method="ordinal"),
    #     stats.rankdata(y, method="ordinal"),
    #     "v",
    #     label="swe",
    #     markersize=3,
    #     color="k",
    # )

    ax.set_title(
        f"{string.ascii_lowercase[igauge]}) {gaugeshortname[gauge]}",
        fontsize=12,
    )
    ax.set_xlim([0.5, len(SELECTED_EXPS) + 0.5])
    ax.set_ylim([0.5, len(SELECTED_EXPS) + 0.5])
    ax.set_xticks(range(1, len(SELECTED_EXPS) + 1))
    ax.set_yticks(range(1, len(SELECTED_EXPS) + 1))
# axs.flatten()[0].legend(
#     loc="upper left",
#     frameon=False,
#     fontsize=8,
#     ncol=2,
#     borderaxespad=0.1,
#     borderpad=0.05,
#     handlelength=0.8,
#     labelspacing=0.02,
#     handletextpad=0.1,
#     columnspacing=0.2,
# )
axs[-1][0].set_xlabel("Rank by median streamflow ϱ")
axs[-1][1].set_xlabel("Rank by median streamflow ϱ")
axs[1][0].set_ylabel("Rank")

plt.savefig(
    FIGROOT.joinpath("rank_rho_relationship").with_suffix(FIGSUFFIX),
    dpi=FIGDPI,
    bbox_inches="tight",
)

print(json.dumps(rank_rho_info, indent=4, ensure_ascii=False))

# %%
