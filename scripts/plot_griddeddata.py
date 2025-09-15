# %%
import json
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mpldates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

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
DATETIME_START = datetime(2013, 6, 20)
DATETIME_STOP = datetime(2013, 10, 1) + timedelta(seconds=1)
EXPS = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13"]
SELECTED_EXPS = EXPS.copy()
SELECTED_EXPS.remove("10")
NPARAMETERS = 100


# %%
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
    return kge


# %%
mask = xr.open_dataset(DATAROOT.joinpath("wrf_mask.nc"))
wrf = xr.open_dataset(DATAROOT.joinpath("wrf_data.nc")).sel(
    time=slice(DATETIME_START, DATETIME_STOP)
)
prgpm = xr.open_dataset(DATAROOT.joinpath("gpm.nc")).sel(
    time=slice(DATETIME_START, DATETIME_STOP)
)
swe = xr.open_dataset(DATAROOT.joinpath("snow.nc")).sel(
    time=slice(DATETIME_START, DATETIME_STOP)
)


# %%
def plot_yj(ax):
    ax.contour(
        mask.lon,
        mask.lat,
        mask.basin,
        levels=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
        linewidths=0.5,
        colors="white",
    )
    ax.set_xticks([82, 86, 90, 94])
    ax.set_xticklabels(["82°E", "86°E", "90°E", "94°E"])
    ax.set_yticks([28, 29, 30, 31])
    ax.set_yticklabels(["28°N", "29°N", "30°N", "31°N"])
    ax.set_aspect("equal")


# %%
gaugemask: dict[str, np.ndarray] = {
    "90601000": mask["gauge"].values == 1,
    "90602000": np.logical_or(mask["gauge"].values == 2, mask["gauge"].values == 1),
    "90603000": np.logical_or(
        mask["gauge"].values == 5,
        np.logical_or(
            np.logical_or(mask["gauge"].values == 3, mask["gauge"].values == 2),
            mask["gauge"].values == 1,
        ),
    ),
    "90604500": mask["gauge"].values > 0,
    "90802500": mask["gauge"].values == 5,
    "90901200": mask["gauge"].values == 6,
}

# %%
for igauge, (cgauge, cmask) in enumerate(gaugemask.items()):
    plt.subplot(3, 2, igauge + 1)
    plt.contourf(cmask)
    plt.title(cgauge)

# %%
rnwrf_clim = (
    wrf["mrro"]
    .mean("time")
    .sel(ensemble=[1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13])
    .where(mask["gauge"] > 0)
)

# %%
fig = plt.figure(figsize=(14 / 2.54, 6 / 2.54), dpi=FIGDPI, layout="constrained")

axs = fig.subplots(3, 2, sharex=True, sharey=True, squeeze=False)

levels = np.linspace(0, 1.5, 16) * 100.0

ax = axs[0][0]
ax.contourf(
    rnwrf_clim.lon,
    rnwrf_clim.lat,
    100 * rnwrf_clim.std("ensemble") / rnwrf_clim.mean("ensemble"),
    levels=levels,
    extend="max",
)
plot_yj(ax)
ax.text(0.01, 0.02, "a)", transform=ax.transAxes, va="bottom", ha="left")
ax.text(
    0.09,
    0.02,
    "All",
    fontsize=8,
    transform=ax.transAxes,
    va="bottom",
    ha="left",
)


ax = axs[1][0]
ax.contourf(
    rnwrf_clim.lon,
    rnwrf_clim.lat,
    100
    * rnwrf_clim.sel(ensemble=[1, 2, 3, 4, 5]).std("ensemble")
    / rnwrf_clim.mean("ensemble"),
    levels=levels,
    extend="max",
)
plot_yj(ax)
ax.text(0.01, 0.02, "b)", transform=ax.transAxes, va="bottom", ha="left")
ax.text(
    0.09,
    0.02,
    "Radiation",
    fontsize=8,
    transform=ax.transAxes,
    va="bottom",
    ha="left",
)


ax = axs[1][1]
ax.contourf(
    rnwrf_clim.lon,
    rnwrf_clim.lat,
    100
    * rnwrf_clim.sel(ensemble=[3, 6, 7]).std("ensemble")
    / rnwrf_clim.mean("ensemble"),
    levels=levels,
    extend="max",
)
plot_yj(ax)
ax.text(0.01, 0.02, "c)", transform=ax.transAxes, va="bottom", ha="left")
ax.text(
    0.09,
    0.02,
    "Microphysics",
    fontsize=8,
    transform=ax.transAxes,
    va="bottom",
    ha="left",
)


ax = axs[2][0]
ax.contourf(
    rnwrf_clim.lon,
    rnwrf_clim.lat,
    100
    * rnwrf_clim.sel(ensemble=[3, 8, 9, 11, 12]).std("ensemble")
    / rnwrf_clim.mean("ensemble"),
    levels=levels,
    extend="max",
)
plot_yj(ax)
ax.text(0.01, 0.02, "d)", transform=ax.transAxes, va="bottom", ha="left")
ax.text(
    0.09,
    0.02,
    "Boundary layer",
    fontsize=8,
    transform=ax.transAxes,
    va="bottom",
    ha="left",
)


ax = axs[2][1]
cf = ax.contourf(
    rnwrf_clim.lon,
    rnwrf_clim.lat,
    100
    * rnwrf_clim.sel(ensemble=[3, 8, 9, 11, 12]).std("ensemble")
    / rnwrf_clim.mean("ensemble"),
    levels=levels,
    extend="max",
)
plot_yj(ax)
ax.text(0.01, 0.02, "e)", transform=ax.transAxes, va="bottom", ha="left")
ax.text(
    0.09,
    0.02,
    "Orographic drag",
    fontsize=8,
    transform=ax.transAxes,
    va="bottom",
    ha="left",
)


plt.colorbar(
    cf,
    ax=axs,
    pad=0.02,
    aspect=20,
    shrink=0.8,
    label="Coefficient of Variation (%)",
)

axs[0][1].set_visible(False)
fig.savefig(
    FIGROOT.joinpath("runoff_coeffcient_of_variation")
    .with_suffix(FIGSUFFIX)
    .as_posix(),
    dpi=FIGDPI,
)

# %%
prwrf_clim = (
    wrf["pr"]
    .mean("time")
    .sel(ensemble=[1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13])
    .where(mask["gauge"] > 0)
)

# %%
fig = plt.figure(figsize=(14 / 2.54, 6 / 2.54), dpi=FIGDPI, layout="constrained")

axs = fig.subplots(3, 2, sharex=True, sharey=True, squeeze=False)

levels = np.linspace(0, 1.5, 16) * 100.0

ax = axs[0][0]
ax.contourf(
    prwrf_clim.lon,
    prwrf_clim.lat,
    100 * prwrf_clim.std("ensemble") / prwrf_clim.mean("ensemble"),
    levels=levels,
    extend="max",
)
plot_yj(ax)
ax.text(0.01, 0.02, "a)", transform=ax.transAxes, va="bottom", ha="left")
ax.text(
    0.09,
    0.02,
    "All",
    fontsize=8,
    transform=ax.transAxes,
    va="bottom",
    ha="left",
)


ax = axs[1][0]
ax.contourf(
    prwrf_clim.lon,
    prwrf_clim.lat,
    100
    * prwrf_clim.sel(ensemble=[1, 2, 3, 4, 5]).std("ensemble")
    / prwrf_clim.mean("ensemble"),
    levels=levels,
    extend="max",
)
plot_yj(ax)
ax.text(0.01, 0.02, "b)", transform=ax.transAxes, va="bottom", ha="left")
ax.text(
    0.09,
    0.02,
    "Radiation",
    fontsize=8,
    transform=ax.transAxes,
    va="bottom",
    ha="left",
)


ax = axs[1][1]
ax.contourf(
    prwrf_clim.lon,
    prwrf_clim.lat,
    100
    * prwrf_clim.sel(ensemble=[3, 6, 7]).std("ensemble")
    / prwrf_clim.mean("ensemble"),
    levels=levels,
    extend="max",
)
plot_yj(ax)
ax.text(0.01, 0.02, "c)", transform=ax.transAxes, va="bottom", ha="left")
ax.text(
    0.09,
    0.02,
    "Microphysics",
    fontsize=8,
    transform=ax.transAxes,
    va="bottom",
    ha="left",
)


ax = axs[2][0]
ax.contourf(
    prwrf_clim.lon,
    prwrf_clim.lat,
    100
    * prwrf_clim.sel(ensemble=[3, 8, 9, 11, 12]).std("ensemble")
    / prwrf_clim.mean("ensemble"),
    levels=levels,
    extend="max",
)
plot_yj(ax)
ax.text(0.01, 0.02, "d)", transform=ax.transAxes, va="bottom", ha="left")
ax.text(
    0.09,
    0.02,
    "Boundary layer",
    fontsize=8,
    transform=ax.transAxes,
    va="bottom",
    ha="left",
)


ax = axs[2][1]
cf = ax.contourf(
    prwrf_clim.lon,
    prwrf_clim.lat,
    100
    * prwrf_clim.sel(ensemble=[3, 8, 9, 11, 12]).std("ensemble")
    / prwrf_clim.mean("ensemble"),
    levels=levels,
    extend="max",
)
plot_yj(ax)
ax.text(0.01, 0.02, "e)", transform=ax.transAxes, va="bottom", ha="left")
ax.text(
    0.09,
    0.02,
    "Orographic drag",
    fontsize=8,
    transform=ax.transAxes,
    va="bottom",
    ha="left",
)


plt.colorbar(
    cf,
    ax=axs,
    pad=0.02,
    aspect=20,
    shrink=0.8,
    label="Coefficient of Variation (%)",
)

axs[0][1].set_visible(False)
fig.savefig(
    FIGROOT.joinpath("precipitation_coefficient_of_variation")
    .with_suffix(FIGSUFFIX)
    .as_posix(),
    dpi=FIGDPI,
)

# %%
etwrf_clim = (
    wrf["et"]
    .mean("time")
    .sel(ensemble=[1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13])
    .where(mask["gauge"] > 0)
)

# %%
fig = plt.figure(figsize=(14 / 2.54, 6 / 2.54), dpi=FIGDPI, layout="constrained")

axs = fig.subplots(3, 2, sharex=True, sharey=True, squeeze=False)

levels = np.linspace(0, 1.0, 10) * 100.0

ax = axs[0][0]
ax.contourf(
    etwrf_clim.lon,
    etwrf_clim.lat,
    100 * etwrf_clim.std("ensemble") / etwrf_clim.mean("ensemble"),
    levels=levels,
    extend="max",
)
plot_yj(ax)
ax.text(0.01, 0.02, "a)", transform=ax.transAxes, va="bottom", ha="left")
ax.text(
    0.09,
    0.02,
    "All",
    fontsize=8,
    transform=ax.transAxes,
    va="bottom",
    ha="left",
)


ax = axs[1][0]
ax.contourf(
    etwrf_clim.lon,
    etwrf_clim.lat,
    100
    * etwrf_clim.sel(ensemble=[1, 2, 3, 4, 5]).std("ensemble")
    / etwrf_clim.mean("ensemble"),
    levels=levels,
    extend="max",
)
plot_yj(ax)
ax.text(0.01, 0.02, "b)", transform=ax.transAxes, va="bottom", ha="left")
ax.text(
    0.09,
    0.02,
    "Radiation",
    fontsize=8,
    transform=ax.transAxes,
    va="bottom",
    ha="left",
)


ax = axs[1][1]
ax.contourf(
    etwrf_clim.lon,
    etwrf_clim.lat,
    100
    * etwrf_clim.sel(ensemble=[3, 6, 7]).std("ensemble")
    / etwrf_clim.mean("ensemble"),
    levels=levels,
    extend="max",
)
plot_yj(ax)
ax.text(0.01, 0.02, "c)", transform=ax.transAxes, va="bottom", ha="left")
ax.text(
    0.09,
    0.02,
    "Microphysics",
    fontsize=8,
    transform=ax.transAxes,
    va="bottom",
    ha="left",
)


ax = axs[2][0]
ax.contourf(
    etwrf_clim.lon,
    etwrf_clim.lat,
    100
    * etwrf_clim.sel(ensemble=[3, 8, 9, 11, 12]).std("ensemble")
    / etwrf_clim.mean("ensemble"),
    levels=levels,
    extend="max",
)
plot_yj(ax)
ax.text(0.01, 0.02, "d)", transform=ax.transAxes, va="bottom", ha="left")
ax.text(
    0.09,
    0.02,
    "Boundary layer",
    fontsize=8,
    transform=ax.transAxes,
    va="bottom",
    ha="left",
)


ax = axs[2][1]
cf = ax.contourf(
    etwrf_clim.lon,
    etwrf_clim.lat,
    100
    * etwrf_clim.sel(ensemble=[3, 8, 9, 11, 12]).std("ensemble")
    / etwrf_clim.mean("ensemble"),
    levels=levels,
    extend="max",
)
plot_yj(ax)
ax.text(0.01, 0.02, "e)", transform=ax.transAxes, va="bottom", ha="left")
ax.text(
    0.09,
    0.02,
    "Orographic drag",
    fontsize=8,
    transform=ax.transAxes,
    va="bottom",
    ha="left",
)


plt.colorbar(
    cf,
    ax=axs,
    pad=0.02,
    aspect=20,
    shrink=0.8,
    label="Coefficient of Variation (%)",
)

axs[0][1].set_visible(False)
fig.savefig(
    FIGROOT.joinpath("evapotranspiration_coefficient_of_variation")
    .with_suffix(FIGSUFFIX)
    .as_posix(),
    dpi=FIGDPI,
)

# %%
pretdwrf_clim = (
    (wrf["pr"] - wrf["et"])
    .mean("time")
    .sel(ensemble=[1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13])
    .where(mask["gauge"] > 0)
)

# %%
fig = plt.figure(figsize=(14 / 2.54, 6 / 2.54), dpi=FIGDPI, layout="constrained")

axs = fig.subplots(3, 2, sharex=True, sharey=True, squeeze=False)

levels = np.linspace(0, 1.5, 16) * 100.0

ax = axs[0][0]
ax.contourf(
    pretdwrf_clim.lon,
    pretdwrf_clim.lat,
    100 * pretdwrf_clim.std("ensemble") / pretdwrf_clim.mean("ensemble"),
    levels=levels,
    extend="max",
)
plot_yj(ax)
ax.text(0.01, 0.02, "a)", transform=ax.transAxes, va="bottom", ha="left")
ax.text(
    0.09,
    0.02,
    "All",
    fontsize=8,
    transform=ax.transAxes,
    va="bottom",
    ha="left",
)


ax = axs[1][0]
ax.contourf(
    pretdwrf_clim.lon,
    pretdwrf_clim.lat,
    100
    * pretdwrf_clim.sel(ensemble=[1, 2, 3, 4, 5]).std("ensemble")
    / pretdwrf_clim.mean("ensemble"),
    levels=levels,
    extend="max",
)
plot_yj(ax)
ax.text(0.01, 0.02, "b)", transform=ax.transAxes, va="bottom", ha="left")
ax.text(
    0.09,
    0.02,
    "Radiation",
    fontsize=8,
    transform=ax.transAxes,
    va="bottom",
    ha="left",
)


ax = axs[1][1]
ax.contourf(
    pretdwrf_clim.lon,
    pretdwrf_clim.lat,
    100
    * pretdwrf_clim.sel(ensemble=[3, 6, 7]).std("ensemble")
    / pretdwrf_clim.mean("ensemble"),
    levels=levels,
    extend="max",
)
plot_yj(ax)
ax.text(0.01, 0.02, "c)", transform=ax.transAxes, va="bottom", ha="left")
ax.text(
    0.09,
    0.02,
    "Microphysics",
    fontsize=8,
    transform=ax.transAxes,
    va="bottom",
    ha="left",
)


ax = axs[2][0]
ax.contourf(
    pretdwrf_clim.lon,
    pretdwrf_clim.lat,
    100
    * pretdwrf_clim.sel(ensemble=[3, 8, 9, 11, 12]).std("ensemble")
    / pretdwrf_clim.mean("ensemble"),
    levels=levels,
    extend="max",
)
plot_yj(ax)
ax.text(0.01, 0.02, "d)", transform=ax.transAxes, va="bottom", ha="left")
ax.text(
    0.09,
    0.02,
    "Boundary layer",
    fontsize=8,
    transform=ax.transAxes,
    va="bottom",
    ha="left",
)


ax = axs[2][1]
cf = ax.contourf(
    pretdwrf_clim.lon,
    pretdwrf_clim.lat,
    100
    * pretdwrf_clim.sel(ensemble=[3, 8, 9, 11, 12]).std("ensemble")
    / pretdwrf_clim.mean("ensemble"),
    levels=levels,
    extend="max",
)
plot_yj(ax)
ax.text(0.01, 0.02, "e)", transform=ax.transAxes, va="bottom", ha="left")
ax.text(
    0.09,
    0.02,
    "Orographic drag",
    fontsize=8,
    transform=ax.transAxes,
    va="bottom",
    ha="left",
)


plt.colorbar(
    cf,
    ax=axs,
    pad=0.02,
    aspect=20,
    shrink=0.8,
    label="Coefficient of Variation (%)",
)

axs[0][1].set_visible(False)
fig.savefig(
    FIGROOT.joinpath("pretdiff_coefficient_of_variation")
    .with_suffix(FIGSUFFIX)
    .as_posix(),
    dpi=FIGDPI,
)

# %%
smwrf_clim = (
    wrf["sm"][
        :,
        :,
        0,
        :,
        :,
    ]
    .mean("time")
    .sel(ensemble=[1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13])
    .where(mask["gauge"] > 0)
)

# %%
fig = plt.figure(figsize=(14 / 2.54, 6 / 2.54), dpi=FIGDPI, layout="constrained")

axs = fig.subplots(3, 2, sharex=True, sharey=True, squeeze=False)

levels = np.linspace(0, 0.4, 21) * 100.0

ax = axs[0][0]
ax.contourf(
    smwrf_clim.lon,
    smwrf_clim.lat,
    100 * smwrf_clim.std("ensemble") / smwrf_clim.mean("ensemble"),
    levels=levels,
    extend="max",
)
plot_yj(ax)
ax.text(0.01, 0.02, "a)", transform=ax.transAxes, va="bottom", ha="left")
ax.text(
    0.09,
    0.02,
    "All",
    fontsize=8,
    transform=ax.transAxes,
    va="bottom",
    ha="left",
)


ax = axs[1][0]
ax.contourf(
    smwrf_clim.lon,
    smwrf_clim.lat,
    100
    * smwrf_clim.sel(ensemble=[1, 2, 3, 4, 5]).std("ensemble")
    / smwrf_clim.mean("ensemble"),
    levels=levels,
    extend="max",
)
plot_yj(ax)
ax.text(0.01, 0.02, "b)", transform=ax.transAxes, va="bottom", ha="left")
ax.text(
    0.09,
    0.02,
    "Radiation",
    fontsize=8,
    transform=ax.transAxes,
    va="bottom",
    ha="left",
)


ax = axs[1][1]
ax.contourf(
    smwrf_clim.lon,
    smwrf_clim.lat,
    100
    * smwrf_clim.sel(ensemble=[3, 6, 7]).std("ensemble")
    / smwrf_clim.mean("ensemble"),
    levels=levels,
    extend="max",
)
plot_yj(ax)
ax.text(0.01, 0.02, "c)", transform=ax.transAxes, va="bottom", ha="left")
ax.text(
    0.09,
    0.02,
    "Microphysics",
    fontsize=8,
    transform=ax.transAxes,
    va="bottom",
    ha="left",
)


ax = axs[2][0]
ax.contourf(
    smwrf_clim.lon,
    smwrf_clim.lat,
    100
    * smwrf_clim.sel(ensemble=[3, 8, 9, 11, 12]).std("ensemble")
    / smwrf_clim.mean("ensemble"),
    levels=levels,
    extend="max",
)
plot_yj(ax)
ax.text(0.01, 0.02, "d)", transform=ax.transAxes, va="bottom", ha="left")
ax.text(
    0.09,
    0.02,
    "Boundary layer",
    fontsize=8,
    transform=ax.transAxes,
    va="bottom",
    ha="left",
)


ax = axs[2][1]
cf = ax.contourf(
    smwrf_clim.lon,
    smwrf_clim.lat,
    100
    * smwrf_clim.sel(ensemble=[3, 8, 9, 11, 12]).std("ensemble")
    / smwrf_clim.mean("ensemble"),
    levels=levels,
    extend="max",
)
plot_yj(ax)
ax.text(0.01, 0.02, "e)", transform=ax.transAxes, va="bottom", ha="left")
ax.text(
    0.09,
    0.02,
    "Orographic drag",
    fontsize=8,
    transform=ax.transAxes,
    va="bottom",
    ha="left",
)


plt.colorbar(
    cf,
    ax=axs,
    pad=0.02,
    aspect=20,
    shrink=0.8,
    label="Coefficient of Variation (%)",
)

axs[0][1].set_visible(False)
fig.savefig(
    FIGROOT.joinpath("soilmoisture_coefficient_of_variation")
    .with_suffix(FIGSUFFIX)
    .as_posix(),
    dpi=FIGDPI,
)

# %%
rnaccum = wrf["mrro"].where(mask["basin"] > 0).mean(dim=["lat", "lon"]).cumsum("time")
swets = wrf["swe"].where(mask["basin"] > 0).mean(dim=["lat", "lon"])

# %%
fig = plt.figure(figsize=(8 / 2.54, 6 / 2.54), dpi=FIGDPI, layout="constrained")
axs = fig.subplots(1, 1, sharex=True, squeeze=False)

ax = axs[0][0]
for iexp in range(rnaccum.ensemble.size):
    ll = ax.plot(
        rnaccum.time,
        rnaccum.isel(ensemble=iexp) * 3600.0,
        color=mpl.color_sequences["tab20"][iexp],
        linewidth=1,
        label=f"E{iexp + 1}",
    )
    ax.plot(
        swets.time,
        swets.isel(ensemble=iexp),
        color=mpl.color_sequences["tab20"][iexp],
        linewidth=1,
        linestyle="--",
    )
ax.set_ylabel("Cumulative runoff / SWE (mm)")
ax.set_xlim(DATETIME_START, DATETIME_STOP)
ax.set_ylim(0, 450)
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
ax.xaxis.set_tick_params(rotation=30)
plt.legend(
    frameon=False,
    ncol=2,
    fontsize=8,
    handlelength=0.8,
    labelspacing=0.1,
    handletextpad=0.1,
    columnspacing=0.5,
)

fig.savefig(
    FIGROOT.joinpath("rnswe_time_series").with_suffix(FIGSUFFIX).as_posix(),
    dpi=FIGDPI,
)


# %%
pr_scc: dict[str, dict[str, float]] = {g: {} for g in gaugemask.keys()}

prwrf_clim = wrf["pr"].mean("time")
prgpm_clim = prgpm["pr"].mean("time")
for iens, ens in enumerate(prwrf_clim["ensemble"]):
    ensname = f"{ens:02d}"
    x = prgpm_clim.values
    y = prwrf_clim.values[iens, :, :]
    for gauge, mm in gaugemask.items():
        cc = np.corrcoef(x[mm], y[mm])[0, 1]
        pr_scc[gauge][ensname] = float(cc)

# %%
with open(DATAROOT.joinpath("skills/pr_spatial_correlation.json"), "wt") as f:
    json.dump(pr_scc, f, indent=4)

# %%
pr_tcc: dict[str, dict[str, float]] = {g: {} for g in gaugemask.keys()}
pr_kge: dict[str, dict[str, float]] = {g: {} for g in gaugemask.keys()}

for iens, ens in enumerate(prwrf_clim["ensemble"]):
    ensname = f"{ens:02d}"
    for gauge, mm in gaugemask.items():
        x = prgpm["pr"].where(mm).mean(dim=["lat", "lon"])
        y = wrf["pr"].sel(ensemble=ens).where(mm).mean(dim=["lat", "lon"])
        cc = np.corrcoef(x, y)[0, 1]
        pr_tcc[gauge][ensname] = float(cc)
        kge = calculate_skill(x, y)
        pr_kge[gauge][ensname] = float(kge)

# %%
with open(DATAROOT.joinpath("skills/pr_temporal_correlation.json"), "wt") as f:
    json.dump(pr_tcc, f, indent=4)
with open(DATAROOT.joinpath("skills/pr_kge.json"), "wt") as f:
    json.dump(pr_kge, f, indent=4)

# %%
wrfdaily = wrf.resample(time="1D").mean(dim="time")

# %%
swe_scc: dict[str, dict[str, float]] = {g: {} for g in gaugemask.keys()}

sweobs_clim = swe["snow"].mean("time")
swewrf_clim = wrfdaily["swe"].mean("time")
for iens, ens in enumerate(swewrf_clim["ensemble"]):
    ensname = f"{ens:02d}"
    x = sweobs_clim.values
    y = swewrf_clim.values[iens, :, :]
    for gauge, mm in gaugemask.items():
        cc = np.corrcoef(x[mm], y[mm])[0, 1]
        swe_scc[gauge][ensname] = float(cc)

# %%
swe_tcc: dict[str, dict[str, float]] = {g: {} for g in gaugemask.keys()}
swe_kge: dict[str, dict[str, float]] = {g: {} for g in gaugemask.keys()}

for iens, ens in enumerate(swewrf_clim["ensemble"]):
    ensname = f"{ens:02d}"
    for gauge, mm in gaugemask.items():
        x = swe["snow"].where(mm).mean(dim=["latitude", "longitude"])
        y = wrfdaily["swe"].sel(ensemble=ens).where(mm).mean(dim=["lat", "lon"])
        cc = np.corrcoef(x, y)[0, 1]
        swe_tcc[gauge][ensname] = float(cc)
        kge = calculate_skill(x, y)
        swe_kge[gauge][ensname] = float(kge)

# %%
with open(DATAROOT.joinpath("skills/swe_spatial_correlation.json"), "w") as f:
    json.dump(swe_scc, f, indent=4)
with open(DATAROOT.joinpath("skills/swe_temporal_correlation.json"), "w") as f:
    json.dump(swe_tcc, f, indent=4)
with open(DATAROOT.joinpath("skills/swe_kge.json"), "w") as f:
    json.dump(swe_kge, f, indent=4)

# %%
