import concurrent.futures
from datetime import datetime, timedelta
from pathlib import Path

import h5netcdf
import numpy as np
import polars as pl
from numpy.typing import NDArray


def convert_filepairs(file_pairs: tuple[str, pl.DataFrame, list[datetime], NDArray, Path]):
    exp, rivnet, tm, srcdata, desfile = file_pairs
    return convert(exp, rivnet, tm, srcdata, desfile)


def convert(exp: str, rivnet: pl.DataFrame, tm: list[datetime], srcdata: NDArray, desfile: Path):
    RHO_WATER = 1000.0
    qlat = [pl.Series("tm", tm, dtype=pl.Datetime("ms"))]
    for i in range(rivnet.height):
        latidx = rivnet[i, "latidx"]
        lonidx = rivnet[i, "lonidx"]
        area = rivnet[i, "area"]
        # check for nan
        for ilatidx, ilonidx in zip(latidx, lonidx):
            if any(np.isnan(srcdata[:, ilatidx, ilonidx])):
                print("nan", ilatidx, ilonidx)
        qlat.append(
            pl.Series(
                str(rivnet[i, "uuid"]),
                np.sum((srcdata[:, latidx, lonidx] * np.expand_dims(area, axis=0)), axis=1) / RHO_WATER,
                dtype=pl.Float32,
            )
        )
    qlat = pl.DataFrame(qlat)
    qlat.write_parquet(desfile)
    return desfile


if __name__ == "__main__":
    DATAROOT = Path("../data")
    DATETIME_REF = datetime(2000, 1, 1)

    WEIGHT_FILE = DATAROOT.joinpath("rivnet_weight_experiment.parquet")
    rivnet = pl.read_parquet(WEIGHT_FILE, columns=["uuid", "latidx", "lonidx", "area"])

    EXP_FILE = DATAROOT.joinpath("wrf_experiment.nc")

    rns = []
    with h5netcdf.File(EXP_FILE) as f:
        tm = [DATETIME_REF + timedelta(seconds=x) for x in f.variables["time"][:]]
        nexp = len(f.dimensions["ensemble"])
        for iexp in range(nexp):
            rn = np.array(f.variables["mrro"][iexp, :, :, :], dtype=np.float32)
            rns.append(rn)

    args = []
    for exp in range(nexp):
        desfile = DATAROOT.joinpath("qlat", f"qlat_{exp:02d}.parquet")
        args.append((f"exp{exp:02d}", rivnet, tm, rns[exp], desfile))
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(convert_filepairs, args)
        for result in results:
            print(result)
