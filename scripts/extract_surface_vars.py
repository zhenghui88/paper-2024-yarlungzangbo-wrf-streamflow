#!/usr/bin/env python
# coding: utf-8

import argparse
import datetime
import glob
import os
from collections import OrderedDict

import netCDF4 as nc
import numpy as np

TIMEREF = datetime.datetime(2000, 1, 1)


def get_timestamp(ncfile):
    timestamp = []
    with nc.Dataset(ncfile) as ncf:
        for dtvec in ncf.variables["Times"][:, :]:
            dtstr = dtvec.tobytes().decode("ascii")
            timestamp.append(datetime.datetime.fromisoformat(dtstr))
    return timestamp


def get_filelist(srcdir, prefix):
    files = {}
    for f in glob.iglob(os.path.join(srcdir, f"{prefix}*")):
        for t in get_timestamp(f):
            files[t] = f
    filelist = OrderedDict(sorted(files.items(), key=lambda t: t[0]))
    return filelist


def get_latlon(ncfile):
    with nc.Dataset(ncfile) as ncf:
        ncf.set_auto_mask(False)
        lat = ncf.variables["XLAT"][0, :, :]
        lon = ncf.variables["XLONG"][0, :, :]
    return lat, lon


def get_variable(cdt, cfile):
    with nc.Dataset(cfile) as ncf:
        ncf.set_auto_mask(False)
        for ii, dtvec in enumerate(ncf.variables["Times"][:, :]):
            dtstr = dtvec.tobytes().decode("ascii")
            if cdt == datetime.datetime.fromisoformat(dtstr):
                break
        else:
            return None, None, None
        return (
            ncf.variables["U10"][ii, ...],
            ncf.variables["V10"][ii, ...],
            ncf.variables["T2"][ii, ...],
            ncf.variables["Q2"][ii, ...],
            ncf.variables["PSFC"][ii, ...],
            ncf.variables["SWDOWN"][ii, ...],
            ncf.variables["GLW"][ii, ...],
            ncf.variables["RAINC"][ii, ...],
            ncf.variables["RAINNC"][ii, ...],
            ncf.variables["SNOWNC"][ii, ...],
            ncf.variables["GRAUPELNC"][ii, ...],
            ncf.variables["HAILNC"][ii, ...],
            ncf.variables["SFROFF"][ii, ...],
            ncf.variables["UDROFF"][ii, ...],
        )


def extract(desfile, srcdir, prefix="wrfout_d01_"):
    srcfiles = get_filelist(srcdir, prefix)
    lat, lon = get_latlon(list(srcfiles.values())[0])
    nsn, nwe = lat.shape
    u10 = np.empty((nsn, nwe))
    v10 = np.empty((nsn, nwe))
    t2 = np.empty((nsn, nwe))
    q2 = np.empty((nsn, nwe))
    psfc = np.empty((nsn, nwe))
    swdown = np.empty((nsn, nwe))
    glw = np.empty((nsn, nwe))
    rainc = np.empty((nsn, nwe))
    rainc_p = np.empty_like(rainc)
    rainnc = np.empty((nsn, nwe))
    rainnc_p = np.empty_like(rainnc)
    snownc = np.empty((nsn, nwe))
    snownc_p = np.empty_like(snownc)
    graupelnc = np.empty((nsn, nwe))
    graupelnc_p = np.empty_like(graupelnc)
    hailnc = np.empty((nsn, nwe))
    hailnc_p = np.empty_like(hailnc)
    sfroff = np.empty((nsn, nwe))
    sfroff_p = np.empty_like(sfroff)
    udroff = np.empty((nsn, nwe))
    udroff_p = np.empty_like(udroff)
    with nc.Dataset(desfile, "w") as ncf:
        ncf.createDimension("time", None)
        ncf.createDimension("south_north", nsn)
        ncf.createDimension("west_east", nwe)
        ncf.createVariable("time", "f8", ("time",), zlib=True, complevel=6)
        ncf.createVariable(
            "XLAT", "f8", ("south_north", "west_east"), zlib=True, complevel=6
        )
        ncf.variables["XLAT"].units = "degree_north"
        ncf.variables["XLAT"].description = "latitude"
        ncf.createVariable(
            "XLONG", "f8", ("south_north", "west_east"), zlib=True, complevel=6
        )
        ncf.variables["XLONG"].units = "degree_east"
        ncf.variables["XLONG"].description = "longitude"
        ncf.createVariable(
            "U10", "f4", ("time", "south_north", "west_east"), zlib=True, complevel=6
        )
        ncf.variables["U10"].units = "m s-1"
        ncf.variables["U10"].description = "eastward wind component at 10 m"
        ncf.createVariable(
            "V10", "f4", ("time", "south_north", "west_east"), zlib=True, complevel=6
        )
        ncf.variables["V10"].units = "m s-1"
        ncf.variables["V10"].description = "northward wind component at 10 m"
        ncf.variables["time"].units = f"seconds since {TIMEREF:%Y-%m-%d %H:%M:%S}"
        ncf.createVariable(
            "T2", "f4", ("time", "south_north", "west_east"), zlib=True, complevel=6
        )
        ncf.variables["T2"].units = "K"
        ncf.variables["T2"].description = "temperature at 2 m"
        ncf.createVariable(
            "Q2", "f4", ("time", "south_north", "west_east"), zlib=True, complevel=6
        )
        ncf.variables["Q2"].units = "kg kg-1"
        ncf.variables["Q2"].description = "mixing ratio at 2 m"
        ncf.createVariable(
            "PSFC", "f4", ("time", "south_north", "west_east"), zlib=True, complevel=6
        )
        ncf.variables["PSFC"].units = "Pa"
        ncf.variables["PSFC"].description = "surface pressure"
        ncf.createVariable(
            "SWDOWN", "f4", ("time", "south_north", "west_east"), zlib=True, complevel=6
        )
        ncf.variables["SWDOWN"].units = "W m-2"
        ncf.variables[
            "SWDOWN"
        ].description = "downward shortwave radiation at the surface"
        ncf.createVariable(
            "GLW", "f4", ("time", "south_north", "west_east"), zlib=True, complevel=6
        )
        ncf.variables["GLW"].units = "W m-2"
        ncf.variables["GLW"].description = "downward longwave radiation at the surface"
        ncf.createVariable(
            "RAINC", "f4", ("time", "south_north", "west_east"), zlib=True, complevel=6
        )
        ncf.variables["RAINC"].units = "kg m-2 s-1"
        ncf.variables["RAINC"].description = "total cumulus precpitation rate"
        ncf.createVariable(
            "RAINNC", "f4", ("time", "south_north", "west_east"), zlib=True, complevel=6
        )
        ncf.variables["RAINNC"].units = "kg m-2 s-1"
        ncf.variables["RAINNC"].description = "total grid scale precipitation rate"
        ncf.createVariable(
            "SNOWNC", "f4", ("time", "south_north", "west_east"), zlib=True, complevel=6
        )
        ncf.variables["SNOWNC"].units = "kg m-2 s-1"
        ncf.variables["SNOWNC"].description = "total grid scale snow and ice rate"
        ncf.createVariable(
            "GRAUPELNC",
            "f4",
            ("time", "south_north", "west_east"),
            zlib=True,
            complevel=6,
        )
        ncf.variables["GRAUPELNC"].units = "kg m-2 s-1"
        ncf.variables["GRAUPELNC"].description = "total grid scale graupel rate"
        ncf.createVariable(
            "HAILNC", "f4", ("time", "south_north", "west_east"), zlib=True, complevel=6
        )
        ncf.variables["HAILNC"].units = "kg m-2 s-1"
        ncf.variables["HAILNC"].description = "total grid scale hail rate"
        ncf.createVariable(
            "SFROFF", "f4", ("time", "south_north", "west_east"), zlib=True, complevel=6
        )
        ncf.variables["SFROFF"].units = "kg m-2 s-1"
        ncf.variables["SFROFF"].description = "surface runoff"
        ncf.createVariable(
            "UDROFF", "f4", ("time", "south_north", "west_east"), zlib=True, complevel=6
        )
        ncf.variables["UDROFF"].units = "kg m-2 s-1"
        ncf.variables["UDROFF"].description = "subsurface runoff"
        ncf.variables["XLAT"][:] = lat
        ncf.variables["XLONG"][:] = lon
        for idt, (cdt, cfile) in enumerate(srcfiles.items()):
            print(cdt)
            (
                u10[:, :],
                v10[:, :],
                t2[:, :],
                q2[:, :],
                psfc[:, :],
                swdown[:, :],
                glw[:, :],
                rainc[:, :],
                rainnc[:, :],
                snownc[:, :],
                graupelnc[:, :],
                hailnc[:, :],
                sfroff[:, :],
                udroff[:, :],
            ) = get_variable(cdt, cfile)
            dt = (
                (cdt - list(srcfiles.keys())[idt - 1]).total_seconds()
                if idt >= 0
                else 0
            )
            ncf.variables["time"][idt] = (cdt - TIMEREF).total_seconds()
            ncf.variables["U10"][idt, ...] = u10
            ncf.variables["V10"][idt, ...] = v10
            ncf.variables["T2"][idt, ...] = t2
            ncf.variables["Q2"][idt, ...] = q2
            ncf.variables["PSFC"][idt, ...] = psfc
            ncf.variables["SWDOWN"][idt, ...] = swdown
            ncf.variables["GLW"][idt, ...] = glw
            ncf.variables["RAINC"][idt, ...] = (
                (rainc - rainc_p) / dt if idt > 0 else 0.0
            )
            ncf.variables["RAINNC"][idt, ...] = (
                (rainnc - rainnc_p) / dt if idt > 0 else 0.0
            )
            ncf.variables["SNOWNC"][idt, ...] = (
                (snownc - snownc_p) / dt if idt > 0 else 0.0
            )
            ncf.variables["GRAUPELNC"][idt, ...] = (
                (graupelnc - graupelnc_p) / dt if idt > 0 else 0.0
            )
            ncf.variables["HAILNC"][idt, ...] = (
                (hailnc - hailnc_p) / dt if idt > 0 else 0.0
            )
            ncf.variables["SFROFF"][idt, ...] = (
                (sfroff - sfroff_p) / dt if idt > 0 else 0.0
            )
            ncf.variables["UDROFF"][idt, ...] = (
                (udroff - udroff_p) / dt if idt > 0 else 0.0
            )
            rainc, rainc_p = rainc_p, rainc
            rainnc, rainnc_p = rainnc_p, rainnc
            snownc, snownc_p = snownc_p, snownc
            graupelnc, graupelnc_p = graupelnc_p, graupelnc
            hailnc, hailnc_p = hailnc_p, hailnc
            sfroff, sfroff_p = sfroff_p, sfroff
            udroff, udroff_p = udroff_p, udroff


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("desfile", help="the output netCDF4 file")
    parser.add_argument(
        "srcdir",
        help="the directory containing the source files with the pattern of 'PREFIX*'",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        default="wrfout_d01_",
        help="prefix of the wrfout files (default: wrfout_d01_)",
    )
    args = parser.parse_args()
    extract(args.desfile, args.srcdir, prefix=args.prefix)
