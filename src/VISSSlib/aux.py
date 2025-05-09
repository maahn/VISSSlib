import glob
import logging
import os
import warnings

import numpy as np
import pandas as pd
import requests
import xarray as xr

from . import __version__, files, scripts, tools

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# for performance
logDebug = log.isEnabledFor(logging.DEBUG)


def getCloudnet(date, site, path, kind, item):
    print(f"downloading {item} for {date}")
    url = "https://cloudnet.fmi.fi/api/files"
    payload = {
        "date": date,
        kind: item,
        "site": site,
    }
    metadata = requests.get(url, payload).json()
    if (len(metadata) == 0) or ("status" in metadata[0].keys()):
        raise FileNotFoundError(url)
    fnames = []
    for row in metadata[:1]:
        res = requests.get(row["downloadUrl"])
        fname = f"{path}/{row['filename']}"
        with tools.open2(fname, "wb") as f:
            f.write(res.content)
        fnames.append(fname)
    if len(metadata) > 1:
        log.warning("Found more than one file on Cloudnet")
    print(f"done {fnames}")
    return fnames


def getARM(date, site, path, product, user):
    print(f"downloading {product} for {date}")
    url = "https://adc.arm.gov/armlive/data/query"
    payload = {
        "user": user,
        "ds": f"{site}{product}",
        "start": date,
        "end": date,
        "wt": "json",
    }
    metadata = requests.get(url, payload).json()
    fnames = []
    if metadata["status"] == "success":
        for file in metadata["files"]:
            url = "https://adc.arm.gov/armlive/data/saveData"
            payload = {
                "user": user,
                "file": file,
            }
            fname = f"{path}/{file}"
            res = requests.get(url, payload)
            with tools.open2(fname, "wb") as f:
                f.write(res.content)
            fnames.append(fname)
    else:
        raise FileNotFoundError(metadata["status"])

    print(f"done {fnames}")
    return fnames


def getMeteoData(case, config):
    if type(config) is str:
        config = tools.readSettings(config)

    fn = files.FindFiles(case, config.leader, config)
    dat = _getMeteoData1(case, config)

    # add data of previous files - new fiels ar enot always reated at 00:00
    fnY = files.FindFiles(fn.yesterday, config.leader, config)
    datY = _getMeteoData1(fn.yesterday, config)

    # merge data
    dat = xr.concat((datY, dat), dim="time")
    today = (dat.time >= fn.datetime64) & (
        dat.time < (fn.datetime64 + np.timedelta64(1, "D"))
    )
    dat = dat.isel(time=today).load()
    return dat


def _getMeteoData1(case, config):
    if config.aux.meteo.source == "cloudnetMeteo":
        return getMeteoDataCloudnet(case, config)

    if config.aux.meteo.source == "ARMmet":
        return getMeteoDataARM(case, config)

    if config.aux.meteo.source == "RPG":
        return getMeteoDataRPG(case, config)

    else:
        raise ValueError(
            f"Do not understand config.aux.meteo.source:{config.aux.meteo.source}"
        )


def getMeteoDataCloudnet(case, config):
    fn = files.FindFiles(case, config.leader, config)
    date = f"{fn.year}-{fn.month}-{fn.day}"

    fStr = f"{config.aux.meteo.path.format(year=fn.year, month=fn.month, day=fn.day)}/{case}_{config.aux.cloudnet.site}_weather-station_*.nc"
    fnames = glob.glob(fStr)

    if config.aux.meteo.downloadData and (len(fnames) == 0):
        fnames = getCloudnet(
            date,
            config.aux.cloudnet.site,
            config.aux.meteo.path.format(year=fn.year, month=fn.month, day=fn.day),
            "instrument",
            "weather-station",
        )

    if len(fnames) == 0:
        raise FileNotFoundError(f"Did not find {fStr}")
    dat = xr.open_mfdataset(fnames)
    assert config.level2.freq == "1min"

    dat = dat[
        [
            "air_temperature",
            "relative_humidity",
            "air_pressure",
            "wind_speed",
            "wind_direction",
        ]
    ]
    # timestamps are a couple ns off
    dat = dat.resample(time=config.level2.freq, label="left").nearest()
    return dat


def getMeteoDataARM(case, config):
    fn = files.FindFiles(case, config.leader, config)
    date = f"{fn.year}-{fn.month}-{fn.day}"
    product = "met"

    path = config.aux.meteo.path.format(
        site=config.aux.ARM.site, product=product, year=fn.year
    )

    fStr = f"{path}/{config.aux.ARM.site}{product}*{case}*.cdf"
    fnames = glob.glob(fStr)

    if config.aux.meteo.downloadData and (len(fnames) == 0):
        fnames = getARM(date, config.aux.ARM.site, path, product, config.aux.ARM.user)

    if len(fnames) == 0:
        raise FileNotFoundError(f"Did not find {fStr}")
    dat = xr.open_mfdataset(fnames)
    assert config.level2.freq == "1min"
    dat

    vars = [
        "temp_mean",
        "rh_mean",
        "atmos_pressure",
        "wspd_vec_mean",
        "wdir_vec_mean",
    ]

    for var in vars:
        dat[var] = dat[var].where(dat[f"qc_{var}"] == 0)
    dat = dat[vars]

    dat = dat.rename(
        {
            "temp_mean": "air_temperature",
            "rh_mean": "relative_humidity",
            "atmos_pressure": "air_pressure",
            "wspd_vec_mean": "wind_speed",
            "wdir_vec_mean": "wind_direction",
        }
    )

    dat["air_temperature"] = dat["air_temperature"] + 273.15
    dat["relative_humidity"] = dat["relative_humidity"] / 100
    dat["air_pressure"] = dat["air_pressure"] * 1000

    return dat


def getMeteoDataRPG(case, config):
    fn = files.FindFiles(case, config.leader, config)
    date = f"{fn.year}-{fn.month}-{fn.day}"

    path = config.aux.meteo.path.format(year=fn.year, month=fn.month, day=fn.day)
    fStr = f"{path}/*ZEN.LV1.NC"
    fnames = glob.glob(fStr)

    if len(fnames) == 0:
        raise FileNotFoundError(f"Did not find {fStr}")
    dat = xr.open_mfdataset(fnames)
    assert config.level2.freq == "1min"

    vars = [
        "SurfTemp",
        "SurfRelHum",
        "SurfPres",
        "SurfWS",
        "SurfWD",
    ]

    dat = dat[vars]

    dat = dat.rename(
        {
            "SurfTemp": "air_temperature",
            "SurfRelHum": "relative_humidity",
            "SurfPres": "air_pressure",
            "SurfWS": "wind_speed",
            "SurfWD": "wind_direction",
            "Time": "time",
        }
    )

    dat["relative_humidity"] = dat["relative_humidity"] / 100
    dat["air_pressure"] = dat["air_pressure"] * 1000
    dat["time"] = netCDF4.num2date(
        dat.time.values,
        "seconds since 2001-01-01T00:00:00",
        only_use_python_datetimes=True,
        only_use_cftime_datetimes=False,
    )

    dat = dat.resample(time=config.level2.freq, label="left").nearest()
    return dat


def getRadarData(
    case,
    config,
):
    if type(config) is str:
        config = tools.readSettings(config)

    fn = files.FindFiles(case, config.leader, config)

    dat, frequency = _getRadarData1(case, config, fn)

    # add data of previous files - new fiels ar enot always reated at 00:00
    fnY = files.FindFiles(fn.yesterday, config.leader, config)
    datY, frequencyY = _getRadarData1(fn.yesterday, config, fnY)

    # merge data
    dat = xr.concat((datY, dat), dim="time")
    today = (dat.time >= fn.datetime64) & (
        dat.time < (fn.datetime64 + np.timedelta64(1, "D"))
    )
    dat = dat.isel(time=today).load()
    return dat, frequency


def _getRadarData1(case, config, fn):
    if config.aux.radar.source == "cloudnetCategorize":
        dat, frequency = getRadarDataCloudnetCategorize(case, config, fn)

    elif config.aux.radar.source == "cloudnetFMCW94":
        dat, frequency = getRadarDataCloudnetFMCW94(case, config, fn)

    elif config.aux.radar.source == "ARMwcloudradarcel":
        dat, frequency = getRadarDataARMwcloudradarcel(case, config, fn)

    else:
        raise ValueError(
            f"Do not understand config.aux.radar.source:{config.aux.radar.source}"
        )

    dat = dat.isel(range=slice(*config.aux.radar.heightIndices))
    # dat["Z_error"] = 10 ** (0.1 * dat["Z_error"])

    dat["time"] = dat.time + np.timedelta64(config.aux.radar.timeOffset, "s")
    dat = dat.resample(time=config.level2.freq, label="left").mean()

    dat["Ze"] = 10 * np.log10(dat["Ze"])
    # dat["Z_error"] = 10 * np.log10(dat["Z_error"])

    # do a linear extrapolation based on the lowest config.aux.radar.heightIndices data points
    fit = dat.polyfit(dim="range", deg=1)
    hnew = xr.DataArray([0], coords={"range": [0]})
    dat["Ze_ground"] = xr.polyval(coord=hnew, coeffs=fit.Ze_polyfit_coefficients).isel(
        range=0, drop=True
    )
    dat["MDV_ground"] = xr.polyval(
        coord=hnew, coeffs=fit.MDV_polyfit_coefficients
    ).isel(range=0, drop=True)

    dat["Ze_0"] = dat["Ze"].isel(range=0)
    dat["MDV_0"] = dat["MDV"].isel(range=0)

    dat["Ze_std"] = dat["Ze"].std("range")
    dat["MDV_std"] = dat["MDV"].std("range")

    dat = dat.drop_vars(["Ze", "MDV"])
    return dat, frequency


def getRadarDataCloudnetCategorize(case, config, fn):
    date = f"{fn.year}-{fn.month}-{fn.day}"

    fStr = f"{config.aux.radar.path.format(year=fn.year, month=fn.month, day=fn.day)}/{case}_{config.aux.cloudnet.site}_categorize*.nc"
    fnames = glob.glob(fStr)

    if config.aux.radar.downloadData and (len(fnames) == 0):
        fnames = getCloudnet(
            date,
            config.aux.cloudnet.site,
            config.aux.radar.path.format(year=fn.year, month=fn.month, day=fn.day),
            "product",
            "categorize",
        )

    if len(fnames) == 0:
        raise FileNotFoundError(f"Did not find {fStr}")

    dat = xr.open_mfdataset(fnames, preprocess=lambda dat: dat[["v", "Z", "altitude"]])

    dat = dat.rename(v="MDV", Z="Ze", height="range")
    dat1 = dat[
        [
            "Ze",
            "MDV",
        ]
    ]
    dat1["range"] = dat1.range - float(dat.altitude.values)
    dat1["Ze"] = 10 ** (0.1 * dat1["Ze"])

    return dat1, float(dat.radar_frequency.values)


def getRadarDataCloudnetFMCW94(case, config, fn):
    date = f"{fn.year}-{fn.month}-{fn.day}"

    fStr = f"{config.aux.radar.path.format(year=fn.year, month=fn.month, day=fn.day)}/{case}_{config.aux.cloudnet.site}_rpg-fmcw-94*.nc"
    fnames = glob.glob(fStr)

    if config.aux.radar.downloadData and (len(fnames) == 0):
        fnames = getCloudnet(
            date,
            config.aux.cloudnet.site,
            config.aux.radar.path.format(year=fn.year, month=fn.month, day=fn.day),
            "instrument",
            "rpg-fmcw-94",
        )

    if len(fnames) == 0:
        raise FileNotFoundError(f"Did not find {fStr}")

    dat = xr.open_mfdataset(fnames, preprocess=lambda dat: dat[["v", "Zh"]])

    dat = dat.rename(v="MDV", Zh="Ze")
    dat1 = dat[
        [
            "Ze",
            "MDV",
        ]
    ]
    dat1["Ze"] = 10 ** (0.1 * dat1["Ze"])

    return dat1, float(dat.radar_frequency.values)


def getRadarDataARMwcloudradarcel(case, config, fn):
    date = f"{fn.year}-{fn.month}-{fn.day}"

    fStr = f"{config.aux.radar.path.format(year=fn.year, month=fn.month, day=fn.day)}/{config.aux.ARM.site}wcloudradarcel*{case}*.nc"
    fnames = glob.glob(fStr)

    if config.aux.radar.downloadData and (len(fnames) == 0):
        raise FileNotFoundError("IOP data needs to be downloaded from ARM manually")

    if len(fnames) == 0:
        raise FileNotFoundError(f"Did not find {fStr}")
    dat = xr.open_mfdataset(
        fnames, preprocess=lambda dat: dat[["MeanVel", "ZE", "Elv", "Freq"]]
    )

    dat = dat.rename(MeanVel="MDV", ZE="Ze")
    dat1 = dat[
        [
            "Ze",
            "MDV",
        ]
    ].where(np.abs(dat.Elv - config.aux.radar.elevation) < 0.5)

    return dat1, float(dat.Freq.mean().values)
