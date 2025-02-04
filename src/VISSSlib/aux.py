import glob
import logging

import numpy as np
import request
import xarray as xr

from . import __version__, files, scripts, tools

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# for performance
logDebug = log.isEnabledFor(logging.DEBUG)


def getMeteoData(date, config):
    if type(config) is str:
        config = tools.readSettings(config)

    if config.aux.meteo.repository == "cloudnet":
        return getMeteoDataCloudnet(date, config)

    else:
        raise ValueError(
            f"Do not understand config.aux.meteoKind:{config.aux.meteo.repository}"
        )


def getMeteoDataCloudnet(date, config):
    fn = files.FindFiles(case, config.leader, config)
    date = f"{fn.year}-{fn.month}-{fn.day}"

    fname = f"{config.aux.meteo.path.format(year=fn.year, month=fn.month, day=fn.day)}/{case}_{config.aux.cloudnet.site}_weather-station.nc"

    if not os.path.isfile(fname):
        print(f"downloading {fname}")
        url = "https://cloudnet.fmi.fi/api/files"
        payload = {
            "date": date,
            "instrument": "weather-station",
            "site": config.aux.cloudnet.site,
        }
        metadata = requests.get(url, payload).json()
        for row in metadata:
            res = requests.get(row["downloadUrl"])
            with tools.open2(fname, "wb") as f:
                f.write(res.content)

        print(f"done")

    dat = xr.open_dataset(fname)

    assert config.level2.freq == "1min"

    return dat[
        [
            "air_temperature",
            "relative_humidity",
            "air_pressure",
            "wind_speed",
            "wind_direction",
        ]
    ]


def getRadarData(date, config):
    if type(config) is str:
        config = tools.readSettings(config)

    if config.aux.radar.repository == "cloudnet":
        return getRadarDataCloudnet(date, config)

    else:
        raise ValueError(
            f"Do not understand config.aux.meteoKind:{config.aux.radar.repository}"
        )


def getRadarDataCloudnet(date, config):
    fn = files.FindFiles(case, config.leader, config)
    date = f"{fn.year}-{fn.month}-{fn.day}"

    fname = f"{config.aux.radar.path.format(year=fn.year, month=fn.month, day=fn.day)}/{case}_{config.aux.cloudnet.site}_categorize.nc"

    if not os.path.isfile(fname):
        print(f"downloading {fname}")
        url = "https://cloudnet.fmi.fi/api/files"
        payload = {
            "date": date,
            "product": "categorize",
            "site": config.aux.cloudnet.site,
        }
        metadata = requests.get(url, payload).json()
        for row in metadata:
            res = requests.get(row["downloadUrl"])
            with tools.open2(fname, "wb") as f:
                f.write(res.content)

        print(f"done")

    dat = xr.open_dataset(fname)
    dat = dat.rename(v="MDV")
    dat = dat[["Z", "Z_error", "radar_frequency", "MDV"]]

    dat["Z"] = 10 ** (0.1 * dat["Z"])
    dat["Z_error"] = 10 ** (0.1 * dat["Z_error"])

    timeIndex = pd.date_range(
        start=case, end=fn.datetime64 + endTime, freq=freq, inclusive="left"
    )
    dat.resample(time=freq).mean()

    dat["Z"] = 10 * np.log10(dat["Z"])
    dat["Z_error"] = 10 * np.log10(dat["Z_error"])

    return dat
