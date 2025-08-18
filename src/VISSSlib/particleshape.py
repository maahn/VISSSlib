import glob
import importlib.resources
import logging
import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from . import __version__, aux, files, tools

log = logging.getLogger(__name__)

# for performance
logDebug = log.isEnabledFor(logging.DEBUG)


def classifyParticles(fnameLv1Detect, config, writeNc=True):
    config = tools.readSettings(config)

    ffl1 = files.FilenamesFromLevel(fnameLv1Detect, config)

    fnameLv1Shape = ffl1.fname["level1shape"]
    fnameTracking = ffl1.fname["level1track"]

    if os.path.isfile(fnameTracking):
        lv1track = xr.open_dataset(fnameTracking)
        lv1track.load()  # important to do that early, is much slower after applying filters with isel
    elif os.path.isfile("%s.nodata" % fnameTracking):
        with tools.open2(f"{fnameLv1Shape}.nodata", "w") as f:
            f.write("no data, lv1track nodata ")
        log.error(f"NO DATA {fnameTracking}")
        return None, fnameLv1Shape
    elif os.path.isfile("%s.broken.txt" % fnameTracking):
        with tools.open2(f"{fnameLv1Shape}.broken.txt", "w") as f:
            f.write("no data, lv1track  broken")
        log.error(f"NO DATA {fnameTracking}")
        return None, fnameLv1Shape
    else:
        log.error(f"NO DATA lv1track yet {fnameTracking}")
        return None, fnameLv1Shape

    largeEnough = lv1track.Dmax.max("camera") >= config.level1shape.minDmax
    sharpEnough = lv1track.blur.max("camera") >= config.level1shape.minBlur
    matchedEnough = lv1track.matchScore >= config.quality.minMatchScore
    goodQuality = largeEnough & sharpEnough & matchedEnough

    if np.sum(goodQuality) == 0:
        log.warning(
            "no data, all particles too small/too blurry/not matched %s" % fnameLv1Shape
        )
        with tools.open2(f"{fnameLv1Shape}.nodata", "w") as f:
            f.write("no data, all particles too small")
        return None, fnameLv1Shape

    lv1track = lv1track.where(goodQuality)

    lv1shape = applyClassifier(lv1track, config)

    if lv1shape is None:
        with tools.open2(f"{fnameLv1Shape}.nodata", "w") as f:
            f.write("no data, all particles too small")
        return None, fnameLv1Shape

    lv1shape.predictedShape.attrs.update(
        dict(
            units="-",
            long_name="probability that particle corresponds to specified shape",
        )
    )

    lv1shape = tools.finishNc(lv1shape, config.site, config.visssGen)
    lv1shape.load()
    print(lv1shape)
    if writeNc:
        tools.to_netcdf2(lv1shape, fnameLv1Shape)
    print("DONE", fnameLv1Shape)
    return lv1shape, fnameLv1Shape


def add_variables(dataset):
    amp_rat = dataset.contourFFT / dataset.contourFFTsum
    dataset["amp_rat"] = xr.DataArray(
        amp_rat, coords=dataset.contourFFT.coords, dims=dataset.contourFFT.dims
    )
    return dataset


def rename_vars(ds, suffix):
    return ds.rename({var: f"{var}_{suffix}" for var in ds.data_vars})


def applyClassifier(DaTa, config):
    import joblib
    import sklearn
    import sklearn.ensemble
    import sklearn.feature_extraction
    import sklearn.linear_model
    import sklearn.model_selection
    import sklearn.neural_network
    import sklearn.preprocessing

    """
    by Veronika Ettrichrätz
    """

    vars_to_subs = [
        "Droi",
        "Dmax",
        "Dfit",
        "contourFFT",
        "amp_rat",
    ]

    DaTa = add_variables(DaTa)
    DaTa["Dfit"] = (DaTa["Dfit"]) / config.calibration.slope / 1e6
    DaTa["Droi"] = (DaTa["Droi"]) / config.calibration.slope / 1e6
    DaTa["Dmax"] = (DaTa["Dmax"]) / config.calibration.slope / 1e6

    diff_dataset = xr.Dataset()
    max_dataset = xr.Dataset()
    min_dataset = xr.Dataset()
    for var in vars_to_subs:
        # Berechne die absolute Differenz für die aktuelle Variable
        # diff = np.abs(DaTa[var].sel(camera='leader_S1145792') - DaTa[var].sel(camera='follower_S1143155'))
        diff = np.abs(DaTa[var][0] - DaTa[var][1])
        # Füge die Differenz als neue Variable in das neue Dataset ein
        diff_dataset[var] = diff
        maxis = DaTa[var].max(dim="camera")
        max_dataset[var] = maxis
        mins = DaTa[var].min(dim="camera")
        min_dataset[var] = mins

    # Umbenennen der Variablen in beiden Datasets
    ds1_renamed = rename_vars(max_dataset, "max")
    ds2_renamed = rename_vars(diff_dataset, "diff")
    ds3_renamed = rename_vars(min_dataset, "min")

    # Kombinieren der umbenannten Datasets
    combined_ds = xr.merge([ds1_renamed, ds2_renamed, ds3_renamed])

    combined_ds["pair_id"] = DaTa["pair_id"]

    combined_ds_nan_free = combined_ds.dropna(dim="pair_id", how="any")
    combined_ds_nan_free = combined_ds_nan_free.sel(
        FFTfreqs=[1.0, 2.0, 3.0, 6.0, 4.0, 8.0, 9.0, 16.0, 12.0], drop=True
    )

    if len(combined_ds_nan_free.pair_id) == 0:
        print("no data left")
        return None

    selected_variables = [
        "Droi_max",
        "Droi_diff",
        "Dfit_min",
        "Dfit_diff",
        "contourFFT_min",
        "contourFFT_diff",
        "amp_rat_max",
        "amp_rat_diff",
        "Dmax_max",
        "Dmax_diff",
    ]

    data_dict = {}

    # Schleife über die ausgewählten Variablen
    for var_name in selected_variables:
        # print(var_name)
        # print(len(DaTa_cleaned_limited_dataset_small_nan_free[var_name].shape) )
        if len(combined_ds_nan_free[var_name].shape) == 3:
            for j in range(combined_ds_nan_free[var_name].shape[2]):
                for k in range(combined_ds_nan_free[var_name].shape[2]):
                    key = f"{var_name}_{j}_{k}"
                    data_dict[key] = getattr(
                        combined_ds_nan_free[var_name][:, j, k], "values"
                    )
        elif len(combined_ds_nan_free[var_name].shape) == 2:
            for j in range(combined_ds_nan_free[var_name].shape[1]):
                key = f"{var_name}_{j}"
                data_dict[key] = getattr(combined_ds_nan_free[var_name][:, j], "values")
        else:  # len(DaTa_cleaned_limited_dataset_small_nan_free[var_name].shape) == 1:
            key = f"{var_name}"
            data_dict[key] = getattr(combined_ds_nan_free[var_name][:], "values")

    # data_array_large = np.array(list(data_dict_large.values())).T
    # data_array_small = np.array(list(data_dict_small.values())).T
    data_array = np.array(list(data_dict.values())).T

    if config.level1shape.classifier == "DEFAULT":
        pkg = importlib.resources.files("VISSSlib")
        cFile = pkg / "data" / "classifier_20250730.pkl"
    else:
        cFile = config.level1shape.classifier
    classifier = joblib.load(cFile)
    # classifier = classifier_large

    print("scale data")
    scaler = sklearn.preprocessing.StandardScaler().fit(data_array)
    scaled_data_array = scaler.transform(data_array)
    print("data scaled")
    # print("predict data")
    # preds = classifier.predict(scaled_data_array)
    print("data predicted")
    prob = classifier.predict_proba(scaled_data_array)
    # prob_max = prob.max(axis=1)
    print("probability calculated")
    # graupel = prob[:, 0]
    # needles_columns = prob[:, 1]
    # plates = prob[:, 2]
    # sphericals = prob[:, 3]
    # stellars = prob[:, 4]

    shapes = classifier.classes_
    prob = xr.DataArray(
        prob, dims=["pair_id", "shape"], coords=[combined_ds_nan_free.pair_id, shapes]
    )
    prob = xr.Dataset({"predictedShape": prob})

    # go back to intial index
    data = DaTa[["pair_id"]].combine_first(prob)

    return data
