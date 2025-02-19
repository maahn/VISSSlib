import glob
import logging
import os
import warnings

import numpy as np
import pandas as pd
import pyOptimalEstimation as pyOE
import xarray as xr

from . import __version__, aux, files, tools

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=Warning)


# for performance
logDebug = log.isEnabledFor(logging.DEBUG)


def ssrga_parameter(M, elevation):
    import pyPamtra

    if elevation == 90:
        return pyPamtra.descriptorFile.ssrga_parameter(M)
    elif np.abs(elevation - 50) < 11:
        p0 = 0.5035
        p1 = np.array([0.0168, 0.117, -2.648, -0.8126, 0.1125])
        p2 = np.array([0.1609, -0.0022, 0.6949, 1.6618, -0.1316])
        p3 = np.array([0.7234, 0.0429, 2.8542, 2.4369, 0.1158])

        alpha_eff = p1[0] * M ** (2 * p0) + p2[0] * M**p0 + p3[0]
        kappa = p1[1] * M ** (2 * p0) + p2[1] * M**p0 + p3[1]
        beta = p1[2] * M ** (2 * p0) + p2[2] * M**p0 + p3[2]
        gamma = p1[3] * M ** (2 * p0) + p2[3] * M**p0 + p3[3]
        zeta1 = p1[4] * M ** (2 * p0) + p2[4] * M**p0 + p3[4]

        return kappa, beta, gamma, zeta1, alpha_eff
    else:
        ValueError(f"elevation must be 90 or around 50. Got {elevation}")


def reflec_logM(
    X, ice, alt, temp, bins, dmean, dbound, shape, frequency, elevation
):  # ice
    import pyPamtra

    M = 10**X

    a = pyPamtra.descriptorFile.riming_dependent_mass_size(M, shape)[
        0
    ]  # needle, column, rosette, plate, mean
    b = pyPamtra.descriptorFile.riming_dependent_mass_size(M, shape)[1]

    scattering = "ss-rayleigh-gans"

    temp_lev = np.array([temp, temp])
    hgt_lev = np.array([alt - 5, alt + 5])

    dsd_i = ice
    nBins = bins
    Dmean = dmean
    Dbound = dbound
    alt6 = alt

    pam = pyPamtra.pyPamtra()

    pam.df.addHydrometeor(
        (
            "ice",  # name
            -99.0,  # aspect ratio (NOT RELEVANT)
            -1,  # liquid - ice flag
            -99.0,  # density (NOT RELEVANT)
            -99.0,  # mass size relation prefactor a (NOT RELEVANT)
            -99.0,  # mass size relation exponent b (NOT RELEVANT)
            -99.0,  # area size relation prefactor alpha (NOT RELEVANT)
            -99.0,  # area size relation exponent beta (NOT RELEVANT)
            0,  # moment provided later (NOT RELEVANT)
            nBins,  # number of bins
            "fullBin",  # distribution name (NOT RELEVANT)
            -99.0,  # distribution parameter 1 (NOT RELEVANT)
            -99.0,  # distribution parameter 2 (NOT RELEVANT)
            -99.0,  # distribution parameter 3 (NOT RELEVANT)
            -99.0,  # distribution parameter 4 (NOT RELEVANT)
            -99.0,  # minimum diameter (NOT RELEVANT)
            -99.0,  # maximum diameter (NOT RELEVANT)
            scattering,  # scattering model
            "heymsfield10_particles",  # fall velocity relation  (NOT RELEVANT)
            0.0,  # canting angle  (NOT RELEVANT)
        )
    )

    pam = pyPamtra.importer.createUsStandardProfile(
        pam, hgt_lev=list(hgt_lev), temp_lev=list(temp_lev)
    )

    pam.p["turb_edr"][:] = 1e-4

    pam.nmlSet["passive"] = False  # passive mode
    pam.nmlSet["active"] = True  # active mode

    # 0 is real noise, -1 means that the seed is created from latitude and longitude, other value gives always the same random numbers
    pam.nmlSet["randomseed"] = 0
    # Use “simple” radar simulator provides only Z_e by integrating over D. The advanced “spectrum” simulator simulates the complete radar Doppler spectrum and estimates all moments from the spectrum. “moments” is identical to “spectrum” but the full Doppler spectrum is discarded to save memory.
    pam.nmlSet["radar_mode"] = "simple"

    pam.nmlSet[
        "hydro_fullspec"
    ] = True  # pass values directly from python to PAMTRA using numpy arrays.

    pam.p["sfc_type"] = np.zeros(pam._shape2D)
    pam.p["sfc_model"] = np.zeros(pam._shape2D)
    pam.p["sfc_refl"] = np.chararray(pam._shape2D)
    pam.p["sfc_refl"][pam.p["sfc_type"] == 0] = "F"
    pam.p["obs_height"][:, 0] = 0.0

    pam.nmlSet[
        "radar_attenuation"
    ] = "top-down"  # include attenuation by gas and hydrometeors
    pam.set["verbose"] = 0
    pam.set["pyVerbose"] = 0

    pam.df.addFullSpectra()

    pam.df.dataFullSpec["d_bound_ds"][:] = Dbound
    pam.df.dataFullSpec["d_ds"][:] = Dmean
    pam.df.dataFullSpec["n_ds"][0, 0, :, 0, :] = (dsd_i) * np.diff(Dbound)

    # snow
    pam.df.dataFullSpec["area_ds"][0, 0, :, 0, :] = (
        0.3898 * pam.df.dataFullSpec["d_ds"][0, 0, :, 0, :] ** 1.977
    )
    pam.df.dataFullSpec["mass_ds"][0, 0, :, 0, :] = (
        a * pam.df.dataFullSpec["d_ds"][0, 0, :, 0, :] ** b
    )

    kappa, beta, gamma, zeta, alpha = ssrga_parameter(M, elevation)

    pam.df.dataFullSpec["as_ratio"][0, 0, :, 0, :] = alpha
    pam.df.dataFullSpec["rg_kappa_ds"][0, 0, :, 0, :] = kappa
    pam.df.dataFullSpec["rg_beta_ds"][0, 0, :, 0, :] = beta
    pam.df.dataFullSpec["rg_gamma_ds"][0, 0, :, 0, :] = gamma
    pam.df.dataFullSpec["rg_zeta_ds"][0, 0, :, 0, :] = zeta
    pam.df.dataFullSpec["rho_ds"][0, 0, :, 0, :] = (
        6.0 * pam.df.dataFullSpec["mass_ds"][0, 0, :, 0, :]
    ) / (np.pi * pam.df.dataFullSpec["d_ds"][0, 0, :, 0, :] ** 3.0 * 0.6)

    pam.runPamtra([frequency])

    Ze = pam.r["Ze"].squeeze()
    att_hydro = pam.r["Att_hydro"].squeeze()
    att_atmo = pam.r["Att_atmo"].squeeze()

    Ze[Ze == -9.99900000e03] = np.nan

    return Ze


def mass_size(M):
    """
    helper function to get a and b from interpolation as a function of M following
    Maherndl, N., M. Maahn, F. Tridon, J. Leinonen, D. Ori, and S. Kneifel, 2023:
    A riming-dependent parameterization of scattering by snowflakes using the
    self-similar rayleigh–gans approximation. Q. J. R. Meteorolog. Soc., 149,
    3562–3581, doi:10.1002/qj.4573.

    written by N. Maherndl
    """

    M_list = np.array(
        [
            0.0,
            0.0129,
            0.02045,
            0.03245,
            0.05145,
            0.08155,
            0.129,
            0.2045,
            0.3245,
            0.5145,
            0.8155,
        ]
    )
    if hasattr(M, "__len__"):
        M[M > M_list[-1]] = M_list[-1]

    else:
        if M > M_list[-1]:
            M = M_list[-1]

    a_m_list = np.array(
        [0.0324, 0.224, 0.537, 1.54, 4.27, 10.1, 22.2, 43.3, 79.0, 157.0, 173.0]
    )
    b_m_list = np.array(
        [2.10, 2.35, 2.45, 2.57, 2.69, 2.77, 2.85, 2.89, 2.93, 2.97, 2.93]
    )

    a_int = interp1d(M_list, a_m_list, kind="cubic")
    b_int = interp1d(M_list, b_m_list, kind="cubic")

    a_m = a_int(M)
    b_m = b_int(M)

    return a_m, b_m


def dynamic_viscosity_air(temperature):
    """
    ! This function returns the dynamic viscosity of dry air in Pa s
    ! Sutherland law
    ! coefficients from F. M. White, Viscous Fluid Flow, 2nd ed., McGraw-Hill,
    ! (1991). Kim et al., arXiv:physics/0410237v1
    """

    mu0 = 1.716e-5  # Pas
    T0 = 273.0
    C = 111.0  # K

    eta = mu0 * ((T0 + C) / (temperature + C)) * (temperature / T0) ** 1.5

    return eta


def dry_density_air(temperature, press):
    R_s = 287.0500676
    rho = press / (R_s * temperature)

    return rho


def heymsfield10_particles_M(Dmax, M, temperature, press, shape):
    import pyPamtra

    dynamicViscosity = dynamic_viscosity_air(temperature)
    dryAirDensity = dry_density_air(temperature, press)

    a, b = pyPamtra.descriptorFile.riming_dependent_mass_size(M, shape)
    aa, ba = pyPamtra.descriptorFile.riming_dependent_area_size(M, shape)

    a = xr.DataArray(a, [M.time])
    b = xr.DataArray(b, [M.time])
    aa = xr.DataArray(aa, [M.time])
    ba = xr.DataArray(ba, [M.time])

    mass = a * Dmax**b
    crossSectionArea = aa * Dmax**ba

    k = 0.5  # defined in the paper
    delta_0 = 8.0
    C_0 = 0.35
    g = 9.81

    area_proj = crossSectionArea / ((np.pi / 4.0) * Dmax**2)

    # eq 9
    Xstar = (
        8.0
        * dryAirDensity
        * mass
        * g
        / (np.pi * area_proj ** (1.0 - k) * dynamicViscosity**2)
    )
    # eq10
    Re = (
        delta_0**2
        / 4.0
        * ((1.0 + ((4.0 * np.sqrt(Xstar)) / (delta_0**2 * np.sqrt(C_0)))) ** 0.5 - 1)
        ** 2
    )

    velSpec = dynamicViscosity * Re / (dryAirDensity * Dmax)
    return velSpec


def retrieveCombinedRiming(case, config, skipExisting=True, writeNc=True):
    """
    apply combined riming retrieval following
    Maherndl, N., M. Maahn, F. Tridon, J. Leinonen, D. Ori, and S. Kneifel, 2023:
    A riming-dependent parameterization of scattering by snowflakes using the
    self-similar rayleigh–gans approximation. Q. J. R. Meteorolog. Soc., 149,
    3562–3581, doi:10.1002/qj.4573.

    written by N. Maherndl

    """
    import pyPamtra

    if type(config) is str:
        config = tools.readSettings(config)
    fL = files.FindFiles(case, config.leader, config)

    lv3File = fL.fnamesDaily[f"level3combinedRiming"]

    log.info(f"Processing {lv3File}")

    if (
        writeNc
        and skipExisting
        and tools.checkForExisting(
            lv3File,
            parents=fL.listFilesExt(f"level2track"),
        )
    ):
        return None, None

    if (
        writeNc
        and skipExisting
        and tools.checkForExisting(
            "%s.nodata" % lv3File,
            parents=fL.listFilesExt(f"level2track"),
        )
    ):
        return None, None

    if np.all([f.endswith("broken.txt") for f in fL.listFilesExt(f"level2track")]):
        raise RuntimeError(
            f"All level2track in {fL.fnamesPatternExt[f'level2track']} are broken."
        )

    isEmpty = fL.listFilesExt(f"level2track")[0].endswith("nodata")
    if isEmpty:
        with tools.open2("%s.nodata" % lv3File, "w") as f:
            f.write("no data for %s" % case)
        log.warning("no data for %s" % case)
        log.warning("written: %s.nodata" % lv3File)
        return None, lv3File

    radarDat, frequency = aux.getRadarData(case, config)
    meteoDat = aux.getMeteoData(case, config)
    lv2DatA = xr.open_dataset(fL.listFilesExt(f"level2track")[0])

    lv2Dat = lv2DatA.sel(size_definition="Dmax", cameratrack="max", drop=True)
    lv2Dat["velocity_dist"] = lv2DatA["velocity_dist"].sel(
        size_definition="Dmax", cameratrack="mean", dim3D="z", drop=True
    )

    lv2Dat = xr.merge((lv2Dat, meteoDat, radarDat))

    coldEnough = lv2Dat.air_temperature < 272.15
    isPrecip = lv2Dat[config.level3.combinedRiming.Zvar] >= -10
    goodQuality = lv2Dat.qualityFlags == 0
    enoughParticles = lv2Dat.nParticles >= 100

    if np.all(~coldEnough | ~isPrecip | ~goodQuality | ~enoughParticles):
        with tools.open2("%s.nodata" % lv3File, "w") as f:
            f.write("no snowfall for %s" % case)
        log.warning("no snowfall for %s" % case)
        log.warning("written: %s.nodata" % lv3File)
        return None, lv3File

    goodData = coldEnough & isPrecip & goodQuality & enoughParticles
    lv3Dat = lv2Dat[
        [
            "Ze_0",
            "MDV_0",
            "Ze_ground",
            "MDV_ground",
            "Ze_std",
            "MDV_std",
            "air_temperature",
            "air_pressure",
        ]
    ]
    lv3Dat = lv3Dat.sel(time=lv2Dat.time)

    #### do retrieval ####
    x_a = np.float64(-1.0)  # a priori
    S_a = np.array([1.0**2])  # a priori uncertainty

    S_y = np.array([1.5**2])  # ([0.5**2]) # measurement uncertainty

    x_vars = ["M"]
    y_vars = ["Ze"]

    Dbound = np.append(
        lv2Dat.D_bins_left.mean("time").values,
        lv2Dat.D_bins_right.mean("time").values[-1],
    )
    Dmean = lv2Dat.D_bins.values
    psd = np.ma.masked_invalid(lv2Dat.PSD.values).filled(0.0)
    nBins = np.shape(psd)[1]

    good = 0
    bad = 0

    M_oe = np.empty(lv2Dat.time.size) * np.nan
    Ze_combinedRetrieval = np.empty(lv2Dat.time.size) * np.nan
    M_err = np.empty(lv2Dat.time.size) * np.nan

    for j in np.where(goodData)[0]:
        y_obs = lv2Dat[config.level3.combinedRiming.Zvar].isel(time=j).values

        print(j, y_obs)

        psd_data = {
            "ice": psd[j],
            "alt": 5,
            "temp": lv2Dat.air_temperature.values[j],
            "bins": nBins,
            "dmean": Dmean,
            "dbound": Dbound,
            "shape": config.level3.combinedRiming.habit,
            "frequency": frequency,
            "elevation": config.level3.combinedRiming.radarElevation,
        }

        # try:
        # print(x_vars, x_a, S_a, y_vars, y_obs, S_y)
        oe = pyOE.optimalEstimation(
            x_vars,  # state variable names
            x_a,  # a priori
            S_a,  # a priori uncertainty
            y_vars,  # measurement variable names
            y_obs,  # observations
            S_y,  # observation uncertainty
            reflec_logM,  # forward Operator
            forwardKwArgs=psd_data,  # additonal function arguments
        )

        oe.doRetrieval(maxIter=10, maxTime=1.0)

        # how many successes
        good += 1

        try:
            # x
            M_oe[j] = oe.x_op.iloc[0]
            # y
            Ze_combinedRetrieval[j] = oe.y_op.iloc[0]
            # errors
            M_err[j] = oe.S_op.iloc[0]
        except AttributeError:
            bad += 1

            M_oe[j] = np.nan
            Ze_combinedRetrieval[j] = np.nan
            M_err[j] = np.nan

        # except FileNotFoundError:
        #     # how many failures
        #     bad += 1

        #     M_oe[j] = np.nan
        #     Ze_combinedRetrieval[j] = np.nan
        #     M_err[j] = np.nan

    Mlog = xr.DataArray(M_oe, coords=[lv2Dat.time])
    M_err = xr.DataArray(M_err, coords=[lv2Dat.time])
    lv3Dat["Ze_combinedRetrieval"] = xr.DataArray(
        Ze_combinedRetrieval, coords=[lv2Dat.time]
    )
    lv3Dat["combinedNormalizedRimeMass"] = 10**Mlog

    ### derive microphysical parameters
    a, b = pyPamtra.descriptorFile.riming_dependent_mass_size(
        lv3Dat["combinedNormalizedRimeMass"], config.level3.combinedRiming.habit
    )

    lv3Dat["massSizeA"] = ("time", a)
    lv3Dat["massSizeB"] = ("time", b)

    deltaD = lv2Dat.D_bins_right - lv2Dat.D_bins_left

    lv3Dat["IWC"] = (
        lv2Dat.PSD * lv3Dat.massSizeA * (lv2Dat.D_bins) ** lv3Dat.massSizeB * deltaD
    ).sum("D_bins")

    lv3Dat["velocity_dist_heymsfield10"] = heymsfield10_particles_M(
        lv2Dat.D_bins,
        lv3Dat.combinedNormalizedRimeMass,
        lv2Dat.air_temperature,
        lv2Dat.air_pressure,
        config.level3.combinedRiming.habit,
    )

    velocity_dist = lv2Dat.velocity_dist.where(
        lv2Dat.velocity_dist > 0
    )  # negative werte rausschmeißen
    velocity_dist = velocity_dist.interpolate_na(
        dim="D_bins", method="nearest", fill_value="extrapolate"
    )

    lv3Dat["SR_M_dist"] = (
        velocity_dist
        * lv2Dat.PSD
        * lv3Dat.massSizeA
        * (lv2Dat.D_bins) ** lv3Dat.massSizeB  # .max('D_bins')
        * deltaD
    )  # kg/m2/s
    lv3Dat["SR_M"] = lv3Dat["SR_M_dist"].sum("D_bins") * 3600  # mm/h

    lv3Dat["SR_M_heymsfield10_dist"] = (
        lv3Dat["velocity_dist_heymsfield10"]
        * lv2Dat.PSD
        * lv3Dat.massSizeA
        * (lv2Dat.D_bins) ** lv3Dat.massSizeB
        * deltaD
    )
    lv3Dat["SR_M_heymsfield10"] = (
        lv3Dat["SR_M_heymsfield10_dist"].sum("D_bins") * 3600
    )  # mm/h

    lv3Dat.combinedNormalizedRimeMass.attrs.update(
        dict(
            units="-",
            long_name="normalized rime mass distribution (based on combined retrieval)",
        )
    )
    lv3Dat.Ze_combinedRetrieval.attrs.update(
        dict(
            units="dBz",
            long_name="retrieved radar reflectivity",
        )
    )
    lv3Dat.Ze_0.attrs.update(
        dict(
            units="dBz",
            long_name="measured radar reflectivity at lowest altitude",
        )
    )
    a, b = config.aux.radar.heightIndices
    lv3Dat.Ze_ground.attrs.update(
        dict(
            units="dBz",
            long_name=f"measured radar reflectivity extrapolated to 0 m AGL using range gates {a} to {b}",
        )
    )
    lv3Dat.Ze_std.attrs.update(
        dict(
            units="dBz",
            long_name=f"standard deviation of measured radar reflectivity using range gates {a} to {b}",
        )
    )
    lv3Dat.MDV_0.attrs.update(
        dict(
            units="m/s",
            long_name="measured mean Doppler velocity at lowest altitude",
        )
    )
    lv3Dat.MDV_ground.attrs.update(
        dict(
            units="m/s",
            long_name=f"measured mean Doppler velocity extrapolated to 0 m AGL using range gates {a} to {b}",
        )
    )
    lv3Dat.MDV_std.attrs.update(
        dict(
            units="m/s",
            long_name=f"standard deviation of measured mean Doppler velocity using range gates {a} to {b}",
        )
    )
    lv3Dat.massSizeA.attrs.update(
        dict(
            units="SI",
            long_name="prefactor of the mass size relation",
        )
    )
    lv3Dat.massSizeB.attrs.update(
        dict(
            units="SI",
            long_name="exponent of the mass size relation",
        )
    )
    lv3Dat.IWC.attrs.update(
        dict(
            units="kg/m^3",
            long_name="ice wtaer content",
        )
    )
    lv3Dat.SR_M_dist.attrs.update(
        dict(
            units="kg/m^2/s",
            long_name="spectral snowfall rate using observed fall velocity",
        )
    )
    lv3Dat.SR_M_heymsfield10_dist.attrs.update(
        dict(
            units="kg/m^2/s",
            long_name="spectral snowfall rate using Heymsfield10 fall velocity",
        )
    )
    lv3Dat.SR_M.attrs.update(
        dict(
            units="mm/h water equivalent",
            long_name="snowfall rate using observed fall velocity",
        )
    )
    lv3Dat.SR_M_heymsfield10.attrs.update(
        dict(
            units="mm/h water equivalent",
            long_name="snowfall rate using Heymsfield10 fall velocity",
        )
    )

    if writeNc:
        tools.to_netcdf2(lv3Dat, lv3File)
    return lv3Dat, lv3File
