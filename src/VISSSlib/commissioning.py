"""Helpers for bringing a new VISSS instrument into operation.

See :doc:`commissioning` for the step-by-step guide these functions belong to.
"""

import cv2
import numpy as np
from loguru import logger as log


def brightnessOfFile(fname, nFrames=20, heightOffset=0):
    """
    Mean gray-scale brightness of an image or a video.

    Parameters
    ----------
    fname : str
        Image (jpg, png, ...) or video (mkv, mov, ...) file.
    nFrames : int, optional
        Maximum number of frames averaged for a video (default 20).
    heightOffset : int, optional
        Number of rows at the top of each frame to ignore, e.g. the
        annotation strip written by the data acquisition (default 0; use
        ``config.level1detect.height_offset``).

    Returns
    -------
    float
        Mean pixel value (0-255) over all used frames.
    """
    cap = cv2.VideoCapture(str(fname))
    means = []
    while len(means) < nFrames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        means.append(frame[heightOffset:].mean())
    cap.release()
    if len(means) == 0:
        raise IOError(f"Could not read any frame from {fname}")
    return float(np.mean(means))


def estimateExposureDelay(exposure, brightness, flashDuration=None):
    """
    Estimate the delay between exposure start and the backlight flash.

    The backlight is triggered at the start of the leader's exposure and
    lights up after a fixed delay for a fixed duration. The camera only
    collects light while its exposure is open, so the brightness of a scene
    recorded with exposure time :math:`E` is

    .. math::
        B(E) = B_0 + g \\cdot \\mathrm{clip}(E - d, 0, w)

    with the dark level :math:`B_0`, the delay :math:`d`, the effective flash
    duration :math:`w` and the gain :math:`g`. The parameters are obtained by
    a grid search over :math:`d` and :math:`w`; :math:`B_0` and :math:`g`
    follow from linear least squares for each pair.

    Parameters
    ----------
    exposure : array_like
        Exposure times of the test recordings [microseconds]. At least 4 values;
        they should bracket the transition from dark to fully lit.
    brightness : array_like
        Mean brightness for each exposure time, see :func:`brightnessOfFile`.
    flashDuration : float, optional
        If the flash duration is known [microseconds], it is kept fixed instead
        of being estimated.

    Returns
    -------
    dict
        ``delay`` and ``duration`` [microseconds], ``dark`` and ``bright``
        (brightness before the flash and with the complete flash captured),
        ``rmse`` of the fit and ``fit`` (modelled brightness at ``exposure``).
        ``saturated`` is True if the brightest recording is close to 255; in
        this case the gain is underestimated and ``duration`` is not reliable.
    """
    E = np.asarray(exposure, dtype=float)
    B = np.asarray(brightness, dtype=float)
    if E.shape != B.shape or E.ndim != 1 or len(E) < 4:
        raise ValueError("need matching 1D exposure and brightness, at least 4 values")
    order = np.argsort(E)
    E, B = E[order], B[order]

    span = E[-1] - E[0]
    delays = np.linspace(E[0] - 0.1 * span, E[-1], 400)
    if flashDuration is None:
        durations = np.linspace(span / 200, 1.1 * span, 200)
    else:
        durations = [float(flashDuration)]

    best = None
    for d in delays:
        for w in durations:
            x = np.clip(E - d, 0, w)
            if x.max() == 0:
                continue
            A = np.vstack([np.ones_like(x), x]).T
            coef, *_ = np.linalg.lstsq(A, B, rcond=None)
            if coef[1] < 0:
                continue
            err = np.sqrt(np.mean((A @ coef - B) ** 2))
            if best is None or err < best[0] - 1e-12:
                best = (err, d, w, coef)
    if best is None:
        raise ValueError("brightness does not increase with exposure time")
    err, d, w, (b0, g) = best
    result = dict(
        delay=float(d),
        duration=float(w),
        dark=float(b0),
        bright=float(b0 + g * min(w, E[-1] - d)),
        rmse=float(err),
        fit=b0 + g * np.clip(E - d, 0, w),
        saturated=bool(B.max() > 250),
    )
    if result["saturated"]:
        log.warning(
            "Brightest recording is saturated, reduce lens aperture or "
            "backlight intensity and repeat."
        )
    return result


def plotExposureScan(exposure, brightness, result, fname=None):
    """
    Plot measured brightness versus exposure time with the fit of
    :func:`estimateExposureDelay`.

    Parameters
    ----------
    exposure, brightness : array_like
        As passed to :func:`estimateExposureDelay`.
    result : dict
        Return value of :func:`estimateExposureDelay`.
    fname : str, optional
        If given, the figure is saved here.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    E = np.asarray(exposure, dtype=float)
    order = np.argsort(E)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(E, brightness, "ko", label="measured")
    ax.plot(E[order], np.asarray(result["fit"])[order], "r-", label="fit")
    ax.axvline(result["delay"], color="b", ls=":")
    ax.set_xlabel("exposure time [µs]")
    ax.set_ylabel("mean brightness")
    ax.set_title(
        f"delay {result['delay']:.1f} µs, flash duration {result['duration']:.1f} µs"
    )
    ax.legend()
    fig.tight_layout()
    if fname is not None:
        fig.savefig(fname, dpi=150)
    return fig


def analyzeExposureScan(files, heightOffset=0, nFrames=20, flashDuration=None):
    """
    Estimate the exposure delay from recordings at different exposure times.

    Parameters
    ----------
    files : dict
        Maps exposure time [microseconds] to the recorded image or video
        file (or a list of files, which are averaged).
    heightOffset, nFrames
        See :func:`brightnessOfFile`.
    flashDuration : float, optional
        See :func:`estimateExposureDelay`.

    Returns
    -------
    dict
        Result of :func:`estimateExposureDelay` extended by ``exposure`` and
        ``brightness``.
    """
    exposure = np.array(sorted(files), dtype=float)
    brightness = []
    for e in sorted(files):
        fnames = files[e]
        if isinstance(fnames, (str, bytes)) or not hasattr(fnames, "__iter__"):
            fnames = [fnames]
        brightness.append(
            np.mean([brightnessOfFile(f, nFrames, heightOffset) for f in fnames])
        )
    brightness = np.array(brightness)
    result = estimateExposureDelay(exposure, brightness, flashDuration)
    result["exposure"] = exposure
    result["brightness"] = brightness
    return result
