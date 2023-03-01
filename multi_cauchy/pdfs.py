from dataclasses import dataclass, field
from typing import TypeVar

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from lmfit import Parameters, minimize, minimizer
from scipy.signal import find_peaks

from multi_cauchy.c_params import CauchyParams, build_params
from multi_cauchy.datafile_import import MvsHMeasurement

HVal = TypeVar("HVal", float, npt.NDArray)


def cauchy_pdf(h: HVal, params: dict[str, float] | Parameters) -> HVal:
    try:
        m = params["chi_pd"]
    except KeyError:
        m = 0
    m += (8 * params["m_s"] * params["gamma"]) / (
        np.pi * (16 * (h - params["h_c"]) ** 2 + params["gamma"] ** 2)
    )
    return m  # type: ignore # this is because lmfit is not typed


def multi_cauchy_pdf(h: HVal, params: dict[str, float] | Parameters) -> HVal:
    # `params` contains
    #   [
    #       m_s_0, h_c_0, gamma_0,
    #       m_s_1, h_c_1, gamma_1,
    #       ...
    #       chi_pd
    #   ]
    try:
        m = params["chi_pd"]
    except KeyError:
        m = 0
    for i in range(0, len(params) - 1, 3):
        j = i // 3
        m += (8 * params[f"m_s_{j}"] * params[f"gamma_{j}"]) / (
            np.pi * (16 * (h - params[f"h_c_{j}"]) ** 2 + params[f"gamma_{j}"] ** 2)
        )
    return m  # type: ignore # this is because lmfit is not typed


def fit_multi_cauchy_pdf(
    measurement: MvsHMeasurement,
    cauchy_params: list[CauchyParams],
    sequence: str = "both",  # 'forward' or 'reverse' or 'both'
) -> minimizer.MinimizerResult:
    params = build_params(cauchy_params)

    def residual(params: Parameters, h: float, dmdh_data: float) -> float:
        return multi_cauchy_pdf(h, params) - dmdh_data

    seq_to_fit = pd.DataFrame()
    if sequence in ["forward", "both"] and measurement.forward is not None:
        # exclude edges which may have derivative artifacts
        seq_to_fit = pd.concat([seq_to_fit, measurement.forward[2:-2]])
    if sequence in ["reverse", "both"] and measurement.reverse is not None:
        rev = measurement.reverse_sequence(measurement.reverse[2:-2])
        seq_to_fit = pd.concat([seq_to_fit, rev])

    return minimize(
        residual,
        params,
        args=(
            seq_to_fit["field"],
            seq_to_fit["dmdh"],
        ),
    )


def quick_find_peaks(
    measurement: MvsHMeasurement,
    deriv_method: str = "diff",  # 'diff' or 'gradient'
    smoothing_window_type: str = "boxcar",  # 'boxcar' or 'triang'
    smoothing_window_size: int = 3,
    smoothing_window_fn: str = "median",  # 'mean' or 'median'
    find_peaks_args: dict = {},
) -> tuple[list[float], plt.Figure, plt.Axes]:  # type: ignore
    df = pd.DataFrame()
    if measurement.forward is not None:
        df["x"] = measurement.forward["field"]
        df["y"] = measurement.forward["normalized_moment"]
    elif measurement.reverse is not None:
        df["x"] = measurement.reverse["field"]
        df["y"] = measurement.reverse["normalized_moment"]
    else:
        raise ValueError(
            "Measurement must contain at least one sequence (forward or reverse) to find peaks"
        )

    if deriv_method == "diff":
        df["dydx"] = df["y"].diff() / df["x"].diff()
    elif deriv_method == "gradient":
        df["dydx"] = np.gradient(df["y"], df["x"])
    else:
        raise ValueError("`deriv_method` only supports 'diff' or 'gradient'")

    if smoothing_window_type == "boxcar":
        if smoothing_window_fn == "mean":
            df["dydx_smooth"] = (
                df["dydx"].rolling(window=smoothing_window_size, center=True).mean()
            )
        elif smoothing_window_fn == "median":
            df["dydx_smooth"] = (
                df["dydx"].rolling(window=smoothing_window_size, center=True).median()
            )
        else:
            raise ValueError("`smoothing_window_fn` only supports 'mean' or 'median'")
    elif smoothing_window_type == "triang":
        if smoothing_window_fn == "mean":
            df["dydx_smooth"] = (
                df["dydx"]
                .rolling(
                    window=smoothing_window_size,
                    center=True,
                    win_type="triang",
                )
                .mean()
            )
        elif smoothing_window_fn == "median":
            raise ValueError(
                "`smoothing_window_fn` does not support 'median' with `smoothing_window_type` 'triang'"
            )
        else:
            raise ValueError("`smoothing_window_fn` only supports 'mean' or 'median'")

    find_peaks_defaults = {"height": df["dydx_smooth"].max() * 0.2, "width": 2}
    find_peaks_defaults.update(find_peaks_args)
    peaks, _ = find_peaks(df["dydx_smooth"], **find_peaks_defaults)
    peaks_in_data = df["x"][peaks].tolist()

    fig, ax = plt.subplots()
    ax.plot(df["x"], df["dydx"], label="dydx", color="black")
    ax.plot(df["x"], df["dydx_smooth"], label="dydx_smooth", color="red")
    for peak in peaks:
        ax.axvline(df.at[peak, "x"], color="blue")
    ax.legend()

    ax.set_ylim(-df["dydx"][10:-10].max() * 0.1, df["dydx"][10:-10].max() * 1.1)

    return peaks_in_data, fig, ax
