# TODO: plot normalized residuals


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit import Parameters, minimizer

from multi_cauchy.cdfs import cauchy_cdf, multi_cauchy_cdf
from multi_cauchy.datafile_import import MvsHFile, MvsHMeasurement
from multi_cauchy.pdfs import cauchy_pdf, multi_cauchy_pdf


def plot_mvsh(
    file: MvsHFile,
) -> tuple[plt.Figure, plt.Axes]:  # type: ignore
    fig, ax = plt.subplots()
    for i, temp in enumerate(file.temperatures):
        try:
            meas = MvsHMeasurement(file, i)
        except ValueError:
            print(f"Unable to load measurement at {temp} K")
            continue
        ax.plot(meas.data["field"], meas.data["normalized_moment"], label=f"{temp} K")
    ax.set_xlabel("Field (Oe)")
    ax.set_ylabel("Normalized Moment")
    ax.legend()
    return fig, ax


def plot_cdf_fit(
    measurement: MvsHMeasurement,
    fit_result: minimizer.MinimizerResult,
    sequence: str = "both",  # "forward" or "reverse" or "both"
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:  # type: ignore
    _ensure_sequence_exists(measurement, sequence)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, height_ratios=[5, 1])
    fig.subplots_adjust(hspace=0)
    axs = (ax1, ax2)
    h, m_data = _plot_data(ax1, "cdf", measurement, sequence)
    _plot_fit(ax1, "cdf", fit_result, h)
    max_h_c = _plot_fit_terms(ax1, "cdf", fit_result, h)
    _add_residual(ax2, "cdf", measurement, fit_result, sequence, max_h_c)
    _set_ax_lims(ax1, "cdf", h, max_h_c, m_data)
    ax1.legend()
    return fig, axs


def plot_pdf_fit(
    measurement: MvsHMeasurement,
    fit_result: minimizer.MinimizerResult,
    sequence: str = "both",  # "forward" or "reverse" or "both"",
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:  # type: ignore
    _ensure_sequence_exists(measurement, sequence)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, height_ratios=[5, 1])
    fig.subplots_adjust(hspace=0)
    axs = (ax1, ax2)
    h, m_data = _plot_data(ax1, "pdf", measurement, sequence)
    _plot_fit(ax1, "pdf", fit_result, h)
    max_h_c = _plot_fit_terms(ax1, "pdf", fit_result, h)
    _add_residual(ax2, "pdf", measurement, fit_result, sequence, max_h_c)
    _set_ax_lims(ax1, "pdf", h, max_h_c, m_data)
    ax1.legend()
    return fig, axs


def _ensure_sequence_exists(measurement: MvsHMeasurement, sequence: str) -> None:
    if sequence not in ["forward", "reverse", "both"]:
        raise ValueError(
            f"Invalid sequence: {sequence}. Must be 'forward', 'reverse', or 'both'."
        )
    if sequence in ["forward", "both"] and measurement.forward is None:
        raise ValueError("Forward sequence not found in measurement")
    if sequence in ["reverse", "both"] and measurement.reverse is None:
        raise ValueError("Reverse sequence not found in measurement")


def _plot_data(
    ax: plt.Axes, plot_type: str, measurement: MvsHMeasurement, sequence: str
) -> tuple[np.ndarray, pd.Series]:
    m_data_col = {
        "cdf": "normalized_moment",
        "pdf": "dmdh",
    }[plot_type]
    h_range = (-70000, 70000)
    m_data = pd.Series(dtype=np.float64)
    if sequence in ["forward", "both"] and measurement.forward is not None:
        h_range = (
            measurement.forward["field"].min(),
            measurement.forward["field"].max(),
        )
        m_data = measurement.forward[m_data_col]
        ax.plot(
            measurement.forward["field"],
            m_data,
            label="Forward Data",
            color="red",
        )
    if sequence in ["reverse", "both"] and measurement.reverse is not None:
        h_range = (
            measurement.reverse["field"].min(),
            measurement.reverse["field"].max(),
        )
        h_data = -measurement.reverse["field"]
        m_data = (
            -measurement.reverse[m_data_col]
            if plot_type == "cdf"
            else measurement.reverse[m_data_col]
        )
        label = (
            "Reverse Data (Rotated)"
            if plot_type == "cdf"
            else "Reverse Data (Mirrored)"
        )
        ax.plot(
            h_data,
            m_data,
            label=label,
            color="blue",
        )
    return np.linspace(h_range[0], h_range[1], 1000), m_data


def _plot_fit(
    ax: plt.Axes, plot_type: str, fit_result: minimizer.MinimizerResult, h: np.ndarray
):
    model = {
        "cdf": multi_cauchy_cdf,
        "pdf": multi_cauchy_pdf,
    }[plot_type]
    fit_m = model(h, fit_result.params)
    ax.plot(h, fit_m, label="Fit", color="black")


def _plot_fit_terms(
    ax: plt.Axes, plot_type: str, fit_result: minimizer.MinimizerResult, h: np.ndarray
) -> float:
    model = {
        "cdf": cauchy_cdf,
        "pdf": cauchy_pdf,
    }[plot_type]
    max_h_c = 0
    for i in range(0, len(fit_result.params) - 1, 3):
        j = i // 3
        params = {}
        params["m_s"] = fit_result.params[f"m_s_{j}"]
        params["h_c"] = fit_result.params[f"h_c_{j}"]
        params["gamma"] = fit_result.params[f"gamma_{j}"]
        term_m = model(h, params)
        ax.plot(h, term_m, label=f"Term {j}")
        max_h_c = max(max_h_c, abs(params["h_c"]))
    return max_h_c


def _add_residual(
    ax: plt.Axes,
    plot_type: str,
    meas: MvsHMeasurement,
    fit_result: minimizer.MinimizerResult,
    sequence: str,
    max_h_c: float,
) -> None:
    y_data_col = {"cdf": "normalized_moment", "pdf": "dmdh"}[plot_type]
    model = {"cdf": multi_cauchy_cdf, "pdf": multi_cauchy_pdf}[plot_type]

    res_min_max = [0, 0]
    if sequence in ["forward", "both"] and meas.forward is not None:
        h_data = meas.forward["field"]
        y_data = meas.forward[y_data_col]
        y_fit = model(h_data, fit_result.params)
        y_residual = y_data - y_fit
        y_norm_residual = y_residual / y_data
        ax.fill_between(h_data, y_norm_residual, color="red", alpha=0.5)

        # get index where +/- max_h_c is closest to h_data
        start_idx = np.abs(h_data + max_h_c).argmin()
        end_idx = np.abs(h_data - max_h_c).argmin()
        if plot_type == "cdf":
            # find min/max residuals between +/- max_h_c and not in +/- 1000 Oe of h_c
            h_c = h_data[np.abs(y_data).argmin()]
            pre_h_c_idx = np.abs(h_data + (h_c - 1000)).argmin()
            post_h_c_idx = np.abs(h_data - (h_c + 1000)).argmin()
            min_fwd_res1 = np.min(y_norm_residual[start_idx:pre_h_c_idx])
            max_fwd_res1 = np.max(y_norm_residual[start_idx:pre_h_c_idx])
            min_fwd_res2 = np.min(y_norm_residual[post_h_c_idx:end_idx])
            max_fwd_res2 = np.max(y_norm_residual[post_h_c_idx:end_idx])
            res_min_max = [
                min(min_fwd_res1, min_fwd_res2, res_min_max[0]),
                max(max_fwd_res1, max_fwd_res2, res_min_max[1]),
            ]
        else:
            # get min and max residual between +/- h_c
            min_forward_res = np.min(y_norm_residual[start_idx:end_idx])
            max_forward_res = np.max(y_norm_residual[start_idx:end_idx])
            res_min_max = [
                min(min_forward_res, res_min_max[0]),
                max(max_forward_res, res_min_max[1]),
            ]
    if sequence in ["reverse", "both"] and meas.reverse is not None:
        h_data = -meas.reverse["field"]
        y_data = (
            -meas.reverse[y_data_col]
            if plot_type == "cdf"
            else meas.reverse[y_data_col]
        )
        y_fit = model(h_data, fit_result.params)
        y_residual = y_data - y_fit
        y_norm_residual = y_residual / y_data
        ax.fill_between(h_data, y_norm_residual, color="blue", alpha=0.5)

        # get index where +/- max_h_c is closest to h_data
        start_idx = np.abs(h_data + max_h_c).argmin()
        end_idx = np.abs(h_data - max_h_c).argmin()
        if plot_type == "cdf":
            # find min/max residuals between +/- max_h_c and not in +/- 1000 Oe of h_c
            h_c = h_data[np.abs(y_data).argmin()]
            pre_h_c_idx = np.abs(h_data + (h_c - 1000)).argmin()
            post_h_c_idx = np.abs(h_data - (h_c + 1000)).argmin()
            min_reverse_res1 = np.min(y_norm_residual[start_idx:pre_h_c_idx])
            max_reverse_res1 = np.max(y_norm_residual[start_idx:pre_h_c_idx])
            min_reverse_res2 = np.min(y_norm_residual[post_h_c_idx:end_idx])
            max_reverse_res2 = np.max(y_norm_residual[post_h_c_idx:end_idx])
            res_min_max = [
                min(min_reverse_res1, min_reverse_res2, res_min_max[0]),
                max(max_reverse_res1, max_reverse_res2, res_min_max[1]),
            ]
        else:
            # get min and max residual between +/- h_c
            min_reverse_res = np.min(y_norm_residual[start_idx:end_idx])
            max_reverse_res = np.max(y_norm_residual[start_idx:end_idx])
            res_min_max = [
                min(min_reverse_res, res_min_max[0]),
                max(max_reverse_res, res_min_max[1]),
            ]

    ax.axhline(0, color="black", linestyle="dashed")

    y_min = np.floor(res_min_max[0] * 10) / 10
    y_max = np.ceil(res_min_max[1] * 10) / 10
    ax.set_ylim(y_min, y_max)


def _set_ax_lims(
    ax: plt.Axes, plot_type: str, h: np.ndarray, max_h_c: float, m_data: pd.Series
) -> None:
    if (h.max() - h.min()) > 50:
        window = 5000 * round(max_h_c / 5000) + 10000
        ax.set_xlim(-window, window)
    else:
        window = 0.5 * round(max_h_c / 0.5) + 1
        ax.set_xlim(-window, window)

    if plot_type == "cdf":
        # m_data_max = max(abs(m_data.min()), abs(m_data.max()))
        # ax.set_ylim(-m_data_max, m_data_max)
        ax.set_ylim(-1, 1)
    elif plot_type == "pdf":
        ax.set_ylim(-m_data[10:-10].max() * 0.1, m_data[10:-10].max() * 1.1)
