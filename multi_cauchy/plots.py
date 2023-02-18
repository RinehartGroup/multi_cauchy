import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit import Parameters, minimizer

from multi_cauchy.cdfs import cauchy_cdf, multi_cauchy_cdf
from multi_cauchy.pdfs import cauchy_pdf, multi_cauchy_pdf


def _plot_multi_cauchy_fit(
    cauchy_type: str,  # "cdf" or "pdf"
    h_data: pd.Series,
    m_data: pd.Series,
    fit_result: minimizer.MinimizerResult,
) -> tuple[plt.Figure, plt.Axes]:  # type: ignore
    fig, ax = plt.subplots()
    ax.plot(h_data, m_data, label="data", color="black")

    model = {
        "cdf": (multi_cauchy_cdf, cauchy_cdf),
        "pdf": (multi_cauchy_pdf, cauchy_pdf),
    }[cauchy_type]

    h = np.linspace(h_data.min(), h_data.max(), 1000)
    ax.plot(h, model[0](h, fit_result.params), label="fit", color="red")
    max_h_c = 0
    for i in range(0, len(fit_result.params) - 1, 3):
        j = i // 3
        params = {}
        params["m_s"] = fit_result.params[f"m_s_{j}"]
        params["h_c"] = fit_result.params[f"h_c_{j}"]
        params["gamma"] = fit_result.params[f"gamma_{j}"]
        ax.plot(h, model[1](h, params))

        max_h_c = max(max_h_c, abs(params["h_c"]))

    if (h_data.max() - h_data.min()) > 50:
        window = 5000 * round(max_h_c / 5000) + 10000
        ax.set_xlim(-window, window)
    else:
        window = 0.5 * round(max_h_c / 0.5)
        ax.set_xlim(-window, window)

    if cauchy_type == "cdf":
        m_data_max = max(abs(m_data.min()), abs(m_data.max()))
        ax.set_ylim(-m_data_max, m_data_max)
    elif cauchy_type == "pdf":
        ax.set_ylim(-m_data.max() * 0.1, m_data.max() * 1.1)

    ax.legend()

    return fig, ax


def plot_cdf_fit(
    h_data: pd.Series, m_data: pd.Series, fit_result: minimizer.MinimizerResult
) -> tuple[plt.Figure, plt.Axes]:
    return _plot_multi_cauchy_fit("cdf", h_data, m_data, fit_result)


def plot_pdf_fit(
    h_data: pd.Series, m_data: pd.Series, fit_result: minimizer.MinimizerResult
) -> tuple[plt.Figure, plt.Axes]:
    return _plot_multi_cauchy_fit("pdf", h_data, m_data, fit_result)
