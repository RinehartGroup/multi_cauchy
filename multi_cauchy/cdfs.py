from lmfit import Parameters, minimize, minimizer
import numpy as np
import pandas as pd

from multi_cauchy.c_params import CauchyParams, build_params


def cauchy_cdf(h: float, params: dict[str, float] | Parameters) -> float:
    try:
        m = h * params["chi_pd"]
    except KeyError:
        m = 0
    m += (2 * params["m_s"] / np.pi) * np.arctan((h - params["h_c"]) / params["gamma"])
    return m


def multi_cauchy_cdf(h: float, params: dict[str, float] | Parameters) -> float:
    # `params` contains
    #   [
    #       m_s_0, h_c_0, gamma_0,
    #       m_s_1, h_c_1, gamma_1,
    #       ...
    #       chi_pd
    #   ]
    try:
        m = h * params["chi_pd"]
    except KeyError:
        m = 0
    for i in range(0, len(params) - 1, 3):
        j = i // 3
        m += (2 * params[f"m_s_{j}"] / np.pi) * np.arctan(
            (h - params[f"h_c_{j}"]) / params[f"gamma_{j}"]
        )
    return m


def fit_multi_cauchy_cdf(
    h: pd.Series,
    data: pd.Series,
    cauchy_params: list[CauchyParams],
) -> minimizer.MinimizerResult:
    params = build_params(cauchy_params)

    def residual(params: Parameters, h: float, data: float) -> float:
        return multi_cauchy_cdf(h, params) - data

    return minimize(residual, params, args=(h, data))
