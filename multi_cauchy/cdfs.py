from typing import TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd
from lmfit import Parameters, minimize, minimizer

from multi_cauchy.c_params import CauchyParams, build_params
from multi_cauchy.datafile_import import MvsHMeasurement

HVal = TypeVar("HVal", float, npt.NDArray)


def cauchy_cdf(h: HVal, params: dict[str, float] | Parameters) -> HVal:
    try:
        m = h * params["chi_pd"]
    except KeyError:
        m = 0
    m += (2 * params["m_s"] / np.pi) * np.arctan((h - params["h_c"]) / params["gamma"])
    return m  # type: ignore # this is because lmfit is not typed


def multi_cauchy_cdf(h: HVal, params: dict[str, float] | Parameters) -> HVal:
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
    return m  # type: ignore # this is because lmfit is not typed


def fit_multi_cauchy_cdf(
    measurement: MvsHMeasurement,
    cauchy_params: list[CauchyParams],
    sequence: str = "both",  # 'forward' or 'reverse' or 'both'
) -> minimizer.MinimizerResult:
    params = build_params(cauchy_params)

    def residual(params: Parameters, h: float, data: float) -> float:
        return multi_cauchy_cdf(h, params) - data

    seq_to_fit = pd.DataFrame()
    if sequence in ["forward", "both"] and measurement.forward is not None:
        seq_to_fit = pd.concat([seq_to_fit, measurement.forward])
    if sequence in ["reverse", "both"] and measurement.reverse is not None:
        rev = measurement.reverse_sequence(measurement.reverse)
        seq_to_fit = pd.concat([seq_to_fit, rev])

    return minimize(
        residual, params, args=(seq_to_fit["field"], seq_to_fit["normalized_moment"])
    )
