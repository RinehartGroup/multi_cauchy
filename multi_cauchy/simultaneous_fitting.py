import pandas as pd
from lmfit import Parameters, minimize, minimizer
from multi_cauchy.c_params import CauchyParams, build_params
from multi_cauchy.cdfs import multi_cauchy_cdf
from multi_cauchy.pdfs import multi_cauchy_pdf


def fit_multi_cauchy_cdf_and_pdf(
    h: pd.Series,
    m_data: pd.Series,
    dmdh_data: pd.Series,
    cauchy_params: list[CauchyParams],
) -> minimizer.MinimizerResult:
    params = build_params(cauchy_params)

    m_data = m_data / m_data.max()
    dmdh_data = dmdh_data / dmdh_data.max()

    def residual(
        params: Parameters, h: float, m_data: float, dmdh_data: float
    ) -> float:
        return (
            multi_cauchy_cdf(h, params)
            - m_data
            + multi_cauchy_pdf(h, params)
            - dmdh_data
        )

    return minimize(residual, params, args=(h, m_data, dmdh_data))
