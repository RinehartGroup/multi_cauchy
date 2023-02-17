from dataclasses import dataclass
from lmfit import Parameters


@dataclass
class CauchyParams:
    m_s: float | tuple[
        float, float, float
    ] = 0.0  # either an inital value or (initial_value, min, max)
    h_c: float | tuple[float, float, float] = 0.0
    gamma: float | tuple[float, float, float] = 0.0

    def rewrite_as_tuples(self):
        if isinstance(self.m_s, float | int):
            self.m_s = (self.m_s, 0, 1)
        elif isinstance(self.m_s, tuple):
            if len(self.m_s) != 3:
                raise ValueError("m_s must be a tuple of length 3")
        if isinstance(self.h_c, float | int):
            self.h_c = (self.h_c, -700000, 700000)
        elif isinstance(self.h_c, tuple):
            if len(self.h_c) != 3:
                raise ValueError("h_c must be a tuple of length 3")
        if isinstance(self.gamma, float | int):
            self.gamma = (self.gamma, 0, 100000)


def build_params(cauchy_params: list[CauchyParams]) -> Parameters:
    params = Parameters()
    for i, cauchy_param in enumerate(cauchy_params):
        cauchy_param.rewrite_as_tuples()
        params.add(
            f"m_s_{i}",
            value=cauchy_param.m_s[0],
            min=cauchy_param.m_s[1],
            max=cauchy_param.m_s[2],
        )
        params.add(
            f"h_c_{i}",
            value=cauchy_param.h_c[0],
            min=cauchy_param.h_c[1],
            max=cauchy_param.h_c[2],
        )
        params.add(
            f"gamma_{i}",
            value=cauchy_param.gamma[0],
            min=cauchy_param.gamma[1],
            max=cauchy_param.gamma[2],
        )
    params.add("chi_pd", value=0, min=-1, max=1)
    return params
