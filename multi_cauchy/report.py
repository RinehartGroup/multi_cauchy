"""
In addition to recording the analysis results in a standardarized, easily retrievable
format, the report should contain sufficient information regarding the data processing
and fit input parameters to reproduce the results.

There may also need to be some light data verification. One item in particular is a
check that the sweep rate is constant.
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import asdict, dataclass, field
from lmfit.minimizer import MinimizerResult, Parameters

from multi_cauchy.c_params import CauchyParams
from multi_cauchy.datafile_import import MvsHFile, MvsHMeasurement


@dataclass
class ReportFit:
    sum_square_error: float
    reduced_sum_square_error: float
    output_params: Parameters


@dataclass
class ReportMeasurement:
    temperature_index: int
    temperature: float
    sweep_rate: float
    sweep_rate_std: float
    fit_input_params: list[CauchyParams]
    cdf_fit: ReportFit
    pdf_fit: ReportFit
    num_terms: int = field(init=False)

    def __post_init__(self):
        self.num_terms = len(self.fit_input_params)


@dataclass
class ReportFile:
    name: str
    date_created: str
    length: int
    sha512: str
    measurements: list[ReportMeasurement]


def write_report(
    file: MvsHFile,
    measurement: MvsHMeasurement,
    fit_input_params: list[CauchyParams],
    cdf_fit_results: MinimizerResult,
    pdf_fit_results: MinimizerResult,
    path: Path | str | None = None,
) -> None:
    report_cdf_fit = ReportFit(
        sum_square_error=cdf_fit_results.chisqr,
        reduced_sum_square_error=cdf_fit_results.redchi,
        # params is JSON serializable, but for some reason it serializes to an empyt
        # dict if we wait try to serialize the whole `report_file` object at the end
        # of the function. Instead we can turn it into a dict with `loads`/`dumps`
        output_params=json.loads(cdf_fit_results.params.dumps()),  # type: ignore
    )
    report_pdf_fit = ReportFit(
        sum_square_error=pdf_fit_results.chisqr,
        reduced_sum_square_error=pdf_fit_results.redchi,
        output_params=json.loads(pdf_fit_results.params.dumps()),  # type: ignore
    )

    report_measurement = ReportMeasurement(
        temperature_index=measurement.temperature_index,
        temperature=measurement.temperature,
        sweep_rate=measurement.sweep_rate,
        sweep_rate_std=measurement.sweep_rate_std,
        fit_input_params=fit_input_params,
        cdf_fit=report_cdf_fit,
        pdf_fit=report_pdf_fit,
    )

    path = Path(path) if path else file.path.parent
    path = path / f"multi_cauchy_{file.path.stem}.json"
    name = path.stem

    if path.exists():
        report_file = ingest_report(path)
        # need to rewrite the ingested fit params for the same reason as above
        for meas in report_file.measurements:
            meas.cdf_fit.output_params = json.loads(meas.cdf_fit.output_params.dumps())
            meas.pdf_fit.output_params = json.loads(meas.pdf_fit.output_params.dumps())
        # now check if the measurement already exists
        current_temperature = report_measurement.temperature
        current_num_terms = report_measurement.num_terms
        for meas in report_file.measurements:
            if (
                meas.temperature == current_temperature
                and meas.num_terms == current_num_terms
            ):
                response = input(
                    "A measurement with the same temperature and number of terms "
                    f"already exists in {name}. Do you want to (1) overwrite it, "
                    "(2) keep both, or (3) cancel this report? [1/2/3]"
                )
                if response == "1":
                    report_file.measurements.remove(meas)
                    break
                elif response == "2":
                    break
                elif response == "3":
                    return
        report_file.measurements.append(report_measurement)

    else:
        report_file = ReportFile(
            name=name,
            date_created=file.date_created,
            length=file.length,
            sha512=file.sha512,
            measurements=[report_measurement],
        )

    with open(path, "w") as f:
        json.dump(asdict(report_file), f, indent=4)


def ingest_report(path: Path | str) -> ReportFile:
    path = Path(path)
    with open(path, "r") as f:
        data = json.load(f)
    measurements = []
    for measurement in data["measurements"]:
        cdf_fit = ReportFit(
            measurement["cdf_fit"]["sum_square_error"],
            measurement["cdf_fit"]["reduced_sum_square_error"],
            Parameters().loads(json.dumps(measurement["cdf_fit"]["output_params"])),
        )
        pdf_fit = ReportFit(
            measurement["pdf_fit"]["sum_square_error"],
            measurement["pdf_fit"]["reduced_sum_square_error"],
            Parameters().loads(json.dumps(measurement["pdf_fit"]["output_params"])),
        )
        measurements.append(
            ReportMeasurement(
                measurement["temperature_index"],
                measurement["temperature"],
                measurement["sweep_rate"],
                measurement["sweep_rate_std"],
                [CauchyParams(**params) for params in measurement["fit_input_params"]],
                cdf_fit,
                pdf_fit,
            )
        )
    return ReportFile(
        data["name"],
        data["date_created"],
        data["length"],
        data["sha512"],
        measurements,
    )
