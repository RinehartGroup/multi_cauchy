from dataclasses import dataclass, field
from pathlib import Path
import csv
import pandas as pd
import numpy as np

from multi_cauchy.parsing_algos import (
    find_section_starts,
    find_sequence_starts,
    unique_values_w_rounding,
)


class MvsHFile:
    """Class to read in a csv file with the format of the MvsH data files."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.header = self._read_header()
        self.sample_info = self._find_sample_info()
        self.data = self._read_data()
        self.temperatures_w_index = self._find_temperatures()
        self.temperatures = [t[0] for t in self.temperatures_w_index]

    def _read_header(self) -> list[str]:
        self.header: list[str] = []
        with open(self.path, "r") as f:
            csv_reader = csv.reader(f)
            csv_reader.__next__()  # skip first line that just says "[Header]"
            for row in csv_reader:
                if "[Data]" in row:
                    break
                else:
                    self.header.append(row)
        return self.header

    def _find_sample_info(self) -> dict[str, float]:
        self.sample_info: dict[str | float] = {}
        for line in self.header:
            if line[0] == "INFO" and line[2] == "SAMPLE_MASS" and line[1]:
                self.sample_info["mass"] = float(line[1])
            elif line[0] == "INFO" and line[2] == "SAMPLE_VOLUME" and line[1]:
                self.sample_info["eicosane_mass"] = float(line[1])
            elif line[0] == "INFO" and line[2] == "SAMPLE_MOLECULAR_WEIGHT" and line[1]:
                self.sample_info["molecular_weight"] = float(line[1])
            elif line[0] == "INFO" and line[2] == "SAMPLE_SIZE" and line[1]:
                self.sample_info["diamagnetic_correction"] = float(line[1])
        if self.sample_info["mass"] and self.sample_info["molecular_weight"]:
            self.sample_info["mol"] = (
                self.sample_info["mass"] / 1000
            ) / self.sample_info["molecular_weight"]
        return self.sample_info

    def _read_data(self) -> pd.DataFrame:
        self.data = pd.read_csv(self.path, skiprows=len(self.header) + 2)
        self.data = self.data[self.data["Comment"].isna()].reset_index(drop=True)
        self._simplify_data()
        if self.sample_info["diamagnetic_correction"]:
            self._correct_diamagnetic()
        return self.data

    def _simplify_data(self) -> None:
        # determine if the file contains VSM or dc data
        if self.data["DC Moment Free Ctr (emu)"].notnull().values.sum() != 0:
            self.data["uncorrected_moment_emu"] = self.data["DC Moment Free Ctr (emu)"]
            self.data["uncorrected_moment_error"] = self.data[
                "DC Moment Err Free Ctr (emu)"
            ]
        elif self.data["Moment (emu)"].notnull().values.sum() != 0:
            self.data["uncorrected_moment_emu"] = self.data["Moment (emu)"]
            self.data["uncorrected_moment_error"] = self.data["M. Std. Err. (emu)"]
        # prune the dataframe and simplify column names
        old_data = self.data.copy()
        self.data = pd.DataFrame()
        names = {
            "time": "Time Stamp (sec)",
            "temperature": "Temperature (K)",
            "field": "Magnetic Field (Oe)",
            "uncorrected_moment_emu": "uncorrected_moment_emu",
            "uncorrected_moment_error": "uncorrected_moment_error",
        }
        for new_name, old_name in names.items():
            self.data[new_name] = old_data[old_name]

    def _correct_diamagnetic(self) -> None:
        self.data["moment_emu"] = (
            (self.data["uncorrected_moment_emu"] / self.data["field"])
            - self.sample_info["diamagnetic_correction"] * self.sample_info["mol"]
        ) * self.data["field"]
        self.data["moment"] = self.data["moment_emu"] / self.sample_info["mol"] / 5585

    def _find_temperatures(self) -> list[tuple[float, int]]:
        section_starts = find_section_starts(
            self.data["temperature"], fluctuation_tolerance=0.25
        )
        self.temperatures_w_index = []
        for i in section_starts:
            # round temperature to nearset 0.25 K
            temperature = round(self.data["temperature"][i] * 4) / 4
            self.temperatures_w_index.append((temperature, i))
        return self.temperatures_w_index

    def set_sample_info(self, sample_info: dict[str | float]) -> None:
        self.sample_info = sample_info
        if self.sample_info["mass"] and self.sample_info["molecular_weight"]:
            self.sample_info["mol"] = (
                self.sample_info["mass"] / 1000
            ) / self.sample_info["molecular_weight"]
        self._read_data()


class MvsHMeasurement:
    """Contains data and metadata for a single MvsH measurement."""

    def __init__(self, file: MvsHFile, temperature_index: int):
        self.file = file
        self.temperature = self.file.temperatures[temperature_index]
        self.data = self._find_data(temperature_index)
        self.virgin, self.reverse, self.forward = self._set_sequences()
        if isinstance(self.reverse, pd.DataFrame) and self.forward is None:
            self.forward = self._reverse_to_forward()
            self.reverse = None

    def _find_data(self, temperature_index: int) -> pd.DataFrame:
        start = self.file.temperatures_w_index[temperature_index][1]
        try:
            end = self.file.temperatures_w_index[temperature_index + 1][1]
            df = self.file.data[start:end].copy().reset_index(drop=True)
        except IndexError:
            df = self.file.data[start:].copy().reset_index(drop=True)
        return df

    def _set_sequences(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        sequence_starts = find_sequence_starts(self.data["field"])
        virgin, reverse, forward = None, None, None
        if len(sequence_starts) == 3:
            virgin = (
                self.data[sequence_starts[0] : sequence_starts[1]]
                .copy()
                .reset_index(drop=True)
            )
            reverse = (
                self.data[sequence_starts[1] : sequence_starts[2]]
                .copy()
                .reset_index(drop=True)
            )
            forward = self.data[sequence_starts[2] :].copy().reset_index(drop=True)
        elif len(sequence_starts) == 2:
            # check to see if forward or reverse is first
            if (
                self.data.at[sequence_starts[0], "field"]
                > self.data.at[sequence_starts[1], "field"]
            ):
                reverse = (
                    self.data[sequence_starts[0] : sequence_starts[1]]
                    .copy()
                    .reset_index(drop=True)
                )
                forward = self.data[sequence_starts[1] :].copy().reset_index(drop=True)
            else:
                forward = (
                    self.data[sequence_starts[0] : sequence_starts[1]]
                    .copy()
                    .reset_index(drop=True)
                )
                reverse = self.data[sequence_starts[1] :].copy().reset_index(drop=True)
        elif len(sequence_starts) == 1:
            # check to see if forward or reverse is first
            if self.data.at[sequence_starts[0], "field"] > 0:
                forward = self.data[sequence_starts[0] :].copy().reset_index(drop=True)
            else:
                reverse = self.data[sequence_starts[0] :].copy().reset_index(drop=True)
        return virgin, reverse, forward

    def _reverse_to_forward(self) -> pd.DataFrame:
        """Converts a reverse sequence to a forward sequence."""
        df = self.reverse.copy()
        columns = [
            "field",
            "uncorrected_moment_emu",
            "uncorrected_moment_error",
            "moment_emu",
            "moment",
        ]
        for column in columns:
            df[column] = df[column] * -1
        return df
