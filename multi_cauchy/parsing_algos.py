import pandas as pd
import numpy as np

def unique_values_w_rounding(x: list, tolerance) -> list[float]:
    """Returns a list of unique values in `x` where the values are considered equal
    if they are within `tolerance` of each other.

    Examples:
    ```
    >>>_unique_values_w_rounding([5, 5.1, 4.9, 300.1, 299.6, 300.2], 1)
    [5, 300]
    ```
    """
    return list({round(i / tolerance) * tolerance for i in x})

def find_section_starts(x: pd.Series, fluctuation_tolerance: float = 0) -> list[int]:
    """Find the indices of the start of each section in a series of data,
    where a section is defined as a series of identical numbers (within the `fluctuation_tolerance`).

    Example:
    ```
    >>>x = pd.Series([1, 1, 1, 1, 2, 2, 2, 2])
    >>>_find_sequence_starts(x)
    [0, 4]

    >>>y = pd.Series([1, 1.01, 0.95, 1.07, 1.03, 2.01, 2.01, 2.02, 1.92])
    >>>_find_sequence_starts(y, 0.5)
    [0, 5]
    ```
    """
    values = unique_values_w_rounding(x, fluctuation_tolerance)
    df = pd.DataFrame({"x": x})
    for value in values:
        df[f"{value}"] = df["x"].mask(
            abs(df["x"] - value) < fluctuation_tolerance, value
        )
        df[f"{value}"] = df[f"{value}"].mask(
            abs(df["x"] - value) > fluctuation_tolerance, 0
        )
    df.drop(columns=["x"], inplace=True)
    df["y"] = sum([df[f"{value}"] for value in values])
    section_starts = [0]
    for i in df.index[1:]:
        if df["y"][i] != df["y"][i - 1]:
            section_starts.append(i)
    return section_starts

def find_sequence_starts(x: pd.Series, flucuation_tolerance: float = 0) -> list[int]:
    """Find the indices of the start of each sequence in a series of data,
    where a sequences is defined as a series of numbers that constantly increase or decrease.
    Changes below `fluctuation_tolerance` are ignored.

    Example:
    ```
    >>>x = pd.Series([0, 1, 2, 3, 4, 3, 2, 1])
    >>>_find_sequence_starts(x)
    [0, 5]

    >>>y = pd.Series([0, 1, 2, 3, 0, 1, 2, 3])
    >>>_find_sequence_starts(y)
    [0, 4]
    ```
    """
    df = pd.DataFrame({"x": x, "diff": x.diff()})
    df["direction"] = np.where(df["diff"] > 0, 1, -1)
    start: int = df.index.start
    df.at[start, "direction"] = df.at[
        start + 1, "direction"
    ]  # since the first value of diff is NaN
    # if there's a really small diff value with the opposite sign of diff values around it, it's probably a mistake
    sequence_starts = [0]
    for i in df[start + 2 :].index:
        last2_dir = df.at[i - 2, "direction"]
        last1_dir = df.at[i - 1, "direction"]
        current_dir = df.at[i, "direction"]
        try:
            next1_dir = df.at[i + 1, "direction"]
            next2_dir = df.at[i + 2, "direction"]
            next3_dir = df.at[i + 3, "direction"]
        except KeyError:
            # reached end of dataframe
            break

        # below handles, for example, zfc from 5 to 300 K, drop temp to 5 K, fc from 5 to 300 K
        if (current_dir != last1_dir) and (current_dir != next1_dir):
            if abs(df.at[i, "diff"]) < flucuation_tolerance:
                # this is a fluctuation and should be ignored
                df.at[i, "direction"] = last1_dir
                current_dir = last1_dir
            else:
                sequence_starts.append(i)

        # below handles, for example, zfc from 5 to 300 K, fc from 300 to 5 K
        # assumes there won't be any fluctuations at the beginning of a sequence
        if (
            (last2_dir == last1_dir)
            and (current_dir != last1_dir)
            and (current_dir == next1_dir == next2_dir == next3_dir)
        ):
            sequence_starts.append(i)

    return sequence_starts