"""
data_processing.py
~~~~~~~~~~~~~~~~~~
Functions for loading, merging, and transforming ER visit data into
sliding-window feature matrices suitable for KDE-based anomaly detection.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Columns that contain date/time information (expanded as needed)
# ---------------------------------------------------------------------------
_DATE_COLUMNS = [
    "DATAORA_ACCETTAZIONE",
    "DATAORA_TRIAGE",
    "DATAORA_DIMISSIONE",
    "DATA_NASCITA",
]


def _coerce_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert any known date columns present in *df* to ``datetime64``."""
    for col in _DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
    return df


def merge_er_data(
    df_accessi: pd.DataFrame,
    df_clinici: pd.DataFrame,
    df_triage: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge the three core ER dataframes on the shared access key.

    Parameters
    ----------
    df_accessi:
        Master access registry.  Must contain ``ID_ACCESSO``.
    df_clinici:
        Clinical records keyed on ``ID_ACCESSO``.
    df_triage:
        Triage records keyed on ``ID_ACCESSO``.

    Returns
    -------
    pd.DataFrame
        Wide dataframe with one row per ER access, all columns from the
        three sources merged, and date columns cast to ``datetime64``.
    """
    if "ID_ACCESSO" not in df_accessi.columns:
        raise ValueError("df_accessi must contain an 'ID_ACCESSO' column.")

    df = (
        df_accessi
        .merge(df_clinici, on="ID_ACCESSO", how="left", suffixes=("", "_clin"))
        .merge(df_triage,  on="ID_ACCESSO", how="left", suffixes=("", "_triage"))
    )

    df = _coerce_dates(df)
    df.sort_values("DATAORA_ACCETTAZIONE", inplace=True, ignore_index=True)
    return df


def create_sliding_windows(
    df: pd.DataFrame,
    group_col: str = "ID_PAZIENTE",
    time_col: str = "DATAORA_ACCETTAZIONE",
    window_len: int = 3,
    numeric_only: bool = True,
) -> pd.DataFrame:
    """
    Build a flattened sliding-window feature matrix that captures the
    Markov property over consecutive ER visits.

    For each patient with at least *window_len* visits the function
    produces one row per window by concatenating the numeric feature
    vectors of the *window_len* most recent visits (t-k, …, t-1, t).

    Parameters
    ----------
    df:
        Merged ER dataframe returned by :func:`merge_er_data`.
    group_col:
        Column used to group visits by patient.
    time_col:
        Datetime column used to sort visits chronologically.
    window_len:
        Number of consecutive visits per window (≥ 2).
    numeric_only:
        When ``True`` only numeric columns (excluding the group/time keys)
        are included in the flattened feature vector.

    Returns
    -------
    pd.DataFrame
        Each row is a flattened window.  Column names are suffixed with
        ``_t``, ``_t-1``, … ``_t-(window_len-1)`` from newest to oldest.
        The columns ``ID_PAZIENTE`` and the anchor ``DATAORA_ACCETTAZIONE``
        of the last visit in each window are preserved for traceability.
    """
    if window_len < 2:
        raise ValueError("window_len must be >= 2.")

    if group_col not in df.columns:
        raise ValueError(f"Column '{group_col}' not found in dataframe.")

    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' not found in dataframe.")

    # Select feature columns (drop keys and dates to avoid dtype issues)
    exclude = {group_col, time_col, "ID_ACCESSO"}
    if numeric_only:
        feature_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude
        ]
    else:
        feature_cols = [c for c in df.columns if c not in exclude]

    rows: list[dict] = []

    for patient_id, group in df.groupby(group_col, sort=False):
        group = group.sort_values(time_col).reset_index(drop=True)

        if len(group) < window_len:
            continue  # not enough visits to form a window

        for end_idx in range(window_len - 1, len(group)):
            window = group.iloc[end_idx - window_len + 1 : end_idx + 1]
            row: dict = {
                group_col: patient_id,
                time_col: group.at[end_idx, time_col],
            }

            # Flatten: newest visit = _t, next = _t-1, …
            for offset, (_, visit) in enumerate(window.iloc[::-1].iterrows()):
                suffix = "_t" if offset == 0 else f"_t-{offset}"
                for col in feature_cols:
                    row[f"{col}{suffix}"] = visit[col]

            rows.append(row)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).reset_index(drop=True)
    return result
