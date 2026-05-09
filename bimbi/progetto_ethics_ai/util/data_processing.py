"""
data_processing.py
~~~~~~~~~~~~~~~~~~
Functions for loading, merging, and transforming ER visit data into
sliding-window feature matrices suitable for KDE-based anomaly detection.

Pipeline order
--------------
  1. merge_er_data()         -- join the three raw ER tables
  2. build_features()        -- engineer temporal + clinical features
  3. create_sliding_windows() -- flatten consecutive visits into windows
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Known date/time columns
# ---------------------------------------------------------------------------

_DATE_COLUMNS = [
    "DATAORA_ACCETTAZIONE",
    "DATAORA_TRIAGE",
    "DATAORA_DIMISSIONE",
    "DATA_NASCITA",
]


def _coerce_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Cast any recognised date columns present in *df* to ``datetime64``."""
    for col in _DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
    return df


# ---------------------------------------------------------------------------
# Step 1 – Merge
# ---------------------------------------------------------------------------

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
        Sorted ascending by ``DATAORA_ACCETTAZIONE``.
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
    logger.info("merge_er_data: %d rows, %d columns.", len(df), df.shape[1])
    return df


# ---------------------------------------------------------------------------
# Step 2 – Feature Engineering
# ---------------------------------------------------------------------------

def _rolling_prior_90d(group: pd.DataFrame) -> pd.Series:
    """
    Count ER visits in the 90 days BEFORE each visit (current visit excluded).

    Uses a time-indexed rolling window (O(n log n)) rather than a nested loop.
    The group must already be sorted by ``DATAORA_ACCETTAZIONE``.
    """
    group = group.sort_values("DATAORA_ACCETTAZIONE")
    dt_idx = pd.DatetimeIndex(group["DATAORA_ACCETTAZIONE"])

    # A Series of 1s with a DatetimeIndex is required for offset-based rolling.
    ones = pd.Series(1.0, index=dt_idx)

    # rolling("90D") includes the current observation → subtract 1 for prior only.
    counts = (
        ones
        .rolling("90D", min_periods=1)
        .sum()
        .sub(1.0)
        .clip(lower=0.0)
    )
    return pd.Series(counts.values, index=group.index, dtype=float)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer temporal and clinical features from the merged ER DataFrame.

    Must be called after :func:`merge_er_data` and before
    :func:`create_sliding_windows`.  Adds four derived columns that capture
    the longitudinal visit pattern used by the KDE anomaly detector:

    age_months
        Patient age in months at each visit, computed from ``DATA_NASCITA``
        and ``DATAORA_ACCETTAZIONE``.  Missing birth dates are imputed with
        the column median; ages < 0 (data entry errors) are clipped to 0.

    days_since_last_visit
        Integer days between consecutive visits for the same patient, sorted
        chronologically.  The **first visit** per patient receives the sentinel
        value ``3650`` (approx. 10 years) to represent "no prior visit" without
        introducing NaN into downstream numeric pipelines.

    num_visits_90d
        Count of ER visits by the same patient in the **90 days before** the
        current visit (the current visit is excluded).  Computed via a
        time-indexed rolling window so that it is O(n log n), not O(n^2).

    inter_visit_variance
        Rolling standard deviation (window=3, min_periods=1) of
        ``days_since_last_visit`` within each patient's history.  High and
        increasing variance signals irregular, escalating visit cadence — a
        known indicator of non-accidental injury patterns.  Undefined values
        (fewer than 2 data points in the window) are filled with ``0``.

    Parameters
    ----------
    df:
        Merged ER dataframe returned by :func:`merge_er_data`.  Must contain
        ``ID_PAZIENTE`` and ``DATAORA_ACCETTAZIONE``; ``DATA_NASCITA`` is
        optional but recommended.

    Returns
    -------
    pd.DataFrame
        Original dataframe with the four engineered columns appended, sorted
        by ``(ID_PAZIENTE, DATAORA_ACCETTAZIONE)``.

    Notes
    -----
    Rows where ``DATAORA_ACCETTAZIONE`` is NaT are dropped with a warning,
    as they cannot participate in any time-based computation.
    """
    df = df.copy()

    # Ensure datetime columns are correctly typed.
    for col in ("DATAORA_ACCETTAZIONE", "DATA_NASCITA"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

    # Drop rows with missing visit timestamp — cannot be used in time windows.
    n_before = len(df)
    df = df.dropna(subset=["DATAORA_ACCETTAZIONE"])
    if len(df) < n_before:
        logger.warning(
            "Dropped %d rows with NaT in DATAORA_ACCETTAZIONE.",
            n_before - len(df),
        )

    # ------------------------------------------------------------------
    # age_months
    # ------------------------------------------------------------------
    if "DATA_NASCITA" in df.columns:
        age_days = (df["DATAORA_ACCETTAZIONE"] - df["DATA_NASCITA"]).dt.days
        df["age_months"] = (age_days / 30.44).clip(lower=0.0)
        median_age = df["age_months"].median()
        n_missing = df["age_months"].isna().sum()
        if n_missing:
            logger.warning(
                "%d patients have missing DATA_NASCITA; imputing age_months "
                "with median (%.1f months).",
                n_missing,
                median_age if pd.notna(median_age) else 0.0,
            )
        df["age_months"] = df["age_months"].fillna(
            median_age if pd.notna(median_age) else 0.0
        )
    else:
        logger.warning("DATA_NASCITA not found; age_months set to 0.0 for all rows.")
        df["age_months"] = 0.0

    # Sort chronologically per patient before computing sequential features.
    df = df.sort_values(["ID_PAZIENTE", "DATAORA_ACCETTAZIONE"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # days_since_last_visit
    # ------------------------------------------------------------------
    df["days_since_last_visit"] = (
        df.groupby("ID_PAZIENTE")["DATAORA_ACCETTAZIONE"]
        .diff()
        .dt.days
        .fillna(3650.0)   # first visit sentinel: ~10 years
        .clip(lower=0.0)
    )

    # ------------------------------------------------------------------
    # num_visits_90d
    # ------------------------------------------------------------------
    df["num_visits_90d"] = (
        df.groupby("ID_PAZIENTE", group_keys=False)
        .apply(_rolling_prior_90d)
    )

    # ------------------------------------------------------------------
    # inter_visit_variance
    # ------------------------------------------------------------------
    df["inter_visit_variance"] = (
        df.groupby("ID_PAZIENTE")["days_since_last_visit"]
        .transform(
            lambda x: x.rolling(3, min_periods=1).std().fillna(0.0)
        )
    )

    logger.info(
        "build_features complete: %d rows — new columns: "
        "age_months, days_since_last_visit, num_visits_90d, inter_visit_variance.",
        len(df),
    )
    return df


# ---------------------------------------------------------------------------
# Step 3 – Sliding Window Matrix
# ---------------------------------------------------------------------------

def create_sliding_windows(
    df: pd.DataFrame,
    group_col: str = "ID_PAZIENTE",
    time_col: str = "DATAORA_ACCETTAZIONE",
    window_len: int = 3,
    numeric_only: bool = False,
) -> pd.DataFrame:
    """
    Build a flattened sliding-window feature matrix that captures the
    Markov property over consecutive ER visits.

    For each patient with at least *window_len* visits the function produces
    one row per window by concatenating the feature vectors of the
    *window_len* most recent visits (t-(k-1), ..., t-1, t), with column names
    suffixed ``_t``, ``_t-1``, ..., ``_t-(window_len-1)`` from newest to oldest.

    CRITICAL — ``numeric_only`` defaults to ``False``:
        Text columns such as ``Sintomi``, ``Diagnosi``, and ``Motivo_accesso``
        **must** be preserved in the output DataFrame.  They are required for
        the Italian text-matching rules in
        ``EthicalKDEAnomalyDetector._evaluate_rule()`` to fire correctly.
        Setting ``numeric_only=True`` silently drops all string columns,
        making the symbolic-AI layer of the neuro-symbolic fusion inoperative
        even though no error is raised.  Only set it to ``True`` if you
        explicitly need a numeric-only matrix for inspection purposes.

    Parameters
    ----------
    df:
        Merged ER dataframe returned by :func:`merge_er_data` and
        :func:`build_features`.
    group_col:
        Column used to group visits by patient (default: ``"ID_PAZIENTE"``).
    time_col:
        Datetime column used to sort visits chronologically
        (default: ``"DATAORA_ACCETTAZIONE"``).
    window_len:
        Number of consecutive visits per window (>= 2).
    numeric_only:
        When ``True`` only numeric columns are kept.  Default ``False``
        so that text columns survive into the window for rule matching.

    Returns
    -------
    pd.DataFrame
        Each row is a flattened window.  Column names carry temporal suffixes.
        ``ID_PAZIENTE`` and the anchor ``DATAORA_ACCETTAZIONE`` of the most
        recent visit in each window are preserved for traceability.
        Returns an empty DataFrame (with a logged warning) if no patient has
        enough visits to form a single window.
    """
    if window_len < 2:
        raise ValueError("window_len must be >= 2.")
    if group_col not in df.columns:
        raise ValueError(f"Column '{group_col}' not found in dataframe.")
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' not found in dataframe.")

    # Columns to exclude from the feature vector (keys + raw date columns).
    exclude: set[str] = {group_col, time_col, "ID_ACCESSO"}
    exclude |= {c for c in _DATE_COLUMNS if c in df.columns and c != time_col}

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
            continue  # not enough visits to form even one window

        for end_idx in range(window_len - 1, len(group)):
            window = group.iloc[end_idx - window_len + 1 : end_idx + 1]
            row: dict = {
                group_col: patient_id,
                time_col:  group.at[end_idx, time_col],
            }

            # Flatten newest→oldest: _t, _t-1, _t-2, …
            for offset, (_, visit) in enumerate(window.iloc[::-1].iterrows()):
                suffix = "_t" if offset == 0 else f"_t-{offset}"
                for col in feature_cols:
                    row[f"{col}{suffix}"] = visit[col]

            rows.append(row)

    if not rows:
        logger.warning(
            "create_sliding_windows produced 0 windows. "
            "Check that window_len (%d) <= visit counts per patient.",
            window_len,
        )
        return pd.DataFrame()

    result = pd.DataFrame(rows).reset_index(drop=True)
    logger.info(
        "create_sliding_windows: %d windows from %d patients (window_len=%d, "
        "numeric_only=%s, feature_cols=%d).",
        len(result),
        result[group_col].nunique(),
        window_len,
        numeric_only,
        len(feature_cols),
    )
    return result
