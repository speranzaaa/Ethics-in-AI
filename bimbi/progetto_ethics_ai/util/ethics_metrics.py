"""
ethics_metrics.py
~~~~~~~~~~~~~~~~~
Fairness metrics (SPD, DI) and a Cost Model for threshold optimisation
in the child-abuse CDSS.

References
----------
- Barocas, S., Hardt, M., & Narayanan, A. (2023). *Fairness and Machine
  Learning*. fairmlbook.org
- Verma, S. & Rubin, J. (2018). Fairness Definitions Explained.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _positive_rate(
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Return the fraction of positive predictions inside *mask*."""
    subset = y_pred[mask]
    if len(subset) == 0:
        return 0.0
    return float(np.mean(subset))


# ---------------------------------------------------------------------------
# Statistical Parity Difference
# ---------------------------------------------------------------------------


def calculate_spd(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attribute: np.ndarray,
    privileged_group: Any,
    unprivileged_group: Any,
) -> float:
    """
    Calculate the Statistical Parity Difference (SPD).

    SPD = P(Ŷ=1 | S=unprivileged) − P(Ŷ=1 | S=privileged)

    A value of 0 indicates perfect demographic parity.
    Negative values mean the unprivileged group receives fewer positive
    predictions (e.g., fewer referrals).

    Parameters
    ----------
    y_true:
        Ground-truth labels (unused but kept for API symmetry with DI).
    y_pred:
        Binary predictions array.
    sensitive_attribute:
        Array of group membership values (same length as *y_pred*).
    privileged_group:
        The value in *sensitive_attribute* that identifies the privileged
        group (e.g., ``"caucasian"`` or ``1``).
    unprivileged_group:
        The value in *sensitive_attribute* that identifies the
        unprivileged group.

    Returns
    -------
    float
        SPD in [-1, 1].
    """
    y_pred = np.asarray(y_pred)
    sensitive_attribute = np.asarray(sensitive_attribute)

    priv_mask   = sensitive_attribute == privileged_group
    unpriv_mask = sensitive_attribute == unprivileged_group

    pr_priv   = _positive_rate(y_pred, priv_mask)
    pr_unpriv = _positive_rate(y_pred, unpriv_mask)

    spd = pr_unpriv - pr_priv
    logger.debug("SPD: PR(unpriv)=%.4f, PR(priv)=%.4f → SPD=%.4f", pr_unpriv, pr_priv, spd)
    return spd


# ---------------------------------------------------------------------------
# Disparate Impact
# ---------------------------------------------------------------------------


def calculate_di(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attribute: np.ndarray,
    privileged_group: Any,
    unprivileged_group: Any,
) -> float:
    """
    Calculate the Disparate Impact ratio (DI).

    DI = P(Ŷ=1 | S=unprivileged) / P(Ŷ=1 | S=privileged)

    The 4/5ths rule requires DI ≥ 0.8 to pass basic fairness screening.
    A value of 1.0 indicates perfect demographic parity.

    Parameters
    ----------
    (same as :func:`calculate_spd`)

    Returns
    -------
    float
        DI ratio in [0, ∞).  Returns ``np.nan`` if PR(privileged) == 0.
    """
    y_pred = np.asarray(y_pred)
    sensitive_attribute = np.asarray(sensitive_attribute)

    priv_mask   = sensitive_attribute == privileged_group
    unpriv_mask = sensitive_attribute == unprivileged_group

    pr_priv   = _positive_rate(y_pred, priv_mask)
    pr_unpriv = _positive_rate(y_pred, unpriv_mask)

    if pr_priv == 0.0:
        logger.warning("Privileged positive rate is 0 — DI is undefined.")
        return float("nan")

    di = pr_unpriv / pr_priv
    logger.debug("DI: PR(unpriv)=%.4f, PR(priv)=%.4f → DI=%.4f", pr_unpriv, pr_priv, di)
    return di


# ---------------------------------------------------------------------------
# Cost Model
# ---------------------------------------------------------------------------


@dataclass
class CostModel:
    """
    Asymmetric cost model for binary classification thresholds.

    In child-abuse detection, a missed abuse case (False Negative) is
    dramatically more costly than a false alarm (False Positive).

    Parameters
    ----------
    cost_alarm:
        Cost incurred per False Positive (unnecessary investigation).
    cost_missed:
        Cost incurred per False Negative (missed abuse case).
    """

    cost_alarm:  float = 1.0
    cost_missed: float = 10.0
    _history: list[dict] = field(default_factory=list, repr=False)

    def evaluate_threshold(
        self,
        signals: np.ndarray,
        true_labels: np.ndarray,
        threshold: float,
    ) -> dict[str, float]:
        """
        Evaluate a single decision threshold.

        Parameters
        ----------
        signals:
            Continuous anomaly scores from the KDE detector.
        true_labels:
            Ground-truth binary labels (1 = abuse, 0 = normal).
        threshold:
            Decision boundary; samples ≥ threshold are flagged as positive.

        Returns
        -------
        dict
            Keys: ``tp``, ``fp``, ``tn``, ``fn``, ``total_cost``,
            ``precision``, ``recall``, ``f1``.
        """
        signals     = np.asarray(signals, dtype=float)
        true_labels = np.asarray(true_labels, dtype=int)

        y_pred = (signals >= threshold).astype(int)

        tp = int(np.sum((y_pred == 1) & (true_labels == 1)))
        fp = int(np.sum((y_pred == 1) & (true_labels == 0)))
        tn = int(np.sum((y_pred == 0) & (true_labels == 0)))
        fn = int(np.sum((y_pred == 0) & (true_labels == 1)))

        total_cost = (fp * self.cost_alarm) + (fn * self.cost_missed)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        result = {
            "threshold":  threshold,
            "tp":         tp,
            "fp":         fp,
            "tn":         tn,
            "fn":         fn,
            "total_cost": total_cost,
            "precision":  precision,
            "recall":     recall,
            "f1":         f1,
        }
        self._history.append(result)
        return result

    def optimize_threshold(
        self,
        signals: np.ndarray,
        true_labels: np.ndarray,
        threshold_range: np.ndarray | None = None,
    ) -> dict[str, float]:
        """
        Find the threshold that minimises total cost.

        Parameters
        ----------
        signals:
            Continuous anomaly scores.
        true_labels:
            Ground-truth binary labels.
        threshold_range:
            1-D array of candidate thresholds.  Defaults to 50 evenly
            spaced values between ``min(signals)`` and ``max(signals)``.

        Returns
        -------
        dict
            The result dict of :meth:`evaluate_threshold` for the best
            threshold, with an additional ``"all_results"`` key containing
            a :class:`pandas.DataFrame` of all evaluated thresholds.
        """
        signals = np.asarray(signals, dtype=float)

        if threshold_range is None:
            threshold_range = np.linspace(signals.min(), signals.max(), 50)

        results = [
            self.evaluate_threshold(signals, true_labels, float(t))
            for t in threshold_range
        ]

        best = min(results, key=lambda r: r["total_cost"])
        best["all_results"] = pd.DataFrame(results)

        logger.info(
            "Optimal threshold=%.4f → total_cost=%.2f (FP=%d, FN=%d)",
            best["threshold"],
            best["total_cost"],
            best["fp"],
            best["fn"],
        )
        return best
