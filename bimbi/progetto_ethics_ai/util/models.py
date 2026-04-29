"""
models.py
~~~~~~~~~
EthicalKDEAnomalyDetector: multivariate KDE-based anomaly scorer with
Prior-Knowledge integration via LLM-extracted defeasible rules.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class EthicalKDEAnomalyDetector:
    """
    Anomaly detector that combines Kernel Density Estimation with a set of
    clinician/LLM-derived prior rules applied as penalty multipliers.

    The anomaly signal is the negative log-likelihood of the KDE model
    (higher = more anomalous).  If a sample violates one or more prior
    rules the raw signal is amplified by a configurable penalty factor,
    encoding domain knowledge in a transparent, auditable way.

    Parameters
    ----------
    bandwidth:
        KDE bandwidth (``h``).  Smaller → sharper density, more sensitive.
    kernel:
        Kernel function passed to :class:`sklearn.neighbors.KernelDensity`.
    prior_penalty:
        Multiplicative amplification applied per violated rule.
    """

    def __init__(
        self,
        bandwidth: float = 1.0,
        kernel: str = "gaussian",
        prior_penalty: float = 1.5,
    ) -> None:
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.prior_penalty = prior_penalty

        self._kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
        self._scaler = StandardScaler()
        self._fitted = False
        self.prior_rules: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Prior Knowledge API
    # ------------------------------------------------------------------

    def set_prior_knowledge(self, rules_json: str | list[dict]) -> None:
        """
        Load defeasible prior rules extracted by the LLM.

        Each rule is a dict with at least:
        - ``"feature"`` (str): feature name in the window dataframe.
        - ``"condition"`` (str): one of ``"gt"``, ``"lt"``, ``"eq"``,
          ``"gte"``, ``"lte"``.
        - ``"threshold"`` (float): comparison value.
        - ``"description"`` (str): human-readable rationale.

        Parameters
        ----------
        rules_json:
            Either a JSON string or an already-parsed list of rule dicts.
        """
        if isinstance(rules_json, str):
            self.prior_rules = json.loads(rules_json)
        else:
            self.prior_rules = list(rules_json)
        logger.info("Loaded %d prior rules.", len(self.prior_rules))

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray) -> "EthicalKDEAnomalyDetector":
        """
        Fit the KDE on a matrix of normal patient pathways.

        Parameters
        ----------
        X_train:
            2-D array of shape ``(n_samples, n_features)`` containing
            only *normal* (non-abused) patient windows.

        Returns
        -------
        self
        """
        X_scaled = self._scaler.fit_transform(X_train)
        self._kde.fit(X_scaled)
        self._fitted = True
        logger.info(
            "KDE fitted on %d samples, %d features. Bandwidth=%.3f.",
            X_train.shape[0],
            X_train.shape[1],
            self.bandwidth,
        )
        return self

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _check_rule(self, feature_values: dict[str, float], rule: dict) -> bool:
        """Return ``True`` if *feature_values* violates *rule*."""
        col = rule.get("feature")
        if col not in feature_values:
            return False  # rule references a column not present → skip

        val = feature_values[col]
        threshold = float(rule["threshold"])
        cond = rule["condition"].lower()

        ops = {
            "gt":  val >  threshold,
            "gte": val >= threshold,
            "lt":  val <  threshold,
            "lte": val <= threshold,
            "eq":  val == threshold,
        }
        return ops.get(cond, False)

    def get_anomaly_signal(
        self,
        X: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> np.ndarray:
        """
        Compute the anomaly signal for each sample in *X*.

        The base signal is the negative log-likelihood under the KDE.
        For each sample the signal is multiplied by ``prior_penalty``
        once per violated prior rule.

        Parameters
        ----------
        X:
            2-D array of shape ``(n_samples, n_features)``.
        feature_names:
            Column names matching the columns of *X*.  Required for
            prior-rule evaluation; if ``None`` rules are skipped.

        Returns
        -------
        np.ndarray
            1-D array of anomaly scores (higher = more suspicious).
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before scoring. Call .fit() first.")

        X_scaled = self._scaler.transform(X)
        base_signal = -self._kde.score_samples(X_scaled)  # (n_samples,)

        if not self.prior_rules or feature_names is None:
            return base_signal

        amplified = base_signal.copy()
        for i, sample in enumerate(X):
            fv = dict(zip(feature_names, sample))
            for rule in self.prior_rules:
                if self._check_rule(fv, rule):
                    amplified[i] *= self.prior_penalty
                    logger.debug(
                        "Sample %d violated rule '%s' → signal amplified.",
                        i,
                        rule.get("description", rule.get("feature", "?")),
                    )

        return amplified

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def predict(
        self,
        X: np.ndarray,
        threshold: float = 0.0,
        feature_names: list[str] | None = None,
    ) -> np.ndarray:
        """
        Binary prediction: ``1`` = anomalous, ``0`` = normal.

        Parameters
        ----------
        X:
            Feature matrix.
        threshold:
            Anomaly signal cutoff.  Samples above this value are flagged.
        feature_names:
            See :meth:`get_anomaly_signal`.
        """
        signals = self.get_anomaly_signal(X, feature_names=feature_names)
        return (signals >= threshold).astype(int)

    def __repr__(self) -> str:
        return (
            f"EthicalKDEAnomalyDetector("
            f"bandwidth={self.bandwidth}, kernel='{self.kernel}', "
            f"prior_rules={len(self.prior_rules)}, fitted={self._fitted})"
        )
