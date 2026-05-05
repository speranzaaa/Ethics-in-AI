"""
models.py
~~~~~~~~~
EthicalKDEAnomalyDetector: multivariate KDE-based anomaly scorer with
Prior-Knowledge integration via LLM-extracted defeasible rules.

NEURO-SYMBOLIC FUSION ARCHITECTURE:
  This detector combines two complementary AI paradigms:

  - Sub-symbolic (KDE): models the probability density P(x | normal) over
    the numerical features of a patient's sliding-window ER visit timeline.
    The base anomaly signal is the negative log-likelihood: -log P_KDE(x).

  - Symbolic (Prior Rules): the LLM—acting as an expert system—extracts
    deterministic clinical constraints from Italian medical guidelines
    ("Quaderni della Regione Emilia-Romagna").  Each rule encodes a known
    red flag as a case-insensitive Italian substring that is searched across
    the string-valued columns of the patient feature row.

  Fusion yields the final anomaly score:

      score(x) = -log P_KDE(x)
               + SUM_i penalty_weight_i · I[clinical_conditions_i ∈ x]   (text rules)
               × prior_penalty^(# violated numeric rules)                (numeric rules)

  The additive text penalties ensure that a single documented clinical
  finding (e.g., "fratture multiple" in the Diagnosi field) raises the
  score even when the visit frequency pattern looks normal—covering the
  common scenario of a first-time presentation of abuse.

ITALIAN TERMINOLOGY:
  Text rules produced by :mod:`util.knowledge_extraction` carry their
  ``clinical_conditions`` field in Italian.  This is mandatory: our CSV
  datasets (Triage, Sintomi, Dati clinici) are in Italian, and matching
  is performed via case-insensitive substring search.  Translating terms
  to English would silently break detection.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
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
        """Return ``True`` if *feature_values* violates a numeric threshold rule."""
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

    def _evaluate_rule(self, row: pd.Series, rule: dict) -> bool:
        """
        Check whether a patient's feature row matches a text-based clinical rule.

        This method implements the SYMBOLIC AI matching step of the neuro-symbolic
        fusion pipeline.  It performs a case-insensitive substring search: it scans
        every string-typed cell in ``row`` and returns ``True`` if the Italian
        clinical condition phrase (``rule["clinical_conditions"]``) appears in any
        of them.

        ITALIAN TERMINOLOGY REQUIREMENT:
          The ``rule["clinical_conditions"]`` string must be in Italian (e.g.,
          "fratture multiple", "lesioni cutanee", "ustioni").  This matches the
          language of the clinical CSV datasets (Triage, Sintomi, Dati clinici).
          The search is intentionally case-insensitive to handle capitalisation
          variation in free-text fields (e.g., "Ustioni" vs "ustioni").

        Parameters
        ----------
        row:
            A single patient feature row from the sliding-window DataFrame
            (one row of ``X`` passed to :meth:`get_anomaly_signal`).
            May contain a mix of numeric and string-typed values.
        rule:
            A rule dictionary produced by
            :func:`util.knowledge_extraction.extract_rules_with_llm`, containing
            at minimum the key ``"clinical_conditions"`` (an Italian string).

        Returns
        -------
        bool
            ``True`` if the Italian condition phrase is found in at least one
            string column of ``row``; ``False`` otherwise or if the condition
            string is empty.
        """
        condition_text = str(rule.get("clinical_conditions", "")).strip().lower()
        if not condition_text:
            return False

        for value in row:
            if isinstance(value, str) and condition_text in value.lower():
                return True
        return False

    def get_anomaly_signal(self, X: pd.DataFrame) -> pd.Series:
        """
        Compute the fused anomaly signal for each patient window in *X*.

        This method implements the neuro-symbolic fusion step:

          1. **Sub-symbolic base signal** — the negative log-likelihood under the
             fitted KDE, computed on the *numeric* columns of ``X`` after standard
             scaling: ``base_signal[i] = -log P_KDE(x_i)``.

          2. **Symbolic penalty — text rules** — for each row, every rule in
             ``self.prior_rules`` that carries a ``"clinical_conditions"`` key is
             evaluated via :meth:`_evaluate_rule`.  If the Italian condition string
             is found in any string column, the rule's ``penalty_weight`` is *added*
             to the signal:
             ``signal[i] += penalty_weight  if clinical_conditions_i ∈ x_i``.

          3. **Symbolic penalty — numeric rules** (legacy) — rules carrying
             ``"feature"`` / ``"condition"`` / ``"threshold"`` keys are evaluated
             via :meth:`_check_rule`.  A match multiplies the signal by
             ``self.prior_penalty``.

        Additive penalties (step 2) are preferred for text rules because they
        guarantee a minimum signal boost regardless of the KDE base value,
        which is essential when the statistical model assigns low anomaly scores
        to presentations that are clinically rare but not statistically unusual
        in the training population.

        Parameters
        ----------
        X:
            DataFrame of shape ``(n_samples, n_features)`` representing patient
            sliding-window feature matrices.  May contain a mix of numeric columns
            (used by KDE) and string/object columns (used by text rules).

        Returns
        -------
        pd.Series
            Anomaly scores indexed by ``X.index``.  Higher values indicate more
            suspicious patient pathways.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before scoring. Call .fit() first.")

        # KDE operates on numeric features only.
        numeric_X = X.select_dtypes(include=[np.number])
        X_scaled = self._scaler.transform(numeric_X.values)
        base_signal = -self._kde.score_samples(X_scaled)  # shape: (n_samples,)

        if not self.prior_rules:
            return pd.Series(base_signal, index=X.index, name="anomaly_signal")

        adjusted = base_signal.copy()
        for i, (_, row) in enumerate(X.iterrows()):
            numeric_fv: dict[str, float] = numeric_X.iloc[i].to_dict()
            for rule in self.prior_rules:
                if "clinical_conditions" in rule and "penalty_weight" in rule:
                    # New-style text rule: additive penalty fuses symbolic + statistical.
                    if self._evaluate_rule(row, rule):
                        adjusted[i] += float(rule["penalty_weight"])
                        logger.debug(
                            "Sample %d: Italian rule '%s' matched → +%.2f penalty.",
                            i,
                            rule.get("rule_id", rule.get("clinical_conditions", "?")),
                            rule["penalty_weight"],
                        )
                elif "feature" in rule and "condition" in rule and "threshold" in rule:
                    # Legacy numeric rule: multiplicative amplification.
                    if self._check_rule(numeric_fv, rule):
                        adjusted[i] *= self.prior_penalty
                        logger.debug(
                            "Sample %d violated numeric rule '%s' → ×%.2f amplification.",
                            i,
                            rule.get("description", rule.get("feature", "?")),
                            self.prior_penalty,
                        )

        return pd.Series(adjusted, index=X.index, name="anomaly_signal")

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def predict(
        self,
        X: pd.DataFrame,
        threshold: float = 0.0,
    ) -> np.ndarray:
        """
        Binary prediction: ``1`` = anomalous, ``0`` = normal.

        Parameters
        ----------
        X:
            Feature DataFrame (see :meth:`get_anomaly_signal`).
        threshold:
            Anomaly signal cutoff.  Samples with a score at or above this
            value are flagged as suspicious.

        Returns
        -------
        np.ndarray
            Integer array of shape ``(n_samples,)`` with values in ``{0, 1}``.
        """
        signals = self.get_anomaly_signal(X)
        return (signals >= threshold).astype(int).values

    def __repr__(self) -> str:
        return (
            f"EthicalKDEAnomalyDetector("
            f"bandwidth={self.bandwidth}, kernel='{self.kernel}', "
            f"prior_rules={len(self.prior_rules)}, fitted={self._fitted})"
        )
