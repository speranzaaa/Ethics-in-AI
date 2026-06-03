"""
models.py
~~~~~~~~~
EthicalKDEAnomalyDetector: multivariate KDE-based anomaly scorer with
Prior-Knowledge integration via LLM-extracted defeasible rules.

NEURO-SYMBOLIC FUSION ARCHITECTURE:
  This detector combines two complementary AI paradigms:

  - Sub-symbolic (KDE): models the probability density P(x | normal) over
    the numerical features of a patient's sliding-window ER visit timeline.
    The raw anomaly signal is the negative log-likelihood: −log P_KDE(x).
    This is then **calibrated to [0, 1]** via a MinMaxScaler fitted on the
    training-data score distribution, making it commensurable with the
    symbolic penalties (also in [0, ∞)).

  - Symbolic (Prior Rules): the LLM—acting as an expert system—extracts
    deterministic clinical constraints from Italian medical guidelines
    ("Quaderni della Regione Emilia-Romagna").  Each rule encodes a known
    red flag as a case-insensitive Italian substring searched across the
    string-valued columns of the patient feature row.

  Fusion yields the final anomaly score:

      score(x) = calibrate(−log P_KDE(x))                               ← [0, 1+]
               + SUM_i penalty_weight_i · I[clinical_conditions_i ∈ x]  ← text rules
               × prior_penalty^(# violated numeric rules)               ← numeric rules

  where calibrate(s) = clip(MinMaxScaler(s), 0, ∞):
    - Normal training samples map to [0, 1].
    - Out-of-distribution anomalies exceed 1.0 (preserved — not clipped).
    - Values below 0 (extremely "normal") are clipped to 0.

  The additive text penalties ensure that a single documented clinical
  finding (e.g., "fratture multiple" in the Diagnosi field) raises the
  score even when the visit frequency pattern looks normal — covering the
  common scenario of a first-time presentation of abuse.

BANDWIDTH SELECTION:
  .fit() runs GridSearchCV over a log-spaced grid of 30 bandwidth candidates
  (0.05 – 5.0), evaluated by cross-validated log-likelihood.  The optimal
  bandwidth is selected automatically; no manual tuning is required.

ITALIAN TERMINOLOGY:
  Text rules from :mod:`util.knowledge_extraction` carry ``clinical_conditions``
  in Italian.  This is mandatory: our CSV datasets (Triage, Sintomi, Dati
  clinici) are in Italian, and matching is case-insensitive substring search.
  Translating terms to English would silently break detection.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler, RobustScaler

logger = logging.getLogger(__name__)


class EthicalKDEAnomalyDetector:
    """
    Anomaly detector that combines a calibrated Kernel Density Estimator
    with clinician/LLM-derived prior rules.

    The statistical (sub-symbolic) component estimates the density of normal
    patient pathways.  Its output — the negative log-likelihood — is
    calibrated to **[0, 1]** during `.fit()` using a MinMaxScaler so that it
    is commensurable with the additive symbolic penalty scores.

    Fitting automatically selects the optimal KDE bandwidth via cross-validated
    log-likelihood (GridSearchCV over a log-spaced grid of 30 candidates),
    scales features with :class:`~sklearn.preprocessing.RobustScaler` (robust
    to clinical outliers), and fits a MinMaxScaler on the resulting NLL scores
    for score calibration.

    Parameters
    ----------
    bandwidth:
        Initial KDE bandwidth hint.  Overridden by GridSearchCV during
        `.fit()` unless ``bandwidth_cv=False`` is passed to `.fit()`.
    kernel:
        Kernel function passed to :class:`sklearn.neighbors.KernelDensity`.
    prior_penalty:
        Multiplicative amplification applied per violated *numeric* rule
        (legacy path).  Text rules use additive ``penalty_weight`` instead.
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

        self._scaler = RobustScaler()
        self._kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
        self._score_scaler: MinMaxScaler | None = None
        self._best_bandwidth: float = bandwidth
        self._fitted = False
        self.prior_rules: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Prior Knowledge API
    # ------------------------------------------------------------------

    def set_prior_knowledge(self, rules_json: str | list[dict]) -> None:
        """
        Load defeasible prior rules extracted by the LLM.

        Each rule dict should carry at least one of:

        Text rules (new-style, additive penalty):
          - ``"clinical_conditions"`` (str) — Italian clinical term.
          - ``"penalty_weight"`` (float) — additive score boost on match.

        Numeric rules (legacy, multiplicative):
          - ``"feature"`` / ``"condition"`` / ``"threshold"`` keys.

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

    def fit(
        self,
        X_train: np.ndarray,
        cv: int = 5,
        bandwidth_cv: bool = True,
    ) -> "EthicalKDEAnomalyDetector":
        """
        Fit the KDE on a matrix of normal patient pathways.

        Steps
        -----
        1. **RobustScaler** — median-centred, IQR-normalised feature scaling.
           Preferred over StandardScaler for medical data with extreme outliers
           (e.g., ``days_since_last_visit`` = 3650 sentinel value).
        2. **Bandwidth selection** — GridSearchCV over 30 log-spaced candidates
           (0.05 – 5.0), scored by cross-validated log-likelihood.
           Skipped automatically when ``n_samples < 10`` or
           ``bandwidth_cv=False``.
        3. **KDE fit** — refit with the optimal bandwidth.
        4. **Score calibration** — fit a MinMaxScaler on the training-data NLL
           so that ``get_anomaly_signal()`` returns values in **[0, 1]** for
           training-like inputs.  Out-of-distribution samples may exceed 1.0.

        Parameters
        ----------
        X_train:
            2-D array of shape ``(n_samples, n_features)`` containing only
            *normal* (non-abused) patient windows.
        cv:
            Cross-validation folds for bandwidth selection.  Clamped to
            ``max(2, min(cv, n_samples))``.
        bandwidth_cv:
            When ``False``, skip GridSearchCV and use ``self.bandwidth``
            directly (useful for reproducibility or tiny datasets).

        Returns
        -------
        self
        """
        X_scaled = self._scaler.fit_transform(X_train)
        n_samples = len(X_scaled)

        # ── Bandwidth selection ───────────────────────────────────────────────
        if bandwidth_cv and n_samples >= 10:
            bandwidths = np.logspace(-1.3, 0.7, 30)   # 0.05 → ~5.0
            n_folds = max(2, min(cv, n_samples))
            grid_cv = GridSearchCV(
                KernelDensity(kernel=self.kernel),
                {"bandwidth": bandwidths},
                cv=n_folds,
                n_jobs=-1,
            )
            grid_cv.fit(X_scaled)
            self._best_bandwidth = float(grid_cv.best_params_["bandwidth"])
            logger.info(
                "Bandwidth CV: best=%.4f over %d candidates, %d folds.",
                self._best_bandwidth,
                len(bandwidths),
                n_folds,
            )
        else:
            self._best_bandwidth = self.bandwidth
            if n_samples < 10:
                logger.warning(
                    "Only %d training samples — bandwidth CV skipped, "
                    "using bandwidth=%.3f.",
                    n_samples,
                    self.bandwidth,
                )

        # ── KDE fit ───────────────────────────────────────────────────────────
        self._kde = KernelDensity(bandwidth=self._best_bandwidth, kernel=self.kernel)
        self._kde.fit(X_scaled)

        # ── Score calibration ─────────────────────────────────────────────────
        # MinMaxScaler fitted on training NLL so that normal samples → [0, 1].
        # This makes the KDE base signal commensurable with symbolic penalties.
        raw_train_nll = -self._kde.score_samples(X_scaled)
        self._score_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        self._score_scaler.fit(raw_train_nll.reshape(-1, 1))

        self._fitted = True
        logger.info(
            "KDE fitted: %d samples, %d features. "
            "Bandwidth=%.4f. Raw NLL [%.3f, %.3f] → calibrated [0, 1].",
            X_train.shape[0],
            X_train.shape[1],
            self._best_bandwidth,
            float(raw_train_nll.min()),
            float(raw_train_nll.max()),
        )
        return self

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _check_rule(self, feature_values: dict[str, float], rule: dict) -> bool:
        """Return ``True`` if *feature_values* violates a numeric threshold rule."""
        col = rule.get("feature")
        if col not in feature_values:
            return False

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
        Check whether a patient feature row matches a text-based clinical rule.

        Performs a case-insensitive substring search across every string-typed
        cell in ``row``.  Returns ``True`` if the Italian phrase in
        ``rule["clinical_conditions"]`` appears in any field.

        ITALIAN TERMINOLOGY REQUIREMENT:
          ``rule["clinical_conditions"]`` must be in Italian (e.g.,
          "fratture multiple", "lesioni cutanee", "ustioni") to match the
          Italian-language fields in the clinical CSV datasets.

        Parameters
        ----------
        row:
            One row of the ``X`` DataFrame passed to :meth:`get_anomaly_signal`.
        rule:
            Rule dict from :func:`util.knowledge_extraction.extract_rules_local_ollama`
            with key ``"clinical_conditions"`` holding an Italian string.

        Returns
        -------
        bool
            ``True`` if the Italian phrase is found in at least one string
            column; ``False`` if the field is absent or empty.
        """
        condition_text = str(rule.get("clinical_conditions", "")).strip().lower()
        if not condition_text:
            return False

        for value in row:
            if isinstance(value, str) and condition_text in value.lower():
                return True
        return False

    # ------------------------------------------------------------------
    # Main scoring method
    # ------------------------------------------------------------------

    def get_anomaly_signal(self, X: pd.DataFrame) -> pd.Series:
        """
        Compute the fused anomaly signal for each patient window in *X*.

        Steps
        -----
        1. **Calibrated KDE base** — raw NLL transformed to [0, 1] by the
           MinMaxScaler fitted during `.fit()`.  Values below 0 are clipped
           to 0; values above 1 are preserved (out-of-distribution anomalies).

        2. **Text rules (additive)** — each rule with ``"clinical_conditions"``
           and ``"penalty_weight"`` is evaluated via :meth:`_evaluate_rule`.
           A match adds ``penalty_weight`` to the calibrated base.

        3. **Numeric rules (multiplicative, legacy)** — rules with
           ``"feature"`` / ``"condition"`` / ``"threshold"`` amplify the
           signal by ``self.prior_penalty`` per violation.

        Parameters
        ----------
        X:
            DataFrame of shape ``(n_samples, n_features)``.  Numeric columns
            feed the KDE; string/object columns are used by text rules.

        Returns
        -------
        pd.Series
            Fused anomaly scores indexed by ``X.index``.
            Higher = more suspicious.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before scoring. Call .fit() first.")

        # ── KDE base signal ───────────────────────────────────────────────────
        numeric_X = X.select_dtypes(include=[np.number])
        X_scaled = self._scaler.transform(numeric_X.values)
        raw_nll = -self._kde.score_samples(X_scaled)

        score_scaler = getattr(self, "_score_scaler", None)
        if score_scaler is not None:
            calibrated = score_scaler.transform(raw_nll.reshape(-1, 1)).ravel()
            base_signal = np.clip(calibrated, 0.0, None)
        else:
            base_signal = raw_nll

        if not self.prior_rules:
            return pd.Series(base_signal, index=X.index, name="anomaly_signal")

        # ── Symbolic penalties ────────────────────────────────────────────────
        adjusted = base_signal.copy()
        for i, (_, row) in enumerate(X.iterrows()):
            numeric_fv: dict[str, float] = numeric_X.iloc[i].to_dict()
            for rule in self.prior_rules:
                if "clinical_conditions" in rule and "penalty_weight" in rule:
                    if self._evaluate_rule(row, rule):
                        adjusted[i] += float(rule["penalty_weight"])
                        logger.debug(
                            "Sample %d: rule '%s' matched → +%.2f penalty.",
                            i,
                            rule.get("rule_id", rule.get("clinical_conditions", "?")),
                            rule["penalty_weight"],
                        )
                elif "feature" in rule and "condition" in rule and "threshold" in rule:
                    if self._check_rule(numeric_fv, rule):
                        adjusted[i] *= self.prior_penalty
                        logger.debug(
                            "Sample %d violated numeric rule '%s' → x%.2f.",
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
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Binary prediction: ``1`` = anomalous, ``0`` = normal.

        Parameters
        ----------
        X:
            Feature DataFrame (see :meth:`get_anomaly_signal`).
        threshold:
            Anomaly signal cutoff.  With calibrated scores in [0, 1+], 0.5
            sits halfway across the training distribution.  Tune this via
            :class:`util.ethics_metrics.CostModel` for your false-positive /
            false-negative cost ratio.

        Returns
        -------
        np.ndarray
            Integer array of shape ``(n_samples,)`` with values in ``{0, 1}``.
        """
        signals = self.get_anomaly_signal(X)
        return (signals >= threshold).astype(int).values

    def __repr__(self) -> str:
        best_bw = getattr(self, "_best_bandwidth", self.bandwidth)
        bw_label = (
            f"{best_bw:.4f} (CV-optimised)"
            if best_bw != self.bandwidth
            else f"{self.bandwidth:.4f} (fixed)"
        )
        return (
            f"EthicalKDEAnomalyDetector("
            f"bandwidth={bw_label}, kernel='{self.kernel}', fitted={self._fitted})"
        )
