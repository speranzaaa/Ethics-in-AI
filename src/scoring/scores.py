import numpy as np

from ..config import SOGLIE


def compute_score_llm(violated_rules, N_target=3, alpha=0.3):
    if not violated_rules:
        return 0.0

    contribs = [
        max(0.0, min(1.0, r["grav"] * r["conf"]))
        for r in violated_rules
    ]

    forza = sum(contribs) / len(contribs)
    quantita = 1.0 - np.exp(-len(violated_rules) / N_target)

    return (forza ** alpha) * (quantita ** (1.0 - alpha))


def compute_score_kde(detector, sample):
    kde_score = float(detector.get_anomaly_signal(sample).iloc[0])
    return max(0.0, kde_score)


def risk_level(total_score, soglie=None):
    if soglie is None:
        soglie = SOGLIE

    label, azione = "MOLTO ALTO", ""
    for soglia, lbl, az in soglie:
        if soglia is None or total_score < soglia:
            label, azione = lbl, az
            break
    return label, azione
