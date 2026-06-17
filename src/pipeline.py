import pandas as pd

from .config import (
    ID_COL, DATE_COL, BIRTH_COL, COL_MAP,
    KDEMODEL_PATH, WINDOW_PATH, RULE_PATH,
    TEST_LLM_PATH, TEST_KDE_PATH,
)
from .kde import load_or_train_kde
from .llm import load_llm, load_rules, valuta_caso
from .scoring import compute_score_llm, compute_score_kde, risk_level
from .pdf_report import genera_pdf
from .utils.dates import eta_in_anni as _eta_in_anni


# These are populated on first call to run() 
detector = None
kde_numeric_cols = None
model = None
tokenizer = None
rules = None
rules_by_id = None


def _init():
    global detector, kde_numeric_cols, model, tokenizer, rules, rules_by_id

    if detector is None:
        detector, kde_numeric_cols = load_or_train_kde(
            model_path=KDEMODEL_PATH,
            window_path=WINDOW_PATH,
        )

    if model is None:
        model, tokenizer = load_llm()

    if rules is None:
        rules_list, _ = load_rules(RULE_PATH)
        rules = rules_list
        rules_by_id = {r["id"]: r for r in rules_list}


def run(patient_id):

    _init()

    # --- LLM sample ---
    test_llm = pd.read_parquet(TEST_LLM_PATH)
    visit = test_llm[test_llm[ID_COL] == patient_id].sort_values(DATE_COL).iloc[-1]
    prompt_data = {campo: visit[col] for campo, col in COL_MAP.items()}
    prompt_data["eta_in_anni"] = _eta_in_anni(visit[DATE_COL], visit[BIRTH_COL])

    # --- KDE sample ---
    test_kde = pd.read_parquet(TEST_KDE_PATH)
    windows_sample = test_kde[test_kde[ID_COL] == patient_id]
    test_sample_kde = (
        windows_sample[kde_numeric_cols].iloc[[0]].reset_index(drop=True)
    )

    # --- Scores ---
    violated_rules_raw = valuta_caso(rules, prompt_data)
    llm_score = compute_score_llm(violated_rules_raw)
    kde_score = compute_score_kde(detector, test_sample_kde)
    total = llm_score + kde_score
    risk_label, _ = risk_level(total)

    has_history = kde_score > 0
    num_accessi_90d = int(test_sample_kde["num_visits_90d_t"].item())

    # Build enriched rule list for the PDF
    regole_per_pdf = [
        {
            "id": rv["id"],
            "descrizione": rules_by_id.get(rv["id"], {}).get("descrizione", str(rv["id"])),
            "gravita": rv["grav"],
            "confidenza": rv["conf"],
        }
        for rv in violated_rules_raw
    ]
    for rule in regole_per_pdf:
        if isinstance(rule.get("descrizione"), str):
            rule["descrizione"] = rule["descrizione"].replace("’", "'")

    paziente = {
        "eta_in_anni":prompt_data["eta_in_anni"],
        "sesso":prompt_data["sesso"],
        "gravita":visit.get("GRAVITA", "-"),
    }

    return genera_pdf(
        paziente=paziente,
        risk_level=risk_label,
        kde_score=kde_score,
        has_history=has_history,
        num_accessi_90d=num_accessi_90d,
        regole_violate=regole_per_pdf,
    )
