from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent

DATA_DIR = SRC_DIR / "data"
MODELS_DIR = SRC_DIR / "models"
ASSETS_FONTS_DIR = ROOT / "assets" / "fonts"

KDE_MODEL_PATH = MODELS_DIR / "kde_model.joblib"
WINDOW_PATH = DATA_DIR / "nap_negativi_windows.parquet"
RULE_PATH = DATA_DIR / "prior_rules_categorized.json"

TEST_LLM_PATH = DATA_DIR / "test_LLM.parquet"
TEST_KDE_PATH = DATA_DIR / "test_KDE.parquet"
TEST_RESULTS_PATH = DATA_DIR / "test_results.parquet"

LLM_MODEL_NAME = "unsloth/qwen2.5-14b-instruct-bnb-4bit"

ID_COL = "ID_PAZIENTE"
DATE_COL = "DATA_ACCETTAZIONE"
BIRTH_COL = "DATA_NASCITA"

COL_MAP = {
    "sesso": "SESSO",
    "problema_principale": "PROBLEMA_PRINCIPALE",
    "dati_riferiti": "DATI_RIFERITI",
    "diagnosi": "DIAGNOSI",
    "causale": "CAUSALE",
    "anamnesi": "ANAMNESI",
    "note_aggiuntive": "NOTE_AGGIUNTIVE",
}

SOGLIE = [
    (0.25, "BASSO",      "Nessuna azione immediata."),
    (0.75, "MODERATO",   "Documentare e aumentare il monitoraggio."),
    (1.50, "ALTO",       "Segnalare al medico responsabile di turno."),
    (None, "MOLTO ALTO", "Attivare il protocollo di salvaguardia immediato."),
]

GRAVITA_MAP = {"BIANCO": 0.0, "VERDE": 1.0, "GIALLO": 2.0, "ROSSO": 3.0}

BASE_FEATURES = [
    "codice_gravita",
    "codice_triage",
    "age_months",
    "days_since_last_visit",
    "num_visits_90d",
]

DEJAVU_FONTS = {
    "": str(ASSETS_FONTS_DIR / "DejaVuSans.ttf"),
    "B": str(ASSETS_FONTS_DIR / "DejaVuSans-Bold.ttf"),
    "I": str(ASSETS_FONTS_DIR / "DejaVuSans-Oblique.ttf"),
}
