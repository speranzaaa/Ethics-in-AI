# ETHICS — Project context for Claude Code

## What this system does

ETHICS is a decision-support system for pediatric emergency departments (NAP = Non-Accidental
Pathology / maltrattamento pediatrico). Two layers produce scores that are summed:

- **LLM layer** (`src/llm.py`): evaluates each rule in `prior_rules_categorized.json`
  against clinical fields; returns (violated, confidence) per rule; produces `llm_score`.
- **KDE layer** (`src/kde.py`): `EthicalKDEAnomalyDetector` trained on temporal windows
  (3 visits) of non-NAP patients; flags anomalous access patterns; produces `kde_score`.

`total_score = llm_score + kde_score` → `SOGLIE` thresholds → risk level
(BASSO / MODERATO / ALTO / MOLTO ALTO) → PDF report via `src/pdf_report.py`.

## Hard constraints

1. **No CPU fallbacks for the LLM layer.** `src/llm.py` has real `unsloth`/`torch` imports
   and assumes a GPU. It is only ever run from the notebook on Colab.
2. **`src/__init__.py` must NOT import `llm` or `pipeline`** — those pull in GPU-only deps.
   The Streamlit demo must be importable on a machine without `unsloth`/`torch`.
3. **Streamlit demo is fully simulated.** Scores come from `app/demo_config.py`.
   Only `genera_pdf` (from `src.pdf_report`) and `risk_level` (from `src.scoring`)
   are real calls.  Do not add real model loading to the demo.
4. **Everything configurable in the demo lives in `app/demo_config.py`** — default patient
   fields, fixed scores, history data, violated rules list.
5. **No datasets in the repo.** Files in `src/data/` and `src/models/` are added by hand.

## Source of truth

Notebooks in `notebooks/` are the ground truth. When in doubt, follow them.
- `notebooks/ETHICS_data_cleaning.ipynb` — preprocessing pipeline
- `notebooks/Ethics_Complete.ipynb` — LLM, KDE, scoring, PDF

## Key identifiers (Italian — do not translate)

- `ID_PAZIENTE`, `DATA_ACCETTAZIONE`, `DATA_NASCITA` — patient/visit columns
- `GRAVITA`, `DIAGNOSI`, `ANAMNESI`, `NOTE_AGGIUNTIVE`, etc. — clinical columns
- `NAP_LABEL`: confermato / sospetto / segnale_sociale / negativo
- `SOGLIE`, `GRAVITA_MAP`, `BASE_FEATURES` — constants in `src/config.py`

## Module layout

```
src/config.py          — all paths and shared constants
src/preprocessing.py   — data cleaning (from raw Excel)
src/kde.py             — EthicalKDEAnomalyDetector, train_kde, load_or_train_kde
src/llm.py             — LLM layer (GPU only; not imported by __init__)
src/pdf_report.py      — genera_pdf + sezione_* helpers
src/pipeline.py        — run(patient_id) end-to-end (GPU only)
src/scoring/scores.py  — compute_score, compute_score_kde, risk_level
src/utils/text.py      — estrai_tupla, CheckRegolaOutput
src/utils/fonts.py     — setup_fonts(pdf)
src/utils/dates.py     — eta_in_anni, age_months
app/streamlit_app.py   — Streamlit demo (CPU-only)
app/demo_config.py     — all demo-configurable values
```

## Running things

```bash
# Install (skip GPU deps if no CUDA)
pip install -r requirements.txt

# Train KDE
python -c "from src.kde import train_kde; train_kde()"

# Streamlit demo
streamlit run app/streamlit_app.py

# Full pipeline (GPU, from Python or notebook)
from src.pipeline import run
pdf_bytes = run("CC90000013")
```
