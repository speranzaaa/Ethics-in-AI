# ETHICS — Sistema di Supporto Decisionale Pediatrico

Decision-support system for pediatric emergency departments that flags potential
non-accidental pathology (NAP) cases.

## Architecture

Two scoring layers run in parallel and their outputs are summed:

| Layer | What it does | Score |
|-------|-------------|-------|
| **LLM** | Evaluates each clinical rule from `prior_rules_categorized.json` and returns (violated, confidence) per rule | `llm_score ∈ [0, 1]` |
| **KDE** | Kernel Density Estimation trained on temporal access windows of non-NAP patients; flags statistically anomalous visit patterns | `kde_score ∈ [0, 1+]` |

`total_score = llm_score + kde_score`

Numeric thresholds (`SOGLIE` in `src/config.py`) map the total to a risk level:

| Score range | Livello | Azione raccomandata |
|-------------|---------|---------------------|
| < 0.25 | BASSO | Nessuna azione immediata |
| 0.25 – 0.75 | MODERATO | Documentare e aumentare il monitoraggio |
| 0.75 – 1.50 | ALTO | Segnalare al medico responsabile di turno |
| ≥ 1.50 | MOLTO ALTO | Attivare il protocollo di salvaguardia immediato |

A PDF explainability report is then generated for the clinician.

## Folder structure

```
ethics-project/
├── requirements.txt
├── notebooks/              # original Colab notebooks (reference)
├── assets/
│   └── fonts/              # place DejaVuSans .ttf files here for Unicode PDF support
├── app/
│   ├── streamlit_app.py    # Streamlit demo (simulated inference)
│   └── demo_config.py      # edit this to change demo defaults / fixed scores / rules
└── src/
    ├── config.py           # all paths and shared constants
    ├── preprocessing.py    # data cleaning pipeline (from raw Excel)
    ├── kde.py              # EthicalKDEAnomalyDetector + train/load helpers
    ├── llm.py              # LLM layer (requires GPU + unsloth)
    ├── pdf_report.py       # PDF explainability report
    ├── pipeline.py         # end-to-end orchestration (GPU path)
    ├── data/               # add datasets here by hand (not committed)
    ├── models/             # trained KDE model goes here (not committed)
    ├── scoring/
    │   └── scores.py       # compute_score, compute_score_kde, risk_level
    └── utils/
        ├── text.py         # estrai_tupla + CheckRegolaOutput
        ├── fonts.py        # DejaVu → Helvetica font setup
        └── dates.py        # age computation helpers
```

## Files to add manually

### `src/data/`
| File | Description |
|------|-------------|
| `prior_rules_categorized.json` | Clinical rules — schema: `{"regole": [{"id", "descrizione", "gravità"}], "definizioni": {}}` |
| `nap_negativi_windows.parquet` | Temporal windows for KDE training (output of preprocessing) |
| `test_LLM.parquet` | Test visit records for LLM inference |
| `test_KDE.parquet` | Test KDE windows |
| `test_results.parquet` | Pre-computed test scores for evaluation plots |

### `src/models/`
| File | Description |
|------|-------------|
| `kde_model.joblib` | Trained `EthicalKDEAnomalyDetector` (produced by `train_kde()` or copied from Colab) |

### `assets/fonts/` (optional)
Place `DejaVuSans.ttf`, `DejaVuSans-Bold.ttf`, `DejaVuSans-Oblique.ttf` here for full
Unicode support in the PDF. Without them the PDF falls back to Helvetica (Latin-1 only).

## Installation

```bash
pip install -r requirements.txt
```

> The LLM dependencies (`unsloth`, `torch`, `transformers`) require a CUDA GPU.
> For CPU-only use (KDE training, scoring, PDF, Streamlit demo) you can skip them.

## Train the KDE model

```python
# from the repo root
from src.kde import train_kde
train_kde()   # reads src/data/nap_negativi_windows.parquet, saves src/models/kde_model.joblib
```

## Streamlit demo

```bash
streamlit run app/streamlit_app.py
```

The demo simulates model loading and scoring (all values come from `app/demo_config.py`).
The only real computation is PDF generation.  No GPU required.

## Full pipeline (GPU, from the notebook)

The original `notebooks/Ethics_Complete.ipynb` is the intended way to run the end-to-end
pipeline with the real LLM. It requires a CUDA GPU.
The `src.pipeline.run(patient_id)` function replicates the same flow as an importable
module, also requiring a GPU.
