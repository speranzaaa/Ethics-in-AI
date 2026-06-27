# GUARDIAN - Guided Understanding And Risk Detection In Abuse Identification Network

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
├── notebooks/              # Colab notebooks (reference)
├── assets/
│   └── fonts/              # place DejaVuSans font
├── app/
│   ├── streamlit_app.py    # Streamlit demo
│   └── demo_config.py
└── src/
    ├── config.py
    ├── preprocessing.py    # data cleaning pipeline (from raw Excel)
    ├── kde.py              # KDE layer
    ├── llm.py              # LLM layer
    ├── pdf_report.py       # PDF explainability report
    ├── pipeline.py         # end-to-end pipeline
    ├── data/
    ├── models/
    ├── scoring/
    │   └── scores.py       # score helpers
    └── utils/
        ├── text.py
        ├── fonts.py
        └── dates.py
```


## Installation

```bash
pip install -r requirements.txt
```

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
