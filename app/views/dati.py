"""Pagina Dati"""

import ast
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from src.config import (
    ID_COL, DATE_COL, COL_MAP,
    TEST_LLM_PATH, TEST_KDE_PATH, TEST_RESULTS_PATH, RULE_PATH,
)
from src.scoring import risk_level

_ASSETS_IMAGES = Path(__file__).resolve().parent.parent.parent / "assets" / "images"

# Campi clinici inviati all'LLM (dal COL_MAP + eta calcolata)
_CAMPI_ETICHETTE = {
    "eta_in_anni": "Età (anni)",
    "sesso": "Sesso",
    "problema_principale": "Problema principale",
    "dati_riferiti": "Dati riferiti",
    "diagnosi": "Diagnosi",
    "causale": "Causale",
    "anamnesi": "Anamnesi",
    "note_aggiuntive": "Note aggiuntive",
}


# Data loaders

@st.cache_data(show_spinner=False)
def carica_parquet(path_str: str):
    path = Path(path_str)
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def carica_regole(path_str):
    path = Path(path_str)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {r["id"]: r for r in data.get("regole", [])}


def parse_regole_violate(raw):
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    try:
        return ast.literal_eval(str(raw))
    except (ValueError, SyntaxError):
        return []


# Sezioni della vista "singolo caso"

def mostra_caso(patient_id: str):
    df_llm = carica_parquet(str(TEST_LLM_PATH))
    df_kde = carica_parquet(str(TEST_KDE_PATH))
    df_res = carica_parquet(str(TEST_RESULTS_PATH))
    regole_by_id = carica_regole(str(RULE_PATH))

    # layer LLM
    st.subheader("Layer LLM")

    llm_score_val = None
    if df_llm is None:
        st.warning("File `test_LLM.parquet` non trovato in `src/data/`.")
    else:
        righe = df_llm[df_llm[ID_COL] == patient_id]
        if righe.empty:
            st.warning(f"Nessuna visita trovata per {patient_id}.")
        else:
            visita = righe.sort_values(DATE_COL).iloc[-1] if DATE_COL in righe.columns else righe.iloc[-1]

            st.markdown("**Dati clinici inviati al modello LLM:**")
            prompt_data: dict = {}
            for campo, col in COL_MAP.items():
                if col in visita.index:
                    prompt_data[campo] = visita[col]
            # Usa ETA_ALLA_VISITA già calcolata nel parquet
            if "ETA_ALLA_VISITA" in visita.index:
                prompt_data["eta_in_anni"] = int(visita["ETA_ALLA_VISITA"])

            for chiave, etichetta in _CAMPI_ETICHETTE.items():
                val = prompt_data.get(chiave, "N/D")
                if isinstance(val, str) and len(val) > 200:
                    with st.expander(f"**{etichetta}**"):
                        st.write(val)
                else:
                    st.markdown(f"- **{etichetta}**: {val}")

    # Regole violate
    st.markdown("**Regole violate:**")
    if df_res is not None and ID_COL in df_res.columns and "regole_violate" in df_res.columns:
        riga_res = df_res[df_res[ID_COL] == patient_id]
        if not riga_res.empty:
            raw_rv = riga_res["regole_violate"].iloc[0]
            regole_lista = parse_regole_violate(raw_rv)
            if not regole_lista:
                st.info("Nessuna regola violata per questo caso.")
            else:
                for rv in sorted(regole_lista, key=lambda x: x.get("conf", 0), reverse=True):
                    rule_id = rv.get("id")
                    regola = regole_by_id.get(rule_id, {})
                    descrizione = regola.get("descrizione", f"Regola #{rule_id}")
                    grav = rv.get("grav", rv.get("gravita", 0))
                    conf = rv.get("conf", rv.get("confidenza", 0))
                    st.markdown(
                        f"- **ID {rule_id}** — Confidenza: `{conf:.2f}` | Gravità: `{grav:.2f}`  \n"
                        f"  {descrizione}"
                    )
        else:
            st.info(f"Nessun risultato trovato per {patient_id} in test_results.parquet.")
    else:
        st.info("Colonna `regole_violate` non disponibile.")

    # LLM score
    if df_res is not None and "llm_score" in df_res.columns and ID_COL in df_res.columns:
        riga_res = df_res[df_res[ID_COL] == patient_id]
        if not riga_res.empty:
            llm_score_val = float(riga_res["llm_score"].iloc[0])
            st.metric("LLM score", f"{llm_score_val:.4f}")
    elif df_res is None:
        st.warning("File `test_results.parquet` non trovato.")

    st.markdown("---")

    # layer KDE 
    st.subheader("Layer KDE")

    kde_score_val = None
    _KDE_TIME_COL = "DATAORA_ACCETTAZIONE"  # nome nel parquet KDE

    if df_kde is None:
        st.warning("File `test_KDE.parquet` non trovato in `src/data/`.")
    else:
        kde_id_col = ID_COL if ID_COL in df_kde.columns else None
        righe_kde = df_kde[df_kde[kde_id_col] == patient_id] if kde_id_col else pd.DataFrame()
        if righe_kde.empty:
            st.warning(f"Nessuna finestra KDE trovata per {patient_id}.")
        else:
            excl = {kde_id_col, _KDE_TIME_COL, "tipologia"}
            feature_cols = [c for c in righe_kde.columns if c not in excl]
            st.markdown("**Feature della finestra temporale:**")
            st.dataframe(
                righe_kde[feature_cols].reset_index(drop=True),
                use_container_width=True,
            )

    if df_res is not None and "kde_score" in df_res.columns and ID_COL in df_res.columns:
        riga_res = df_res[df_res[ID_COL] == patient_id]
        if not riga_res.empty:
            kde_score_val = float(riga_res["kde_score"].iloc[0])
            st.metric("KDE score", f"{kde_score_val:.4f}")

    # riepilogo
    if llm_score_val is not None and kde_score_val is not None:
        st.markdown("---")
        st.subheader("Riepilogo")
        total = llm_score_val + kde_score_val
        label, azione = risk_level(total)
        col1, col2 = st.columns(2)
        col1.metric("Score totale", f"{total:.4f}")
        col2.markdown(f"**Livello di rischio:** `{label}`")
        if azione:
            st.info(azione)


# sezione "Caso totale"

def mostra_caso_totale():
    from data_config import SYSTEM_OVERVIEW, CHARTS

    if SYSTEM_OVERVIEW:
        st.markdown(SYSTEM_OVERVIEW)
        st.markdown("---")

    for chart in CHARTS:
        titolo = chart.get("titolo", "")
        descrizione = chart.get("descrizione", "")
        img_name = chart.get("image", "")
        img_path = _ASSETS_IMAGES / img_name

        st.subheader(titolo)
        
        st.image(str(img_path))
        
        if descrizione:
            st.caption(descrizione)
        st.markdown("---")



def show():
    st.title("Dati")
    st.caption("Esplora i risultati sui casi di test o la valutazione complessiva del sistema.")
    st.markdown("---")

    # Build patient ID list from test_LLM.parquet
    df_llm = carica_parquet(str(TEST_LLM_PATH))
    if df_llm is not None and ID_COL in df_llm.columns:
        ids_pazienti = sorted(df_llm[ID_COL].dropna().unique().tolist())
    else:
        ids_pazienti = []

    opzioni = ["Caso totale"] + ids_pazienti
    scelta = st.selectbox("Seleziona un caso", options=opzioni)
    st.markdown("---")

    if scelta == "Caso totale":
        mostra_caso_totale()
    else:
        st.subheader(f"Paziente: {scelta}")
        mostra_caso(scelta)
