"""Pagina Calcolatore"""

import sys
import time
from pathlib import Path

import streamlit as st

from src.pdf_report import genera_pdf
from src.scoring import risk_level

_APP_DIR = Path(__file__).resolve().parent.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))
from demo_config import DEFAULT_PATIENT, FIXED_SCORES, HISTORY, VIOLATED_RULES


def show():
    st.title("Calcolatore")
    st.caption("Demo — i punteggi sono simulati.")
    st.markdown("---")

    #session-state defaults
    if "fields" not in st.session_state:
        st.session_state.fields = {k: "" for k in DEFAULT_PATIENT}
    if "scores_ready" not in st.session_state:
        st.session_state.scores_ready = False
    if "risultati" not in st.session_state:
        st.session_state.risultati = {}
    if "storico_trovato" not in st.session_state:
        st.session_state.storico_trovato = False

    #default button
    if st.button("Compila con valori di default", type="secondary"):
        st.session_state.fields = dict(DEFAULT_PATIENT)
        st.session_state.scores_ready = False
        st.session_state.storico_trovato = False
        st.rerun()

    # form
    st.subheader("Dati clinici del paziente")

    with st.form("clinical_form"):
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            eta = st.number_input(
                "Età (anni)",
                min_value=0, max_value=17, step=1,
                value=int(st.session_state.fields.get("eta_in_anni") or 0),
            )
        with col2:
            sesso = st.selectbox(
                "Sesso",
                options=["M", "F"],
                index=["M", "F"].index(
                    st.session_state.fields.get("sesso", "M")
                    if st.session_state.fields.get("sesso", "M") in ["M", "F"]
                    else "M"
                ),
            )
        with col3:
            gravita = st.selectbox(
                "Codice triage (gravità)",
                options=["BIANCO", "VERDE", "GIALLO", "ROSSO"],
                index=["BIANCO", "VERDE", "GIALLO", "ROSSO"].index(
                    st.session_state.fields.get("gravita", "VERDE")
                    if st.session_state.fields.get("gravita", "VERDE")
                       in ["BIANCO", "VERDE", "GIALLO", "ROSSO"]
                    else "VERDE"
                ),
            )

        problema_principale = st.text_input(
            "Problema principale",
            value=str(st.session_state.fields.get("problema_principale", "")),
        )
        dati_riferiti = st.text_area(
            "Dati riferiti",
            value=str(st.session_state.fields.get("dati_riferiti", "")),
            height=80,
        )
        diagnosi = st.text_area(
            "Diagnosi di dimissione",
            value=str(st.session_state.fields.get("diagnosi", "")),
            height=80,
        )
        causale = st.text_input(
            "Causale",
            value=str(st.session_state.fields.get("causale", "")),
        )
        anamnesi = st.text_area(
            "Anamnesi",
            value=str(st.session_state.fields.get("anamnesi", "")),
            height=120,
        )
        note_aggiuntive = st.text_area(
            "Note aggiuntive (esame obiettivo)",
            value=str(st.session_state.fields.get("note_aggiuntive", "")),
            height=100,
        )

        submitted = st.form_submit_button("Calcola", type="primary")

    if submitted:
        st.session_state.fields = {
            "eta_in_anni": eta,
            "sesso": sesso,
            "gravita": gravita,
            "problema_principale": problema_principale,
            "dati_riferiti": dati_riferiti,
            "diagnosi": diagnosi,
            "causale": causale,
            "anamnesi": anamnesi,
            "note_aggiuntive": note_aggiuntive,
        }

        # fake patient history lookup
        with st.spinner("Ricerca storico paziente…"):
            time.sleep(2.0)

        st.session_state.storico_trovato = True
        st.session_state.storico_accessi = HISTORY["num_accessi_90d"]

        #fake score computation
        with st.spinner("Calcolo degli score…"):
            time.sleep(2.0)

        kde_score = FIXED_SCORES["kde_score"]
        llm_score = FIXED_SCORES["llm_score"]
        total_score = kde_score + llm_score
        risk_label, risk_azione = risk_level(total_score)

        st.session_state.risultati = {
            "kde_score":kde_score,
            "llm_score":llm_score,
            "total_score":total_score,
            "risk_label":risk_label,
            "risk_azione":risk_azione,
        }
        st.session_state.scores_ready = True
        st.rerun()

    #  results section
    if st.session_state.scores_ready and st.session_state.risultati:
        r = st.session_state.risultati
        st.markdown("---")

        if st.session_state.storico_trovato:
            st.success(
                f"Storico paziente trovato — "
                f"{st.session_state.storico_accessi} accessi negli ultimi 90 giorni."
            )

        st.subheader("Risultati")
        col_k, col_l, col_t = st.columns(3)
        col_k.metric("KDE score",    f"{r['kde_score']:.3f}")
        col_l.metric("LLM score",    f"{r['llm_score']:.3f}")
        col_t.metric("Score totale", f"{r['total_score']:.3f}")

        COLORI = {
            "BASSO":"green",
            "MODERATO":"orange",
            "ALTO":"red",
            "MOLTO ALTO":"red",
        }
        colore = COLORI.get(r["risk_label"], "gray")
        st.markdown(f"### Livello di rischio: :{colore}[**{r['risk_label']}**]")
        if r["risk_azione"]:
            st.info(r["risk_azione"])

        st.markdown("---")
        st.subheader("Scarica il referto PDF")

        pdf_bytes = genera_pdf(
            paziente=dict(st.session_state.fields),
            risk_level=r["risk_label"],
            kde_score=r["kde_score"],
            has_history=HISTORY["has_history"],
            num_accessi_90d=HISTORY["num_accessi_90d"],
            regole_violate=VIOLATED_RULES,
        )

        st.download_button(
            label="Scarica PDF",
            data=pdf_bytes,
            file_name="referto_cdss.pdf",
            mime="application/pdf",
            type="primary",
        )
