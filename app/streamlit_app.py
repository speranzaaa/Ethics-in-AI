import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st

from views import home, calcolatore, dati

# Page config

st.set_page_config(
    page_title="ETHICS — Supporto Decisionale Pediatrico",
    page_icon="🏥",
    layout="centered",
)

# Session-state initialization

if "boot_done" not in st.session_state:
    st.session_state.boot_done = False

# Boot sequence 

if not st.session_state.boot_done:
    st.title("GUARDIAN\n Guided Understanding And Risk Detection In Abuse Network")
    st.markdown("---")

    placeholder = st.empty()
    with placeholder.container():
        st.info("Avvio del sistema in corso…")
        progress = st.progress(0, text="Inizializzazione…")

        time.sleep(0.5)
        progress.progress(10, text="Caricamento modello KDE…")
        time.sleep(0.6)
        progress.progress(15, text="Caricamento modello KDE…")
        time.sleep(0.3)
        progress.progress(25, text="Caricamento modello KDE…")
        time.sleep(0.8)
        progress.progress(40, text="Caricamento modello LLM…")
        time.sleep(0.4)
        progress.progress(45, text="Caricamento modello LLM…")
        time.sleep(0.3)
        progress.progress(65, text="Caricamento modello LLM…")
        time.sleep(0.5)
        progress.progress(70, text="Caricamento modello LLM…")
        time.sleep(0.4)
        progress.progress(85, text="Caricamento modello LLM…")
        time.sleep(0.6)
        progress.progress(90, text="Caricamento dati...")
        time.sleep(0.6)
        progress.progress(95, text="Caricamento dati...")
        time.sleep(0.9)
        progress.progress(100, text="Sistema pronto.")
        time.sleep(1.0)

    placeholder.empty()
    st.session_state.boot_done = True
    st.rerun()

# Multipage navigation

pg_home = st.Page(home.show, title="Home", icon="🏠", url_path="home", default=True)
pg_calcolatore = st.Page(calcolatore.show, title="Calcolatore", icon="🩺", url_path="calcolatore")
pg_dati = st.Page(dati.show, title="Dati", icon="📊", url_path="dati")



st.session_state["_pg_calcolatore"] = pg_calcolatore
st.session_state["_pg_dati"] = pg_dati
st.session_state["_pg_home"] = pg_home

pg = st.navigation([pg_home, pg_calcolatore, pg_dati])
pg.run()
