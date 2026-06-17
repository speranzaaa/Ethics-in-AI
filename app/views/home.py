"""Pagina Home"""

import streamlit as st

def show():
    st.title("GUARDIAN\n Guided Understanding And Risk Detection In Abuse Network")
    st.caption(
        "Supporto alla valutazione del rischio di patologia non accidentale (NAP) "
        "in pronto soccorso pediatrico."
    )
    st.markdown("---")

    st.markdown("### Scegli una sezione")
    col_a, col_b = st.columns(2)

    with col_a:
        if st.button(
            "Calcolatore",
            use_container_width=True,
            type="secondary",
        ):
            st.switch_page(st.session_state["_pg_calcolatore"])

    with col_b:
        if st.button(
            "Dati",
            use_container_width=True,
            type="secondary",
        ):
            st.switch_page(st.session_state["_pg_dati"])
