"""Demo simulation - all values are already setted"""

# ---------------------------------------------------------------------------
# Default clinical fields
# Clicking "Compila con valori di default" fills the form with these values.
# ---------------------------------------------------------------------------

DEFAULT_PATIENT = {
    "eta_in_anni": 4,

    "sesso": "M",

    #(BIANCO, VERDE, GIALLO, ROSSO)
    "gravita": "GIALLO",

    "problema_principale": "Trauma cranico e lesioni multiple agli arti",

    # Testo libero riferito dal genitore/accompagnatore al momento dell'accesso
    "dati_riferiti": (
        "I genitori riferiscono caduta accidentale dal lettino. "
        "Il bambino ha pianto subito dopo. "
        "Non riferiscono perdita di coscienza."
    ),

    "diagnosi": (
        "Trauma cranico minore. Ecchimosi multiple agli arti superiori e inferiori. "
        "Frattura costale destra in fase di consolidamento."
    ),

    #(MALATTIA, TRAUMA, AGGRESSIONE, ALTRO, ...)
    "causale": "TRAUMA",

    "anamnesi": (
        "Bambino di 4 anni giunge in PS accompagnato dai genitori per trauma cranico "
        "riferito come caduta dal lettino. All'esame obiettivo si rilevano ecchimosi "
        "multiple in diverse fasi di evoluzione. Il genitore non sa spiegare l'origine "
        "di alcune lesioni. Molteplici accessi in PS negli ultimi mesi per traumi."
    ),

    "note_aggiuntive": (
        "Bambino vigile e orientato. GCS 15. Ecchimosi a diversi stadi di risoluzione "
        "su tronco e arti. Dolorabilita' alla palpazione dell'emitorace destro. "
        "Dinamica riferita non pienamente compatibile con le lesioni rilevate."
    ),
}

# Fixed scores 

FIXED_SCORES = {
    "kde_score": 0.62, 
    "llm_score": 0.69, 
}

# Patient history shown in the demo

HISTORY = {
    "has_history": True,
    "num_accessi_90d": 4,
}

# Violated rules

VIOLATED_RULES = [
    {
        "id": "R045",
        "descrizione": (
            "Presenza di lesioni cutanee multiple in diverse fasi di evoluzione "
            "non giustificate dall'anamnesi riferita."
        ),
        "gravita": 0.85,
        "confidenza": 0.90,
    },
    {
        "id": "R112",
        "descrizione": (
            "Dinamica del trauma riferita non compatibile con le lesioni rilevate "
            "all'esame obiettivo."
        ),
        "gravita": 0.75,
        "confidenza": 0.82,
    },
    {
        "id": "R203",
        "descrizione": (
            "Accessi ripetuti al pronto soccorso per traumi in un breve arco temporale "
            "(>= 3 accessi in 90 giorni)."
        ),
        "gravita": 0.65,
        "confidenza": 0.78,
    },
    {
        "id": "R067",
        "descrizione": (
            "Frattura in fase di consolidamento non segnalata in precedenza "
            "e non documentata da accessi antecedenti."
        ),
        "gravita": 0.80,
        "confidenza": 0.70,
    },
]
