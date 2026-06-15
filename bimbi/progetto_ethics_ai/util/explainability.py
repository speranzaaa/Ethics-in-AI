from __future__ import annotations
from datetime import datetime

try:
    from fpdf import FPDF
    FPDF_OK = True
except ImportError:
    FPDF_OK = False


RISK_COLORS = {
    "BASSO":      (39, 174, 96),
    "MODERATO":   (241, 196, 15),
    "ALTO":       (230, 126, 34),
    "MOLTO ALTO": (192, 57, 43),
}

CHECKLIST_BASE = [
    "Valutare la coerenza tra il racconto e le lesioni rilevate",
    "Consultare la cartella clinica e lo storico degli accessi",
    "Documentare la valutazione indipendentemente dall'esito",
]
CHECKLIST_ALTO = [
    "Segnalare il caso al medico responsabile di turno",
]
CHECKLIST_MOLTO_ALTO = [
    "Segnalare il caso al medico responsabile di turno",
    "Valutare l'attivazione immediata del protocollo di salvaguardia",
]


def genera_pdf(
    paziente: dict,
    final_score: float,
    risk_level: str,
    risk_action: str,
    indicatori: list[dict],
    kde_score: float,
    has_history: bool,
    num_accessi_90d: int = 0,
    checklist_ai: list[str] | None = None,
) -> bytes:
    """
    Genera un referto PDF clinico leggibile dal personale sanitario.

    Parameters
    ----------
    paziente : dict
        {"eta_in_anni": str, "sesso": str, "gravita": str}
    final_score : float
        Punteggio finale Late Fusion (KDE + LLM).
    risk_level : str
        BASSO / MODERATO / ALTO / MOLTO ALTO
    risk_action : str
        Testo dell'azione raccomandata.
    indicatori : list[dict]
        Indicatori clinici rilevati. Ogni elemento:
        {"descrizione": str, "confidenza": float, "gravita": float}
    kde_score : float
        Score KDE calibrato (0 se storico assente).
    has_history : bool
        True se sono disponibili accessi precedenti.
    num_accessi_90d : int
        Numero di accessi nei 90 giorni precedenti.
    checklist_ai : list[str] | None
        Checklist generata dal modello LLM specifica per il caso.
        Se None, viene usata la checklist standard basata sul livello di rischio.

    Returns
    -------
    bytes
        Contenuto del PDF.
    """
    if not FPDF_OK:
        raise ImportError("fpdf2 non installato. Eseguire: pip install fpdf2")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(10, 10, 10)

    sezione_header(pdf)
    sezione_dati_visita(pdf, paziente)
    sezione_livello_rischio(pdf, risk_level)
    sezione_sintesi(pdf, risk_level, indicatori, has_history, kde_score)
    sezione_indicatori_clinici(pdf, indicatori)
    sezione_pattern_accessi(pdf, has_history, kde_score, num_accessi_90d)
    sezione_checklist(pdf, risk_level, indicatori, checklist_ai)
    sezione_disclaimer(pdf)

    return bytes(pdf.output())


# ---------------------------------------------------------------------------
# Sezioni del PDF
# ---------------------------------------------------------------------------

def sezione_header(pdf: "FPDF") -> None:
    pdf.set_fill_color(44, 62, 80)
    pdf.rect(0, 0, 220, 28, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_xy(10, 7)
    pdf.cell(0, 8, "SISTEMA DI SUPPORTO DECISIONALE PEDIATRICO", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_x(10)
    pdf.cell(0, 5, f"Valutazione del rischio   {datetime.now().strftime('%d/%m/%Y  %H:%M')}", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)


def sezione_dati_visita(pdf: "FPDF", paziente: dict) -> None:
    titolo_sezione(pdf, "DATI VISITA")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6,
        f"Eta: {paziente.get('eta_in_anni', '-')} anni   |   "
        f"Sesso: {paziente.get('sesso', '-')}   |   "
        f"Codice triage: {paziente.get('gravita', '-')}",
        ln=True)
    pdf.ln(4)


def sezione_livello_rischio(pdf: "FPDF", risk_level: str) -> None:
    r, g, b = RISK_COLORS.get(risk_level, (100, 100, 100))
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 12, f"  LIVELLO DI RISCHIO: {risk_level}", ln=True, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)


def sezione_sintesi(
    pdf: "FPDF",
    risk_level: str,
    indicatori: list[dict],
    has_history: bool,
    kde_score: float,
) -> None:
    titolo_sezione(pdf, "SINTESI")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, testo_narrativa(risk_level, indicatori, has_history, kde_score))
    pdf.ln(4)


def sezione_indicatori_clinici(pdf: "FPDF", indicatori: list[dict]) -> None:
    titolo_sezione(pdf, "INDICATORI CLINICI RILEVATI")
    pdf.set_font("Helvetica", "", 10)
    if not indicatori:
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(0, 6, "Nessun indicatore clinico rilevato.", ln=True)
    else:
        for ind in sorted(indicatori, key=lambda x: x.get("gravita", 0), reverse=True):
            conf = ind.get("confidenza", 0.0)
            label = "alta" if conf >= 0.7 else "media" if conf >= 0.4 else "bassa"
            pdf.multi_cell(0, 6, f"  -  {ind.get('descrizione', '')}  (confidenza: {label})")
    pdf.ln(4)


def sezione_pattern_accessi(
    pdf: "FPDF",
    has_history: bool,
    kde_score: float,
    num_accessi_90d: int,
) -> None:
    titolo_sezione(pdf, "PATTERN DI ACCESSO AL PRONTO SOCCORSO")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, testo_pattern(has_history, kde_score, num_accessi_90d))
    pdf.ln(4)


def sezione_checklist(
    pdf: "FPDF",
    risk_level: str,
    indicatori: list[dict],
    checklist_ai: list[str] | None,
) -> None:
    titolo_sezione(pdf, "AZIONI RACCOMANDATE")
    pdf.set_font("Helvetica", "", 10)

    if checklist_ai:
        voci = checklist_ai
    else:
        voci = list(CHECKLIST_BASE)
        if risk_level == "ALTO":
            voci += CHECKLIST_ALTO
        elif risk_level == "MOLTO ALTO":
            voci += CHECKLIST_MOLTO_ALTO
        desc_lower = [ind.get("descrizione", "").lower() for ind in indicatori]
        if any("frattur" in d for d in desc_lower):
            voci.append("Richiedere esame radiologico per escludere fratture occulte")
        if any("trauma cranico" in d for d in desc_lower):
            voci.append("Valutare neuroimaging per trauma cranico")

    for voce in voci:
        pdf.multi_cell(0, 7, f"  [ ]  {voce}")
    pdf.ln(4)


def sezione_disclaimer(pdf: "FPDF") -> None:
    pdf.set_fill_color(245, 245, 245)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 5,
        "AVVISO: Questo documento e generato da un sistema automatico di supporto decisionale. "
        "Non costituisce diagnosi medica, non sostituisce la valutazione del medico responsabile "
        "e non ha valore legale autonomo. La responsabilita della decisione clinica rimane "
        "esclusivamente del professionista sanitario. Il sistema opera secondo i principi "
        "dell'intelligenza artificiale etica: trasparenza, supervisione umana e non maleficenza.",
        fill=True,
    )


# ---------------------------------------------------------------------------
# Helpers interni
# ---------------------------------------------------------------------------

def titolo_sezione(pdf: "FPDF", titolo: str) -> None:
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, titolo, ln=True)
    pdf.set_draw_color(44, 62, 80)
    pdf.set_line_width(0.4)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)


def testo_narrativa(
    risk_level: str,
    indicatori: list[dict],
    has_history: bool,
    kde_score: float,
) -> str:
    n = len(indicatori)
    ind_txt = (
        f" Sono stati identificati {n} indicatori clinici presenti nelle linee guida"
        " per il riconoscimento del maltrattamento pediatrico." if n > 0 else ""
    )
    pattern_txt = (
        " Il pattern di accessi al pronto soccorso risulta statisticamente anomalo"
        " rispetto alla popolazione di riferimento."
        if has_history and kde_score > 0.3 else ""
    )

    if risk_level == "BASSO":
        return (
            "L'analisi non ha rilevato elementi significativi di rischio."
            + (" Il pattern di accessi rientra nei valori attesi e non sono stati"
               " identificati indicatori clinici di allerta." if n == 0
               else f"{ind_txt} Si raccomanda una valutazione ordinaria.")
        )
    if risk_level == "MODERATO":
        return (
            "L'analisi ha rilevato alcuni elementi che meritano attenzione."
            + ind_txt + pattern_txt
            + " Si raccomanda un monitoraggio attento del caso."
        )
    if risk_level == "ALTO":
        return (
            "L'analisi ha rilevato elementi compatibili con una situazione a rischio."
            + ind_txt + pattern_txt
            + " Si raccomanda di segnalare il caso al medico responsabile di turno."
        )
    return (
        "L'analisi ha rilevato elementi fortemente compatibili con una situazione ad alto rischio."
        + ind_txt + pattern_txt
        + " Si raccomanda l'attivazione immediata del protocollo di salvaguardia."
    )


def testo_pattern(has_history: bool, kde_score: float, num_accessi_90d: int) -> str:
    if not has_history:
        return (
            "Non sono disponibili accessi precedenti per questo paziente. "
            "Il pattern temporale non e valutabile con i dati attuali."
        )
    if kde_score == 0.0:
        return (
            "Lo storico degli accessi e presente ma insufficiente per una "
            "valutazione statistica del pattern temporale."
        )
    prefisso = (
        f"Il paziente ha effettuato {num_accessi_90d} accessi negli ultimi 90 giorni. "
        if num_accessi_90d > 0 else ""
    )
    if kde_score < 0.3:
        return prefisso + "La frequenza degli accessi rientra nei valori attesi per la popolazione pediatrica di riferimento."
    if kde_score < 0.7:
        return prefisso + "La frequenza degli accessi e leggermente superiore alla norma per la popolazione pediatrica di riferimento."
    return (
        prefisso
        + "La frequenza degli accessi risulta statisticamente anomala rispetto alla "
        "popolazione pediatrica di riferimento."
    )
