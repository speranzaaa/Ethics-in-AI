from datetime import datetime

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from .utils.fonts import setup_fonts


MAX_VOCI_CHECKLIST = 6

RISK_COLORS = {
    "BASSO": (39, 174, 96),
    "MODERATO": (241, 196, 15),
    "ALTO": (230, 126, 34),
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


# Generate pDF for one visit

def genera_pdf(
    paziente,
    risk_level,
    kde_score,
    has_history,
    num_accessi_90d=0,
    regole_violate=None,
):
    if regole_violate is None:
        regole_violate = []

    indicatori = [
        {
            "descrizione": r.get("descrizione", r.get("id", "")),
            "confidenza": r.get("confidenza", 0.0),
            "gravita": r.get("gravita", 0),
        }
        for r in regole_violate
    ]

    voci_checklist = checklist_statica(risk_level, indicatori)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(10, 10, 10)

    font = setup_fonts(pdf)

    sezione_header(pdf, font)
    sezione_dati_visita(pdf, paziente, font)
    sezione_livello_rischio(pdf, risk_level, font)
    sezione_sintesi(pdf, risk_level, indicatori, has_history, kde_score, font)
    sezione_pattern_accessi(pdf, has_history, kde_score, num_accessi_90d, font)
    sezione_regole_violate(pdf, regole_violate, font)
    sezione_checklist(pdf, voci_checklist, font)
    sezione_disclaimer(pdf, font)

    return bytes(pdf.output())


# PDF sections

def sezione_header(pdf, font):
    pdf.set_fill_color(44, 62, 80)
    pdf.rect(0, 0, 220, 28, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font(font, "B", 14)
    pdf.set_xy(10, 7)
    pdf.cell(
        0, 8,
        "SISTEMA DI SUPPORTO DECISIONALE PEDIATRICO",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )
    pdf.set_font(font, "", 9)
    pdf.set_x(10)
    pdf.cell(
        0, 5,
        f"Valutazione del rischio   {datetime.now().strftime('%d/%m/%Y  %H:%M')}",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)


def sezione_dati_visita(pdf, paziente, font):
    titolo_sezione(pdf, "DATI VISITA", font)
    pdf.set_font(font, "", 10)
    eta   = str(paziente.get("eta_in_anni", "-"))
    sesso = str(paziente.get("sesso", "-"))
    grav  = str(paziente.get("gravita", "-"))
    pdf.cell(
        0, 6,
        f"Eta: {eta} anni   |   Sesso: {sesso}   |   Codice triage: {grav}",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )
    pdf.ln(4)


def sezione_livello_rischio(pdf, risk_level, font):
    r, g, b = RISK_COLORS.get(risk_level, (100, 100, 100))
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font(font, "B", 13)
    pdf.cell(
        0, 12,
        f"  LIVELLO DI RISCHIO: {risk_level}",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True,
    )
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)


def sezione_sintesi(pdf, risk_level, indicatori, has_history, kde_score, font):
    titolo_sezione(pdf, "SINTESI", font)
    pdf.set_font(font, "", 10)
    pdf.multi_cell(
        0, 6,
        testo_narrativa(risk_level, indicatori, has_history, kde_score),
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )
    pdf.ln(4)


def sezione_indicatori_clinici(pdf, indicatori, font):
    titolo_sezione(pdf, "INDICATORI CLINICI RILEVATI", font)
    pdf.set_font(font, "", 10)
    if not indicatori:
        pdf.set_font(font, "I", 10)
        pdf.cell(0, 6, "Nessun indicatore clinico rilevato.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    else:
        for ind in sorted(indicatori, key=lambda x: x.get("gravita", 0), reverse=True):
            conf = ind.get("confidenza", 0.0)
            label = "alta" if conf >= 0.7 else "media" if conf >= 0.4 else "bassa"
            pdf.multi_cell(
                0, 6,
                f"  -  {ind.get('descrizione', '')}  (confidenza: {label})",
                new_x=XPos.LMARGIN, new_y=YPos.NEXT,
            )
    pdf.ln(4)


def sezione_pattern_accessi(pdf, has_history, kde_score, num_accessi_90d, font):
    titolo_sezione(pdf, "PATTERN DI ACCESSO AL PRONTO SOCCORSO", font)
    pdf.set_font(font, "", 10)
    pdf.multi_cell(
        0, 6,
        testo_pattern(has_history, kde_score, num_accessi_90d),
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )
    pdf.ln(4)


def sezione_regole_violate(pdf, regole_violate, font):
    titolo_sezione(pdf, "INDICATORI CLINICI RILEVATI", font)
    pdf.set_font(font, "I", 9)
    pdf.multi_cell(
        0, 5,
        "La confidenza indica quanto il modello e' certo che l'indicatore sia presente "
        "nel caso esaminato (0 = incerto, 1 = molto certo).",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )
    pdf.ln(3)
    pdf.set_font(font, "", 10)
    if not regole_violate:
        pdf.set_font(font, "I", 10)
        pdf.cell(0, 6, "Nessun indicatore clinico rilevato.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(4)
        return
    for r in sorted(regole_violate, key=lambda x: x.get("confidenza", 0.0), reverse=True):
        desc = r.get("descrizione", "")
        conf = r.get("confidenza", 0.0)
        try:
            conf_str = f"{float(conf):.2f}"
        except (TypeError, ValueError):
            conf_str = str(conf)
        pdf.set_font(font, "B", 10)
        pdf.multi_cell(0, 6, f"Confidenza {conf_str}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        if desc:
            pdf.set_font(font, "", 10)
            pdf.multi_cell(0, 6, f"     {desc}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)
    pdf.ln(3)


def sezione_checklist(pdf, voci, font):
    titolo_sezione(pdf, "AZIONI RACCOMANDATE", font)
    pdf.set_font(font, "", 10)
    if not voci:
        pdf.set_font(font, "I", 10)
        pdf.cell(0, 6, "Nessuna azione specifica generata.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(4)
        return
    for voce in voci:
        pdf.multi_cell(0, 7, f"  [ ]  {voce}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)


def sezione_disclaimer(pdf, font):
    pdf.set_fill_color(245, 245, 245)
    pdf.set_font(font, "I", 8)
    pdf.multi_cell(
        0, 5,
        "AVVISO: Questo documento è generato da un sistema automatico di supporto decisionale. "
        "Non costituisce diagnosi medica, non sostituisce la valutazione del medico responsabile "
        "e non ha valore legale autonomo. La responsabilita' della decisione clinica rimane "
        "esclusivamente del professionista sanitario. Il sistema opera secondo i principi "
        "dell'intelligenza artificiale etica: trasparenza, supervisione umana e non maleficenza.",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True,
    )


# Internal helpers

def checklist_statica(risk_level, indicatori):
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
    return voci


def titolo_sezione(pdf, titolo, font):
    pdf.set_font(font, "B", 11)
    pdf.cell(0, 7, titolo, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_draw_color(44, 62, 80)
    pdf.set_line_width(0.4)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)


def testo_narrativa(risk_level, indicatori, has_history, kde_score):
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


def testo_pattern(has_history, kde_score, num_accessi_90d):
    if not has_history:
        return (
            "Non sono disponibili accessi precedenti per questo paziente. "
            "Il pattern temporale non e' valutabile con i dati attuali."
        )
    if kde_score == 0.0:
        return (
            "Lo storico degli accessi e' presente ma insufficiente per una "
            "valutazione statistica del pattern temporale."
        )
    prefisso = (
        f"Il paziente ha effettuato {num_accessi_90d} accessi negli ultimi 90 giorni. "
        if num_accessi_90d > 0 else ""
    )
    if kde_score < 0.3:
        return (
            prefisso
            + "La frequenza degli accessi rientra nei valori attesi per la popolazione "
            "pediatrica di riferimento."
        )
    if kde_score < 0.7:
        return (
            prefisso
            + "La frequenza degli accessi e' leggermente superiore alla norma per la "
            "popolazione pediatrica di riferimento."
        )
    return (
        prefisso
        + "La frequenza degli accessi risulta statisticamente anomala rispetto alla "
        "popolazione pediatrica di riferimento."
    )
