from __future__ import annotations
import os
import re
from datetime import datetime
from typing import Callable

try:
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos
    FPDF_OK = True
except ImportError:
    FPDF_OK = False

# Numero massimo di azioni nella checklist generata dall'LLM.
MAX_VOCI_CHECKLIST = 6

# DejaVu Sans supporta Unicode completo (disponibile su Colab/Linux).
# Su Windows o ambienti senza DejaVu si usa Helvetica (Latin-1).
_DEJAVU = {
    "":  "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "B": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "I": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
}
_FONT = "Body"


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
    regole_violate: list[dict] | None = None,
    llm_generate: Callable[[str], str] | None = None,
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
        Checklist gia' pronta (es. generata altrove). Se valorizzata ha la
        precedenza su qualunque altra generazione.
    regole_violate : list[dict] | None
        Regole violate dal caso. Ogni elemento:
        {"id": str, "descrizione": str, "gravita": float, "confidenza": float}.
        Vengono riportate in una sezione dedicata del referto e usate come
        base per la checklist generata dall'LLM. Se ``indicatori`` non viene
        fornito, viene derivato da questa lista.
    llm_generate : Callable[[str], str] | None
        Funzione che riceve un prompt testuale e restituisce la risposta del
        modello (es. ``lambda p: evaluate_prompt(p, max_new_tokens=250)``).
        Se fornita e ``checklist_ai`` è None, la checklist viene generata
        dall'LLM a partire dalle regole violate. In caso di errore o risposta
        vuota si ricade automaticamente sulla checklist standard.

    Returns
    -------
    bytes
        Contenuto del PDF.
    """
    if not FPDF_OK:
        raise ImportError("fpdf2 non installato. Eseguire: pip install fpdf2")

    regole_violate = regole_violate or []

    # Se gli indicatori non sono forniti, li deriviamo dalle regole violate
    # (una regola violata è di fatto un indicatore clinico rilevato).
    if not indicatori and regole_violate:
        indicatori = [
            {
                "descrizione": r.get("descrizione", r.get("id", "")),
                "confidenza":  r.get("confidenza", 0.0),
                "gravita":     r.get("gravita", 0),
            }
            for r in regole_violate
        ]

    # La checklist viene decisa qui: AI esplicita > LLM dalle regole > statica.
    voci_checklist = costruisci_checklist(
        paziente, risk_level, indicatori, regole_violate,
        checklist_ai, has_history, kde_score, llm_generate,
    )

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(10, 10, 10)

    font = _setup_fonts(pdf)

    sezione_header(pdf, font)
    sezione_dati_visita(pdf, paziente, font)
    sezione_livello_rischio(pdf, risk_level, font)
    sezione_sintesi(pdf, risk_level, indicatori, has_history, kde_score, font)
    sezione_pattern_accessi(pdf, has_history, kde_score, num_accessi_90d, font)
    sezione_regole_violate(pdf, regole_violate, font)
    sezione_checklist(pdf, voci_checklist, font)
    sezione_disclaimer(pdf, font)

    return bytes(pdf.output())


# ---------------------------------------------------------------------------
# Setup font
# ---------------------------------------------------------------------------

def _setup_fonts(pdf: "FPDF") -> str:
    """Carica DejaVu (Unicode) se disponibile, altrimenti usa Helvetica."""
    if os.path.exists(_DEJAVU[""]):
        for style, path in _DEJAVU.items():
            if os.path.exists(path):
                pdf.add_font(_FONT, style=style, fname=path)
        return _FONT
    return "Helvetica"


# ---------------------------------------------------------------------------
# Sezioni del PDF
# ---------------------------------------------------------------------------

def sezione_header(pdf: "FPDF", font: str) -> None:
    pdf.set_fill_color(44, 62, 80)
    pdf.rect(0, 0, 220, 28, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font(font, "B", 14)
    pdf.set_xy(10, 7)
    pdf.cell(0, 8, "SISTEMA DI SUPPORTO DECISIONALE PEDIATRICO", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font(font, "", 9)
    pdf.set_x(10)
    pdf.cell(0, 5, f"Valutazione del rischio   {datetime.now().strftime('%d/%m/%Y  %H:%M')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)


def sezione_dati_visita(pdf: "FPDF", paziente: dict, font: str) -> None:
    titolo_sezione(pdf, "DATI VISITA", font)
    pdf.set_font(font, "", 10)
    eta   = str(paziente.get("eta_in_anni", "-"))
    sesso = str(paziente.get("sesso", "-"))
    grav  = str(paziente.get("gravita", "-"))
    pdf.cell(0, 6, f"Eta: {eta} anni   |   Sesso: {sesso}   |   Codice triage: {grav}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)


def sezione_livello_rischio(pdf: "FPDF", risk_level: str, font: str) -> None:
    r, g, b = RISK_COLORS.get(risk_level, (100, 100, 100))
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font(font, "B", 13)
    pdf.cell(0, 12, f"  LIVELLO DI RISCHIO: {risk_level}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)


def sezione_sintesi(
    pdf: "FPDF",
    risk_level: str,
    indicatori: list[dict],
    has_history: bool,
    kde_score: float,
    font: str,
) -> None:
    titolo_sezione(pdf, "SINTESI", font)
    pdf.set_font(font, "", 10)
    pdf.multi_cell(0, 6, testo_narrativa(risk_level, indicatori, has_history, kde_score), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)


def sezione_indicatori_clinici(pdf: "FPDF", indicatori: list[dict], font: str) -> None:
    titolo_sezione(pdf, "INDICATORI CLINICI RILEVATI", font)
    pdf.set_font(font, "", 10)
    if not indicatori:
        pdf.set_font(font, "I", 10)
        pdf.cell(0, 6, "Nessun indicatore clinico rilevato.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    else:
        for ind in sorted(indicatori, key=lambda x: x.get("gravita", 0), reverse=True):
            conf = ind.get("confidenza", 0.0)
            label = "alta" if conf >= 0.7 else "media" if conf >= 0.4 else "bassa"
            pdf.multi_cell(0, 6, f"  -  {ind.get('descrizione', '')}  (confidenza: {label})", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)


def sezione_pattern_accessi(
    pdf: "FPDF",
    has_history: bool,
    kde_score: float,
    num_accessi_90d: int,
    font: str,
) -> None:
    titolo_sezione(pdf, "PATTERN DI ACCESSO AL PRONTO SOCCORSO", font)
    pdf.set_font(font, "", 10)
    pdf.multi_cell(0, 6, testo_pattern(has_history, kde_score, num_accessi_90d), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)


def sezione_regole_violate(pdf: "FPDF", regole_violate: list[dict], font: str) -> None:
    titolo_sezione(pdf, "INDICATORI CLINICI RILEVATI", font)
    pdf.set_font(font, "I", 9)
    pdf.multi_cell(0, 5,
        "La confidenza indica quanto il modello e' certo che l'indicatore sia presente "
        "nel caso esaminato (0 = incerto, 1 = molto certo).",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT)
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


def sezione_checklist(pdf: "FPDF", voci: list[str], font: str) -> None:
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


def sezione_disclaimer(pdf: "FPDF", font: str) -> None:
    pdf.set_fill_color(245, 245, 245)
    pdf.set_font(font, "I", 8)
    pdf.multi_cell(0, 5,
        "AVVISO: Questo documento è generato da un sistema automatico di supporto decisionale. "
        "Non costituisce diagnosi medica, non sostituisce la valutazione del medico responsabile "
        "e non ha valore legale autonomo. La responsabilita' della decisione clinica rimane "
        "esclusivamente del professionista sanitario. Il sistema opera secondo i principi "
        "dell'intelligenza artificiale etica: trasparenza, supervisione umana e non maleficenza.",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True,
    )


# ---------------------------------------------------------------------------
# Helpers interni
# ---------------------------------------------------------------------------

def costruisci_checklist(
    paziente: dict,
    risk_level: str,
    indicatori: list[dict],
    regole_violate: list[dict],
    checklist_ai: list[str] | None,
    has_history: bool,
    kde_score: float,
    llm_generate: Callable[[str], str] | None,
) -> list[str]:
    """
    Decide quale checklist usare, in ordine di priorita':
      1. checklist_ai gia' fornita dall'esterno;
      2. checklist generata dall'LLM a partire dalle regole violate;
      3. checklist statica basata su livello di rischio e indicatori.
    Qualsiasi errore dell'LLM fa ricadere sulla checklist statica.
    """
    if checklist_ai:
        return list(checklist_ai)

    if llm_generate is not None:
        try:
            voci = genera_checklist_llm(
                paziente, regole_violate or indicatori,
                risk_level, has_history, kde_score, llm_generate,
            )
            if voci:
                return voci
        except Exception:
            pass  # fallback alla checklist statica

    return checklist_statica(risk_level, indicatori)


def checklist_statica(risk_level: str, indicatori: list[dict]) -> list[str]:
    """Checklist deterministica usata come fallback (nessun LLM)."""
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


def genera_checklist_llm(
    paziente: dict,
    regole_violate: list[dict],
    risk_level: str,
    has_history: bool,
    kde_score: float,
    llm_generate: Callable[[str], str],
    max_voci: int = MAX_VOCI_CHECKLIST,
) -> list[str]:
    """Costruisce il prompt dalle regole violate, interroga l'LLM e ripulisce l'output."""
    prompt = _prompt_checklist(paziente, regole_violate, risk_level, has_history, kde_score)
    risposta = llm_generate(prompt)
    return _parse_checklist(risposta, max_voci)


def _prompt_checklist(
    paziente: dict,
    regole_violate: list[dict],
    risk_level: str,
    has_history: bool,
    kde_score: float,
) -> str:
    eta   = paziente.get("eta_in_anni", "-")
    sesso = paziente.get("sesso", "-")
    grav  = paziente.get("gravita", "-")
    righe = [
        f"Paziente: {eta} anni, sesso {sesso}, codice triage {grav}.",
        f"Livello di rischio complessivo: {risk_level}.",
    ]
    if regole_violate:
        righe.append("Regole/indicatori clinici risultati violati per questo caso:")
        for r in regole_violate:
            desc = r.get("descrizione") or r.get("id", "")
            if desc:
                righe.append(f"- {desc}")
    else:
        righe.append("Non sono stati rilevati indicatori clinici specifici.")
    if has_history and kde_score > 0.3:
        righe.append("Il pattern temporale degli accessi risulta anomalo rispetto alla norma.")

    contesto = "\n".join(righe)
    istruzioni = (
        "\n\nSei un medico pediatra esperto in tutela minorile. "
        "Sulla base delle regole/indicatori sopra elencati, genera una checklist clinica "
        f"di massimo {MAX_VOCI_CHECKLIST} azioni concrete e specifiche che il medico deve "
        "verificare per questo caso. Scrivi SOLO le azioni, una per riga, senza numeri, "
        "senza trattini e senza simboli iniziali."
    )
    return contesto + istruzioni


def _parse_checklist(risposta: str, max_voci: int) -> list[str]:
    """Ripulisce la risposta dell'LLM in una lista di azioni."""
    voci = []
    for riga in str(risposta).strip().splitlines():
        riga = riga.strip()
        riga = re.sub(r"^\s*\d+[.)]\s*", "", riga)   # numerazione "1." / "1)"
        riga = riga.lstrip("-*•").strip()             # bullet residui
        if riga:
            voci.append(riga)
    return voci[:max_voci]


def titolo_sezione(pdf: "FPDF", titolo: str, font: str) -> None:
    pdf.set_font(font, "B", 11)
    pdf.cell(0, 7, titolo, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
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
            "Il pattern temporale non è valutabile con i dati attuali."
        )
    if kde_score == 0.0:
        return (
            "Lo storico degli accessi è presente ma insufficiente per una "
            "valutazione statistica del pattern temporale."
        )
    prefisso = (
        f"Il paziente ha effettuato {num_accessi_90d} accessi negli ultimi 90 giorni. "
        if num_accessi_90d > 0 else ""
    )
    if kde_score < 0.3:
        return prefisso + "La frequenza degli accessi rientra nei valori attesi per la popolazione pediatrica di riferimento."
    if kde_score < 0.7:
        return prefisso + "La frequenza degli accessi è leggermente superiore alla norma per la popolazione pediatrica di riferimento."
    return (
        prefisso
        + "La frequenza degli accessi risulta statisticamente anomala rispetto alla "
        "popolazione pediatrica di riferimento."
    )