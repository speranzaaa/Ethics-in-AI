"""
knowledge_extraction.py
~~~~~~~~~~~~~~~~~~~~~~~
Knowledge Extraction module for the Ethical CDSS.

This module implements a two-stage knowledge pipeline that bridges unstructured
Italian medical literature and the CDSS's anomaly detection engine:

  Stage 1 - PDF Parsing:
    Raw text extraction from Italian medical guideline PDFs
    ("Quaderni della Regione Emilia-Romagna") using PyPDF2.

  Stage 2 - LLM Rule Mining (two parallel backends):
    Two functionally equivalent extractors are provided so their output can
    be compared for completeness, accuracy, and Italian terminology fidelity:

      extract_rules_openai()        -- cloud API via OpenAI (gpt-4o-mini default)
      extract_rules_local_ollama()  -- local inference via Ollama (llama3 default)

    Both extractors share the same Italian-language system prompt and the same
    output schema, enabling a direct apples-to-apples comparison.

NEURO-SYMBOLIC FUSION ARCHITECTURE:
  The extracted rules constitute the SYMBOLIC AI layer of the CDSS.  They
  encode expert clinical constraints -- expressed in their original Italian
  terminology -- that are injected as "Prior Knowledge" into the SUB-SYMBOLIC
  AI component: the multivariate Kernel Density Estimator (KDE).

  Formally, the fused anomaly score for patient record x is:

      score(x) = -log P_KDE(x) + SUM_i  penalty_weight_i * I[condition_i in x]

  where:
    - -log P_KDE(x)  is the KDE negative log-likelihood (statistical signal).
    - condition_i    is an Italian clinical string (e.g., "fratture multiple").
    - penalty_weight_i is a domain-calibrated additive penalty.
    - I[...]         is the indicator function (1 if substring found, else 0).

  This hybrid design gives the system two complementary detection mechanisms:
    - Sub-symbolic (KDE): detects statistically unusual temporal ER patterns
      without requiring prior labelling of abuse cases.
    - Symbolic (rules): flags clinically known red-flag presentations that
      may appear in any single visit and might not be rare in the training
      data (e.g., a child with documented burns in a hospital that rarely
      encounters abuse would not be flagged by KDE alone).

CRITICAL DESIGN INVARIANT -- Italian Terminology Preservation:
  Both LLM prompts explicitly instruct the model to extract and retain the
  EXACT Italian medical terms as they appear in the source PDFs.  This is
  mandatory and non-negotiable:

    - Our clinical CSV datasets (Triage, Sintomi, Dati clinici) are in Italian.
    - Rule matching in EthicalKDEAnomalyDetector._evaluate_rule is performed
      via case-insensitive substring search against those Italian text fields.
    - Translating "fratture multiple" to "multiple fractures" would silently
      break the matching pipeline and reduce system sensitivity to zero for
      that rule.  This is a patient-safety risk, not a cosmetic issue.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import ollama
import openai
import PyPDF2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PDF Extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract the full text content from a PDF file using PyPDF2.

    Reads every page of the PDF and concatenates the extracted text,
    separated by newlines.  Designed specifically for Italian medical
    guideline PDFs ("Quaderni della Regione Emilia-Romagna") whose text
    layer is selectable (not scanned images).

    Parameters
    ----------
    pdf_path:
        Absolute or relative file-system path to the PDF.

    Returns
    -------
    str
        Concatenated text from all pages.  Empty pages contribute an
        empty string; they do not raise an exception.

    Raises
    ------
    FileNotFoundError
        If no file exists at ``pdf_path``.
    PyPDF2.errors.PdfReadError
        If the file cannot be parsed as a valid PDF.
    """
    logger.info("Extracting text from PDF: %s", pdf_path)
    pages: list[str] = []

    with open(pdf_path, "rb") as fh:
        reader = PyPDF2.PdfReader(fh)
        n_pages = len(reader.pages)
        logger.info("PDF has %d pages.", n_pages)

        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            pages.append(page_text)
            logger.debug(
                "Page %d/%d: extracted %d characters.",
                page_num + 1,
                n_pages,
                len(page_text),
            )

    full_text = "\n".join(pages)
    logger.info(
        "PDF extraction complete -- %d pages, %d total characters.",
        len(pages),
        len(full_text),
    )
    return full_text


# ---------------------------------------------------------------------------
# Shared prompt and parsing logic
# ---------------------------------------------------------------------------

# This prompt is used by BOTH extractors.  Keeping it identical is essential
# for a fair comparison: any difference in rule quality is attributable to
# the model, not to differences in instruction.
_EXTRACTION_SYSTEM_PROMPT = """
Sei un ingegnere della conoscenza medica specializzato nel rilevamento del maltrattamento infantile.
Il tuo compito e' estrarre regole cliniche strutturate e leggibili da macchina da linee guida mediche italiane.

ISTRUZIONE CRITICA -- CONSERVAZIONE DELLA TERMINOLOGIA ITALIANA:
Devi SEMPRE estrarre e conservare la terminologia medica italiana ESATTA cosi' come appare nel testo sorgente.
NON tradurre MAI i termini medici in inglese.
Il campo "clinical_conditions" DEVE contenere le stringhe italiane originali, ad esempio:
  "fratture multiple", "lesioni cutanee", "ustioni", "ematomi", "ecchimosi",
  "trauma cranico", "abuso sessuale", "trascuratezza", "ritardo di accesso alle cure",
  "lesioni perianali", "lesioni genitali", "sindrome del bambino scosso".
Queste stringhe verranno confrontate direttamente con campi di testo italiano
nei dataset CSV clinici del sistema CDSS (Triage, Sintomi, Dati clinici).
La traduzione in inglese causerebbe il MANCATO RILEVAMENTO di condizioni cliniche
reali e costituisce un rischio per la sicurezza del paziente.

Restituisci SOLO un oggetto JSON con una chiave "rules" contenente un array.
Ogni elemento dell'array deve avere ESATTAMENTE queste chiavi:
  "rule_id"             -- identificatore univoco stringa (es. "R001", "R002")
  "description"         -- una frase concisa che spiega la logica clinica
  "clinical_conditions" -- stringa ITALIANA della condizione clinica o sintomo esatto
                           (DEVE essere in italiano, non tradotta)
  "penalty_weight"      -- float: incremento allo score di anomalia KDE se questa
                           regola viene violata (valori tipici 0.5-3.0;
                           usare valori piu' alti per segnali di abuso piu' forti)

Esempio di output corretto:
{
  "rules": [
    {
      "rule_id": "R001",
      "description": "Multiple fractures in a child are a strong red flag for non-accidental injury.",
      "clinical_conditions": "fratture multiple",
      "penalty_weight": 2.5
    },
    {
      "rule_id": "R002",
      "description": "Skin lesions inconsistent with the reported mechanism of injury.",
      "clinical_conditions": "lesioni cutanee",
      "penalty_weight": 1.8
    },
    {
      "rule_id": "R003",
      "description": "Burns in a child, especially patterned or immersion burns, are highly suspicious.",
      "clinical_conditions": "ustioni",
      "penalty_weight": 2.0
    }
  ]
}
""".strip()

_USER_MESSAGE_TEMPLATE = (
    "Analizza il seguente testo estratto da linee guida mediche italiane "
    "sul maltrattamento e abuso infantile. "
    "Estrai TUTTE le regole cliniche rilevanti come specificato. "
    "RICORDA: il campo 'clinical_conditions' DEVE contenere la terminologia "
    "medica italiana originale -- mai traduzioni in inglese.\n\n"
    "TESTO DELLE LINEE GUIDA:\n{excerpt}"
)


def _parse_and_validate_rules(raw: str, source: str) -> list[dict[str, Any]]:
    """
    Parse a raw JSON string from an LLM response and validate its schema.

    Shared by both extractors so that validation logic is not duplicated.
    Handles both bare JSON arrays and the canonical ``{"rules": [...]}`` envelope.

    Parameters
    ----------
    raw:
        Raw string returned by the LLM, expected to contain JSON.
    source:
        Label identifying the backend (e.g., "OpenAI", "Ollama") used only
        for log messages.

    Returns
    -------
    list[dict]
        Validated and type-coerced rule dictionaries.  Returns an empty list
        on any parse or schema error rather than propagating exceptions, so
        that a single malformed response does not abort the extraction pipeline.
    """
    raw = raw.strip()

    # Some models wrap their output in markdown fences -- strip them.
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error(
            "[%s] Failed to parse JSON response: %s\nRaw (first 500 chars): %s",
            source, exc, raw[:500],
        )
        return []

    # Accept both {"rules": [...]} and a bare [...].
    if isinstance(parsed, list):
        raw_rules: list = parsed
    elif isinstance(parsed, dict):
        raw_rules = next(
            (v for v in parsed.values() if isinstance(v, list)),
            [],
        )
    else:
        logger.warning("[%s] Unexpected JSON root type: %s", source, type(parsed))
        return []

    validated: list[dict[str, Any]] = []
    for i, rule in enumerate(raw_rules):
        if not isinstance(rule, dict):
            logger.warning("[%s] Skipping non-dict item at index %d: %r", source, i, rule)
            continue
        validated.append(
            {
                "rule_id":             str(rule.get("rule_id", f"R{i + 1:03d}")),
                "description":         str(rule.get("description", "")),
                "clinical_conditions": str(rule.get("clinical_conditions", "")),
                "penalty_weight":      float(rule.get("penalty_weight", 1.0)),
            }
        )

    logger.info(
        "[%s] Validated %d clinical rules (Italian terminology preserved).",
        source,
        len(validated),
    )
    return validated


# ---------------------------------------------------------------------------
# Extractor 1: OpenAI cloud API
# ---------------------------------------------------------------------------

def extract_rules_openai(
    text: str,
    model: str = "gpt-4o-mini",
    max_text_chars: int = 12000,
) -> list[dict[str, Any]]:
    """
    Extract formal clinical rules from Italian guideline text using the OpenAI API.

    This is the CLOUD-BASED extractor.  It sends the guideline text to OpenAI
    with ``response_format={"type": "json_object"}`` to guarantee structured output.
    Use this backend when accuracy and consistency are the priority.

    NEURO-SYMBOLIC FUSION ROLE:
      The returned rules constitute the SYMBOLIC AI layer of the CDSS.  Loaded
      into EthicalKDEAnomalyDetector via set_prior_knowledge(), they penalise the
      KDE anomaly score whenever a patient record contains a matching Italian string:

          score(x) = -log P_KDE(x) + SUM_i penalty_weight_i * I[condition_i in x]

    ITALIAN TERMINOLOGY INVARIANT:
      The system prompt instructs the LLM to retain exact Italian medical terms
      (e.g., "fratture multiple", "lesioni cutanee").  These are matched
      case-insensitively against the Italian-language fields in the clinical CSV
      datasets.  Any translation would silently break the downstream matching chain.

    Parameters
    ----------
    text:
        Raw Italian text from a clinical guideline PDF, typically obtained via
        extract_text_from_pdf().
    model:
        OpenAI model identifier.  Must support json_object response format.
    max_text_chars:
        Maximum characters of ``text`` sent to the API to stay within token limits.

    Returns
    -------
    list[dict]
        Validated rule dicts with keys: rule_id, description,
        clinical_conditions (Italian), penalty_weight.

    Raises
    ------
    EnvironmentError
        If OPENAI_API_KEY is not set in the environment.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it to your .env file."
        )
    client = openai.OpenAI(api_key=api_key)

    excerpt = text[:max_text_chars]
    user_message = _USER_MESSAGE_TEMPLATE.format(excerpt=excerpt)

    logger.info(
        "[OpenAI] Calling model=%s to extract clinical rules from %d chars.",
        model,
        len(excerpt),
    )

    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.0,
        max_tokens=2000,
    )

    raw: str = response.choices[0].message.content or "{}"
    return _parse_and_validate_rules(raw, source="OpenAI")


# ---------------------------------------------------------------------------
# Extractor 2: Local Ollama (open-weight models)
# ---------------------------------------------------------------------------

def extract_rules_local_ollama(
    text: str,
    model_name: str = "llama3",
    max_text_chars: int = 12000,
) -> list[dict[str, Any]]:
    """
    Extract formal clinical rules from Italian guideline text using a local Ollama model.

    This is the LOCAL-INFERENCE extractor, provided for technical comparison and
    local testing.  It runs entirely on-device with no data leaving the machine,
    making it suitable for sensitive clinical environments where sending patient
    data to a cloud API is not permitted.

    The extraction logic, system prompt, and output schema are IDENTICAL to those
    of extract_rules_openai().  This allows a direct performance comparison between
    a cloud frontier model (OpenAI) and a local open-weight model (e.g., Llama 3)
    on the specific task of Italian medical terminology extraction.  Differences
    in output quality, rule completeness, and terminology fidelity are attributable
    solely to the model, not to any difference in instructions.

    Ollama is invoked with ``format="json"`` to constrain its output to valid JSON.
    The Ollama server must be running locally (default: http://localhost:11434) and
    the requested model must already be pulled (``ollama pull llama3``).

    NEURO-SYMBOLIC FUSION ROLE:
      Same as extract_rules_openai() -- returned rules are loaded into
      EthicalKDEAnomalyDetector.set_prior_knowledge() and contribute additive
      penalty_weight bonuses to the KDE anomaly signal when a patient record
      contains the Italian clinical_conditions string.

    ITALIAN TERMINOLOGY INVARIANT:
      The same Italian-language system prompt is used as in the OpenAI extractor.
      Open-weight models may exhibit lower instruction-following fidelity regarding
      language constraints.  Post-extraction, verify that clinical_conditions values
      are in Italian before loading rules into the detector.

    Parameters
    ----------
    text:
        Raw Italian text from a clinical guideline PDF, typically obtained via
        extract_text_from_pdf().
    model_name:
        Ollama model tag (e.g., "llama3", "mistral", "phi3").
        The model must be available locally (``ollama list``).
    max_text_chars:
        Maximum characters of ``text`` sent to the model to stay within its
        context window.  Reduce this for smaller models (e.g., 7B parameter).

    Returns
    -------
    list[dict]
        Validated rule dicts with keys: rule_id, description,
        clinical_conditions (Italian), penalty_weight.

    Raises
    ------
    ollama.ResponseError
        If the Ollama server is unreachable or the model is not found locally.
    """
    excerpt = text[:max_text_chars]
    user_message = _USER_MESSAGE_TEMPLATE.format(excerpt=excerpt)

    logger.info(
        "[Ollama] Calling model=%s to extract clinical rules from %d chars.",
        model_name,
        len(excerpt),
    )

    response = ollama.chat(
        model=model_name,
        format="json",
        messages=[
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        options={"temperature": 0},
    )

    # The ollama library returns a dict; content is at response["message"]["content"].
    raw: str = response["message"]["content"] if isinstance(response, dict) else response.message.content
    raw = raw or "{}"

    return _parse_and_validate_rules(raw, source="Ollama")
