"""
llm_wrapper.py
~~~~~~~~~~~~~~
Wrappers around the OpenAI Chat Completions API for:
  1. Extracting clinical Prior-Knowledge rules from PDF guidelines.
  2. Generating defeasible XAI reports for flagged patients.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import openai

logger = logging.getLogger(__name__)


def _get_client() -> openai.OpenAI:
    """Instantiate an OpenAI client using the environment API key."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it to your .env file."
        )
    return openai.OpenAI(api_key=api_key)


# ---------------------------------------------------------------------------
# XAI Report
# ---------------------------------------------------------------------------

_XAI_SYSTEM_PROMPT = """
You are a senior clinical AI ethicist specialising in paediatric emergency medicine.
Your role is to produce transparent, defeasible explanations for anomaly alerts raised
by an automated Clinical Decision Support System (CDSS) screening for potential
child abuse.

Defeasible reasoning means you MUST:
1. State the prima-facie case for concern based on the temporal pattern.
2. Acknowledge plausible alternative (non-abusive) explanations.
3. Specify which medical guidelines or red-flag criteria were triggered.
4. Conclude with a balanced clinical recommendation — never a definitive verdict.

Write in clear language appropriate for a clinical team, not a layperson.
""".strip()


def generate_xai_report(
    patient_timeline: list[dict[str, Any]],
    anomaly_score: float,
    triggered_rules: list[dict[str, Any]],
    model: str = "gpt-4o-mini",
) -> str:
    """
    Generate a defeasible XAI narrative for a flagged patient pathway.

    Parameters
    ----------
    patient_timeline:
        Ordered list of visit dictionaries (oldest → newest).  Each dict
        should include at minimum ``date``, ``chief_complaint``,
        ``triage_code``, and ``discharge_diagnosis``.
    anomaly_score:
        The raw KDE anomaly signal value for this window.
    triggered_rules:
        List of prior-rule dicts (from ``EthicalKDEAnomalyDetector``) that
        were violated by this sample and amplified the score.
    model:
        OpenAI model identifier.

    Returns
    -------
    str
        The LLM-generated clinical explanation.
    """
    timeline_text = "\n".join(
        f"  Visit {i+1}: {json.dumps(visit, default=str)}"
        for i, visit in enumerate(patient_timeline)
    )

    rules_text = (
        "\n".join(
            f"  - [{r.get('rule_id', '?')}] {r.get('descrizione', 'No description')}"
            for r in triggered_rules
        )
        if triggered_rules
        else "  None"
    )

    user_prompt = f"""
A CDSS anomaly alert has been raised for the following paediatric patient.

**Anomaly Score**: {anomaly_score:.4f}

**Temporal ER Visit Sequence** (oldest → newest):
{timeline_text}

**Triggered Clinical Prior Rules**:
{rules_text}

Please provide a defeasible reasoning report covering:
1. Prima-facie concerns from the temporal pattern.
2. Plausible alternative explanations.
3. Specific guidelines / red-flag criteria triggered.
4. Balanced clinical recommendation for the attending team.
""".strip()

    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _XAI_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=800,
    )

    report: str = response.choices[0].message.content or ""
    logger.info("XAI report generated (%d chars).", len(report))
    return report


# ---------------------------------------------------------------------------
# Prior-Knowledge Extraction
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM_PROMPT = """
You are a medical knowledge engineer.  Extract structured, machine-readable
prior rules from clinical guidelines about child abuse red flags.

Return ONLY a valid JSON array where each element has exactly these keys:
  "feature"      – the clinical variable name (snake_case)
  "condition"    – one of: gt, gte, lt, lte, eq
  "threshold"    – numeric value
  "description"  – one concise sentence explaining the clinical rationale

Example:
[
  {
    "feature": "num_er_visits_90d",
    "condition": "gt",
    "threshold": 3,
    "description": "More than 3 ER visits in 90 days is a sentinel red flag for potential abuse."
  }
]
""".strip()


def extract_prior_rules_from_text(
    guideline_text: str,
    model: str = "gpt-4o-mini",
) -> list[dict[str, Any]]:
    """
    Use an LLM to extract structured prior rules from free-text guidelines.

    Parameters
    ----------
    guideline_text:
        Raw text extracted from a clinical PDF or manual.
    model:
        OpenAI model identifier.

    Returns
    -------
    list[dict]
        Parsed list of rule dictionaries, ready for
        ``EthicalKDEAnomalyDetector.set_prior_knowledge()``.
    """
    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",  "content": _EXTRACTION_SYSTEM_PROMPT},
            {"role": "user",    "content": guideline_text[:8000]},
        ],
        temperature=0.0,
        max_tokens=1000,
    )

    raw = response.choices[0].message.content or "[]"

    # Strip markdown fences if the model adds them
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        rules: list[dict] = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse LLM rule extraction output: %s", exc)
        rules = []

    logger.info("Extracted %d prior rules from guideline text.", len(rules))
    return rules
