"""
progetto_ethics_ai.util
~~~~~~~~~~~~~~~~~~~~~~~
Utility modules for the Ethical CDSS for child abuse anomaly detection.
"""

from .data_processing import merge_er_data, build_features, create_sliding_windows
from .models import EthicalKDEAnomalyDetector
from .ethics_metrics import calculate_spd, calculate_di, CostModel
from .llm_wrapper import generate_xai_report
from .knowledge_extraction import (
    extract_text_from_pdf,
    extract_rules_openai,
    extract_rules_local_ollama,
)

__all__ = [
    "merge_er_data",
    "build_features",
    "create_sliding_windows",
    "EthicalKDEAnomalyDetector",
    "calculate_spd",
    "calculate_di",
    "CostModel",
    "generate_xai_report",
    "extract_text_from_pdf",
    "extract_rules_openai",
    "extract_rules_local_ollama",
]
