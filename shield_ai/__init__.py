"""SHIELD-AI: Multi-Layer Governance Framework reference implementation.

Reference implementation of the four-layer architecture introduced in
"SHIELD-AI: A Multi-Layer Governance Framework for Generative AI in
Security Operations Centres".

Layers:
    L1 - Ingest, normalisation, and PII gate
    L2 - LLM/RAG reasoning with reliability triple
    L3 - Decision-theoretic HITL routing
    L4 - Append-only audit log with Merkle chaining
"""

from . import (
    config,
    synthetic_data,
    cicids_loader,
    mitre_attack_kb,
    layer1_ingest,
    layer2_baselines,
    layer2_llm_rag,
    layer3_hitl,
    layer4_audit,
    metrics,
)

__version__ = "0.1.0"
__all__ = [
    "config",
    "synthetic_data",
    "cicids_loader",
    "mitre_attack_kb",
    "layer1_ingest",
    "layer2_baselines",
    "layer2_llm_rag",
    "layer3_hitl",
    "layer4_audit",
    "metrics",
]
