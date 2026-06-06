"""Tests for the Layer-4 audit chain integrity."""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from shield_ai import layer2_llm_rag, layer3_hitl, layer4_audit  # noqa: E402
from shield_ai.layer2_llm_rag import ReliabilityTriple, LLMDecision  # noqa: E402


def _make_decision(label: str = "PortScan") -> LLMDecision:
    triple = ReliabilityTriple(
        theta=0.9, sigma=1.0, gamma=0.8, gamma_prime=0.76,
        sub_cc=1.0, sub_es=1.0, sub_gd=0.7, sub_rs=0.6,
    )
    return LLMDecision(
        predicted_label=label, triple=triple, retrieved=[], latency_seconds=0.01,
        self_consistency_samples=[label, label, label],
    )


def test_chain_integrity_passes_when_untampered():
    log = layer4_audit.AuditLog()
    for i in range(10):
        d = _make_decision()
        o = layer3_hitl.RoutingOutcome(
            route=layer3_hitl.Route.AUTO, composite_R=0.9, rationale="ok", degraded_signals=(),
        )
        log.append("PortScan", d, o, flow_id=i)
    log.flush()
    ok, bad_idx, msg = log.verify_chain()
    assert ok, f"chain should verify but failed at {bad_idx}: {msg}"


def test_chain_integrity_detects_tampering():
    log = layer4_audit.AuditLog()
    for i in range(8):
        d = _make_decision()
        o = layer3_hitl.RoutingOutcome(
            route=layer3_hitl.Route.AUTO, composite_R=0.9, rationale="ok", degraded_signals=(),
        )
        log.append("PortScan", d, o, flow_id=i)
    log.flush()
    # Tamper: flip the predicted label of record 3 without recomputing hash
    object.__setattr__(log.records[3], "predicted_label", "BENIGN")
    ok, bad_idx, msg = log.verify_chain()
    assert not ok, "chain should detect tampered record"
    assert bad_idx == 3, f"expected break at 3, got {bad_idx}"


def test_record_level_verifier_pc5():
    log = layer4_audit.AuditLog()
    for i in range(5):
        d = _make_decision()
        o = layer3_hitl.RoutingOutcome(
            route=layer3_hitl.Route.AUTO, composite_R=0.9, rationale="ok", degraded_signals=(),
        )
        log.append("PortScan", d, o, flow_id=i)
    log.flush()
    for seq in range(5):
        ok, msg = log.verify_record(seq)
        assert ok, f"record {seq} should verify: {msg}"


if __name__ == "__main__":
    test_chain_integrity_passes_when_untampered()
    test_chain_integrity_detects_tampering()
    test_record_level_verifier_pc5()
    print("OK")
