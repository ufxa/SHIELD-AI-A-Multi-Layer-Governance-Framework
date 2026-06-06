"""Tests for the reliability-triple and routing logic."""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from shield_ai import config, layer2_llm_rag, layer3_hitl  # noqa: E402


def test_weights_validate():
    config.DEFAULT_RELIABILITY.validate()


def test_threshold_derivation():
    th = config.HITLThresholds.from_costs(config.DEFAULT_COSTS)
    assert 0.5 < th.r_auto < 1.0
    assert 0 <= th.r_reject < th.r_auto


def test_routing_three_zones():
    # Build synthetic decisions with composite R in each zone.
    th = config.DEFAULT_THRESHOLDS

    def make(theta, sigma, gamma):
        return layer2_llm_rag.LLMDecision(
            predicted_label="PortScan",
            triple=layer2_llm_rag.ReliabilityTriple(
                theta=theta, sigma=sigma, gamma=gamma, gamma_prime=gamma,
                sub_cc=1.0, sub_es=1.0, sub_gd=gamma, sub_rs=gamma,
            ),
            retrieved=[], latency_seconds=0.01, self_consistency_samples=[],
        )
    high = make(0.99, 1.0, 0.99)  # R close to 1
    low = make(0.05, 0.0, 0.05)   # R close to 0
    mid = make(0.5, 0.6, 0.55)    # R around 0.5

    assert layer3_hitl.route_decision(high).route == layer3_hitl.Route.AUTO
    assert layer3_hitl.route_decision(low).route == layer3_hitl.Route.REJECT
    assert layer3_hitl.route_decision(mid).route in (
        layer3_hitl.Route.HITL, layer3_hitl.Route.REJECT, layer3_hitl.Route.AUTO,
    )


def test_degraded_signal_forces_hitl():
    dec = layer2_llm_rag.LLMDecision(
        predicted_label="PortScan",
        triple=layer2_llm_rag.ReliabilityTriple(
            theta=float("nan"), sigma=1.0, gamma=0.9, gamma_prime=0.9,
            sub_cc=1.0, sub_es=1.0, sub_gd=0.9, sub_rs=0.9,
        ),
        retrieved=[], latency_seconds=0.01, self_consistency_samples=[],
    )
    out = layer3_hitl.route_decision(dec)
    assert out.route == layer3_hitl.Route.HITL
    assert "degraded" in out.rationale


if __name__ == "__main__":
    test_weights_validate()
    test_threshold_derivation()
    test_routing_three_zones()
    test_degraded_signal_forces_hitl()
    print("OK")
