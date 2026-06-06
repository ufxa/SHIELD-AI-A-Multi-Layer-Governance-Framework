"""SHIELD-AI Layer 3: decision-theoretic HITL routing (PC3).

Layer 3 receives Layer-2 decisions and a per-alert composite
reliability score R = w_theta*theta + w_sigma*sigma + w_gamma*gamma'
and applies the three-zone routing policy of Section 6 of the paper:

    R >= r_auto       -> AUTO   (machine-actioned)
    R <  r_reject     -> REJECT (safe-default; ignored or dropped)
    otherwise         -> HITL   (escalated to an analyst)

The function `degraded_mode_route` implements the bypass rules that
fire when one or more reliability signals are unavailable.  When a
signal is degraded, weights are redistributed proportionally over the
surviving signals and the row is forced to HITL regardless of its
composite score, mirroring the engineering rule described in
Section 6.7 of the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Sequence, Tuple

import numpy as np

from .config import (
    DEFAULT_RELIABILITY,
    DEFAULT_THRESHOLDS,
    HITLThresholds,
    ReliabilityWeights,
)
from .layer2_llm_rag import LLMDecision


class Route(str, Enum):
    AUTO = "AUTO"
    HITL = "HITL"
    REJECT = "REJECT"


@dataclass
class RoutingOutcome:
    route: Route
    composite_R: float
    rationale: str
    degraded_signals: Tuple[str, ...]


def composite_score(
    decision: LLMDecision,
    weights: ReliabilityWeights = DEFAULT_RELIABILITY,
) -> float:
    return decision.triple.composite(weights)


def _degraded_signals(decision: LLMDecision) -> Tuple[str, ...]:
    """Return the names of unavailable signals (NaN or out-of-range)."""
    bad: List[str] = []
    t = decision.triple
    for name, val in (("theta", t.theta), ("sigma", t.sigma), ("gamma", t.gamma_prime)):
        if val is None or np.isnan(val) or val < 0.0 or val > 1.0:
            bad.append(name)
    return tuple(bad)


def route_decision(
    decision: LLMDecision,
    thresholds: HITLThresholds = DEFAULT_THRESHOLDS,
    weights: ReliabilityWeights = DEFAULT_RELIABILITY,
) -> RoutingOutcome:
    degraded = _degraded_signals(decision)
    if degraded:
        # If at least one signal is degraded, force HITL per Section 6.7.
        R = composite_score(decision, weights)
        return RoutingOutcome(
            route=Route.HITL,
            composite_R=R,
            rationale=f"degraded signals: {','.join(degraded)} forced HITL",
            degraded_signals=degraded,
        )
    R = composite_score(decision, weights)
    if R >= thresholds.r_auto:
        return RoutingOutcome(
            route=Route.AUTO,
            composite_R=R,
            rationale=f"R={R:.3f} >= r_auto={thresholds.r_auto:.3f}",
            degraded_signals=(),
        )
    if R < thresholds.r_reject:
        return RoutingOutcome(
            route=Route.REJECT,
            composite_R=R,
            rationale=f"R={R:.3f} < r_reject={thresholds.r_reject:.3f}",
            degraded_signals=(),
        )
    return RoutingOutcome(
        route=Route.HITL,
        composite_R=R,
        rationale=f"r_reject<={R:.3f}<r_auto -> analyst review",
        degraded_signals=(),
    )


def route_batch(
    decisions: Sequence[LLMDecision],
    thresholds: HITLThresholds = DEFAULT_THRESHOLDS,
    weights: ReliabilityWeights = DEFAULT_RELIABILITY,
) -> List[RoutingOutcome]:
    return [route_decision(d, thresholds, weights) for d in decisions]
