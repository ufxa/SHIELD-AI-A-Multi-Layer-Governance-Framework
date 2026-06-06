"""Centralised configuration for the SHIELD-AI experiment.

All hyperparameters of the reliability triple, PC3 routing thresholds,
and audit-log parameters live here. Changing this file is the single
entry point for reproducible reconfiguration of the experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class ReliabilityWeights:
    """Weights for the composite reliability score R = w_theta*theta + w_sigma*sigma + w_gamma*gamma.

    The gamma score is itself a weighted sum of four sub-signals:
        gamma = w_cc*CC + w_es*ES + w_gd*GD + w_rs*RS
    """

    # Top-level triple weights (must sum to 1.0)
    w_theta: float = 0.30
    w_sigma: float = 0.30
    w_gamma: float = 0.40

    # Groundedness sub-signal weights (must sum to 1.0)
    w_cc: float = 0.20  # citation coverage (binary slot fill)
    w_es: float = 0.45  # entity-set agreement (predicted label in retrieved KB labels)
    w_gd: float = 0.20  # grounded-decision rate (top-1 retrieval score)
    w_rs: float = 0.15  # retrieved-source quality (mean of top-k)

    def validate(self) -> None:
        top = self.w_theta + self.w_sigma + self.w_gamma
        sub = self.w_cc + self.w_es + self.w_gd + self.w_rs
        if abs(top - 1.0) > 1e-6:
            raise ValueError(f"Triple weights must sum to 1.0, got {top}")
        if abs(sub - 1.0) > 1e-6:
            raise ValueError(f"Gamma sub-weights must sum to 1.0, got {sub}")


@dataclass(frozen=True)
class RoutingCosts:
    """Cost parameters for PC3 decision-theoretic routing.

    The optimal AUTO/HITL threshold is derived from
        r*_auto/HITL = (b + c_err - c_analyst - lambda) / (b + c_err)
    where b is benefit of correct AUTO, c_err is cost of a missed
    incident, c_analyst is per-decision analyst cost, and lambda is
    the regret aversion premium.
    """

    benefit_correct_auto: float = 1.0
    cost_incident_miss: float = 2.0
    cost_analyst_review: float = 0.30
    regret_aversion_lambda: float = 0.30

    def auto_threshold(self) -> float:
        b = self.benefit_correct_auto
        c_err = self.cost_incident_miss
        c_analyst = self.cost_analyst_review
        lam = self.regret_aversion_lambda
        return (b + c_err - c_analyst - lam) / (b + c_err)


@dataclass(frozen=True)
class HITLThresholds:
    """Three-zone routing thresholds derived from RoutingCosts.

    AUTO is taken when R >= r_auto.
    REJECT (drop / safe-default) when R < r_reject.
    HITL otherwise.
    """

    r_auto: float = 0.85
    r_reject: float = 0.35

    @classmethod
    def from_costs(cls, costs: RoutingCosts, hitl_margin: float = 0.5) -> "HITLThresholds":
        r_auto = costs.auto_threshold()
        r_reject = max(0.0, hitl_margin * r_auto)
        return cls(r_auto=r_auto, r_reject=r_reject)


@dataclass(frozen=True)
class LLMParameters:
    """Simulated LLM parameters used for the reliability triple."""

    self_consistency_k: int = 5        # number of samples for sigma
    temperature_sigma: float = 0.45    # noise injected between samples
    base_accuracy: float = 0.88        # per-call base classification accuracy
    rag_hit_rate: float = 0.85         # probability of pulling correct ATT&CK technique
    corpus_trust_factor: float = 0.95  # tau(corpus) for gamma' adversarial-aware


@dataclass(frozen=True)
class AuditConfig:
    """Layer-4 append-only Merkle-chained log parameters."""

    merkle_batch_size: int = 1024       # leaves per Merkle root publication
    sign_records: bool = True           # HMAC-sign each record (sim. of HSM signing)
    hmac_secret: bytes = b"SHIELD-AI-DEMO-KEY-NOT-FOR-PROD"


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment configuration."""

    random_seed: int = 42
    n_samples: int = 60_000               # synthetic flow records
    test_fraction: float = 0.25
    train_fraction: float = 0.75
    attack_ratio: float = 0.30            # fraction of samples that are attacks
    n_jobs_classifier: int = -1           # all cores
    output_dir: str = "results"
    figures_dir: str = "results/figures"


# Default instance bound to the values used in the paper.
DEFAULT_RELIABILITY = ReliabilityWeights()
DEFAULT_COSTS = RoutingCosts()
DEFAULT_THRESHOLDS = HITLThresholds.from_costs(DEFAULT_COSTS)
DEFAULT_LLM = LLMParameters()
DEFAULT_AUDIT = AuditConfig()
DEFAULT_EXPERIMENT = ExperimentConfig()


# Mapping CICIDS-2017 label to MITRE ATT&CK technique IDs (used for
# groundedness scoring at Layer 2 and also for paper Table E1).
CICIDS_TO_ATTCK: Dict[str, Tuple[str, ...]] = {
    "BENIGN": (),
    "DoS Hulk": ("T1499.001", "T1499"),
    "DoS GoldenEye": ("T1499.001", "T1499"),
    "DoS slowloris": ("T1499.004", "T1499"),
    "DoS Slowhttptest": ("T1499.004", "T1499"),
    "DDoS": ("T1498.001", "T1498"),
    "PortScan": ("T1046",),
    "FTP-Patator": ("T1110.001", "T1110"),
    "SSH-Patator": ("T1110.001", "T1110"),
    "Web Attack - Brute Force": ("T1110.003", "T1110"),
    "Web Attack - XSS": ("T1059.007",),
    "Web Attack - SQL Injection": ("T1190",),
    "Bot": ("T1583.006", "T1071.001"),
    "Infiltration": ("T1078",),
    "Heartbleed": ("T1190",),
}


ATTACK_CLASSES = tuple(CICIDS_TO_ATTCK.keys())
ATTACK_CLASSES_NON_BENIGN = tuple(c for c in ATTACK_CLASSES if c != "BENIGN")
