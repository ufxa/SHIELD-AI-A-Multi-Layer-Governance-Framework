"""SHIELD-AI Layer 2: LLM/RAG reasoning with reliability triple <theta, sigma, gamma>.

This module implements a *simulated* LLM that performs classification
over Layer-1 alerts while computing the reliability triple defined in
Section 5 of the paper.  The simulator is intentionally not a real
LLM call: real calls would be slow and non-deterministic, and the
contribution of the paper is the *framework* that consumes the
triple, not the language model itself.  The simulator therefore
produces (i) a label prediction; (ii) a point-confidence theta;
(iii) a self-consistency sigma over k samples; and (iv) a
groundedness gamma decomposed into the four sub-signals
CC / ES / GD / RS defined in Section 5.

The simulator's accuracy and consistency parameters are exposed via
`LLMParameters` so that ablations can vary them independently.  Real
LLM outputs can be plugged in by replacing the `LLMClient.classify`
method with an HTTP call to an Ollama / vLLM / OpenAI-compatible
endpoint; the rest of the pipeline does not change.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import (
    CICIDS_TO_ATTCK,
    DEFAULT_LLM,
    DEFAULT_RELIABILITY,
    LLMParameters,
    ReliabilityWeights,
)
from .mitre_attack_kb import AttckEntry, retrieve


# -----------------------------------------------------------------------------

@dataclass
class ReliabilityTriple:
    theta: float
    sigma: float
    gamma: float
    gamma_prime: float
    sub_cc: float
    sub_es: float
    sub_gd: float
    sub_rs: float

    def composite(self, w: ReliabilityWeights = DEFAULT_RELIABILITY) -> float:
        return w.w_theta * self.theta + w.w_sigma * self.sigma + w.w_gamma * self.gamma_prime


@dataclass
class LLMDecision:
    predicted_label: str
    triple: ReliabilityTriple
    retrieved: List[Tuple[AttckEntry, float]]
    latency_seconds: float
    self_consistency_samples: List[str] = field(default_factory=list)


# ---- Simulated LLM ----------------------------------------------------------

class _RNG:
    """Deterministic per-input random number generator.

    Seeded by alert hash + sample index so two runs over the same
    alerts produce the same draws.
    """

    def __init__(self, alert_text: str, base_seed: int) -> None:
        h = hashlib.sha256(alert_text.encode("utf-8")).digest()[:8]
        self.base = int.from_bytes(h, "big") ^ base_seed
        self.counter = 0

    def step(self, sample_idx: int) -> np.random.Generator:
        seed = (self.base + sample_idx * 0x9E3779B9 + self.counter) & 0xFFFFFFFFFFFFFFFF
        self.counter += 1
        return np.random.default_rng(seed)


def _classifier_priors(
    proba_row: np.ndarray,
    classes: Sequence[str],
    temp: float,
    rng: np.random.Generator,
) -> Tuple[str, float, np.ndarray]:
    """Convert a row of classifier probabilities to a temperature-perturbed sample.

    Returns (predicted_label, predicted_probability, full_distribution).
    """
    logits = np.log(np.clip(proba_row, 1e-9, None))
    noise = rng.normal(0.0, temp, size=logits.shape)
    perturbed = logits + noise
    exp = np.exp(perturbed - perturbed.max())
    dist = exp / exp.sum()
    idx = int(np.argmax(dist))
    return classes[idx], float(dist[idx]), dist


def _groundedness_from_retrieved(
    predicted_label: str,
    retrieved: List[Tuple[AttckEntry, float]],
    corpus_trust_factor: float,
) -> Tuple[float, float, float, float, float, float]:
    """Compute gamma sub-signals CC, ES, GD, RS and combine into gamma, gamma'.

    Definitions follow Section 5 of the paper:
        CC : citation coverage   = was anything retrieved at all?
        ES : entity-set agreement= overlap between predicted label and
                                   the top retrieved technique's
                                   CICIDS labels.
        GD : grounded decision   = the top-1 technique cosine score.
        RS : retrieved sources Q = mean cosine over the top-k.
    """
    if not retrieved:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    expected_techs = set(CICIDS_TO_ATTCK.get(predicted_label, ()))
    # CC: 1.0 when every retrieval slot was filled.
    cc = 1.0
    # ES: entity-set agreement; 1.0 if the predicted label is among any
    # retrieved technique's mapped CICIDS labels, 0.85 if the
    # retrieved technique id matches the expected technique set,
    # graded down otherwise based on how deep in the ranking the match
    # appears.  BENIGN has no expected technique and always scores 0.5
    # (i.e. groundedness is undefined but neither correct nor
    # incorrect).
    if predicted_label == "BENIGN" or not expected_techs:
        es = 0.5
    else:
        es = 0.0
        for entry, _ in retrieved:
            if predicted_label in entry.cicids_labels:
                es = 1.0
                break
            if entry.technique_id in expected_techs:
                es = max(es, 0.85)
    # GD: top-1 retrieval match strength.  1.0 if the top-1 technique
    # matches the expected technique set; 0.55 if any of top-3 matches;
    # otherwise the rescaled cosine score (capped low).
    if predicted_label == "BENIGN":
        gd = 0.5
    elif retrieved[0][0].technique_id in expected_techs:
        gd = 1.0
    elif any(e.technique_id in expected_techs for e, _ in retrieved):
        gd = 0.55
    else:
        gd = float(min(0.4, retrieved[0][1] * 4.0))
    # RS: retrieved-source quality.  Fraction of the top-k that are
    # related to the expected technique set, blended with mean cosine.
    if predicted_label == "BENIGN":
        rs = 0.5
    else:
        hit_share = sum(1 for e, _ in retrieved if e.technique_id in expected_techs) / max(1, len(retrieved))
        mean_cos = float(np.mean([s for _, s in retrieved]))
        rs = float(0.7 * hit_share + 0.3 * min(1.0, mean_cos * 4.0))
    w = DEFAULT_RELIABILITY
    gamma = w.w_cc * cc + w.w_es * es + w.w_gd * gd + w.w_rs * rs
    gamma_prime = gamma * corpus_trust_factor
    return cc, es, gd, rs, float(gamma), float(gamma_prime)


class LLMRAGLayer:
    """Simulated LLM classifier with retrieval-grounded reliability triple."""

    def __init__(
        self,
        params: LLMParameters = DEFAULT_LLM,
        reliability: ReliabilityWeights = DEFAULT_RELIABILITY,
    ) -> None:
        self.params = params
        self.reliability = reliability

    def classify_batch(
        self,
        alert_texts: Sequence[str],
        proba_matrix: np.ndarray,
        classes: Sequence[str],
    ) -> Tuple[List[LLMDecision], float]:
        """Classify a batch of alerts.

        `proba_matrix` and `classes` come from the upstream classical
        ML classifier and act as the LLM's prior beliefs.  The
        simulator perturbs this prior with self-consistency sampling
        to derive sigma, and grounds the top prediction against the
        ATT&CK KB to derive gamma.
        """
        decisions: List[LLMDecision] = []
        t0 = time.perf_counter()
        for i, text in enumerate(alert_texts):
            decision = self._classify_one(text, proba_matrix[i], classes, alert_idx=i)
            decisions.append(decision)
        elapsed = time.perf_counter() - t0
        return decisions, elapsed

    def _classify_one(
        self,
        alert_text: str,
        proba_row: np.ndarray,
        classes: Sequence[str],
        alert_idx: int,
    ) -> LLMDecision:
        start = time.perf_counter()
        rng_factory = _RNG(alert_text, base_seed=self.params.self_consistency_k * 7919 + alert_idx)
        k = self.params.self_consistency_k
        labels: List[str] = []
        confidences: List[float] = []
        dists: List[np.ndarray] = []
        for s in range(k):
            rng = rng_factory.step(s)
            label, conf, dist = _classifier_priors(
                proba_row, classes, temp=self.params.temperature_sigma, rng=rng,
            )
            labels.append(label)
            confidences.append(conf)
            dists.append(dist)
        # theta = mean confidence of the majority answer
        majority = max(set(labels), key=labels.count)
        theta = float(np.mean([c for l, c in zip(labels, confidences) if l == majority]))
        # sigma = fraction of samples agreeing with majority
        sigma = labels.count(majority) / k
        # Retrieve ATT&CK context
        retrieved = retrieve(alert_text + " " + majority, k=3)
        cc, es, gd, rs, gamma, gamma_prime = _groundedness_from_retrieved(
            majority, retrieved, self.params.corpus_trust_factor,
        )
        triple = ReliabilityTriple(
            theta=theta,
            sigma=sigma,
            gamma=gamma,
            gamma_prime=gamma_prime,
            sub_cc=cc,
            sub_es=es,
            sub_gd=gd,
            sub_rs=rs,
        )
        latency = time.perf_counter() - start
        return LLMDecision(
            predicted_label=majority,
            triple=triple,
            retrieved=retrieved,
            latency_seconds=latency,
            self_consistency_samples=labels,
        )
