"""SHIELD-AI Layer 2 baselines: rule-based and classical ML.

These baselines exist so that the LLM/RAG path of Layer 2 can be
compared head-to-head against rule and ML alternatives, as required
by OP1 (quantitative validation) and the experimental section of
the paper.

* `RuleBasedClassifier` encodes a small set of hand-tuned heuristics
  reflecting how a SOC analyst would write Sigma-style rules for
  CICIDS-2017 traffic.
* `MLBaseline` wraps a scikit-learn Random Forest, trained on the
  numeric features extracted by Layer 1.

Both classifiers return per-row predicted labels, predicted
probabilities for the positive (attack) class, and elapsed wall-clock
time in seconds for the prediction step.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


# ---- Rule-based -------------------------------------------------------------

@dataclass(frozen=True)
class RuleThresholds:
    portscan_syn_min: int = 1
    portscan_pps_min: float = 8_000.0
    portscan_avg_pkt_max: float = 90.0

    flood_pps_min: float = 5_000.0
    flood_pkts_min: int = 1_000
    flood_avg_pkt_max: float = 180.0

    slowloris_duration_min: float = 4.0e6
    slowloris_pkts_max: int = 600
    slowloris_pps_max: float = 100.0

    brute_force_psh_min: int = 1
    brute_force_pkts_min: int = 200
    brute_force_pkts_max: int = 3_000

    web_attack_pkt_size_min: float = 200.0
    web_attack_psh_min: int = 1

    bot_duration_min: float = 2.0e6
    bot_pkts_max: int = 800
    bot_avg_pkt_min: float = 200.0


class RuleBasedClassifier:
    """Hand-written Sigma-style rules for CICIDS-2017."""

    def __init__(self, thresholds: RuleThresholds | None = None) -> None:
        self.t = thresholds or RuleThresholds()

    def _classify_row(self, r: pd.Series) -> Tuple[str, float]:
        t = self.t
        # PortScan
        if (r["SYN Flag Count"] >= t.portscan_syn_min
            and r["Flow Packets/s"] >= t.portscan_pps_min
            and r["Average Packet Size"] <= t.portscan_avg_pkt_max):
            return "PortScan", 0.92
        # DoS Hulk / DDoS (high-rate flood)
        if (r["Flow Packets/s"] >= t.flood_pps_min
            and (r["Total Fwd Packets"] + r["Total Backward Packets"]) >= t.flood_pkts_min
            and r["Average Packet Size"] <= t.flood_avg_pkt_max):
            return "DoS Hulk", 0.85
        # DoS slowloris
        if (r["Flow Duration"] >= t.slowloris_duration_min
            and (r["Total Fwd Packets"] + r["Total Backward Packets"]) <= t.slowloris_pkts_max
            and r["Flow Packets/s"] <= t.slowloris_pps_max):
            return "DoS slowloris", 0.78
        # Brute force (FTP/SSH)
        if (r["PSH Flag Count"] >= t.brute_force_psh_min
            and t.brute_force_pkts_min <= (r["Total Fwd Packets"] + r["Total Backward Packets"]) <= t.brute_force_pkts_max
            and r["Average Packet Size"] < 280):
            return "FTP-Patator", 0.72
        # Web attack family
        if (r["Average Packet Size"] >= t.web_attack_pkt_size_min
            and r["PSH Flag Count"] >= t.web_attack_psh_min
            and r["Init_Win_bytes_forward"] >= 1024):
            return "Web Attack - Brute Force", 0.65
        # Bot beacon
        if (r["Flow Duration"] >= t.bot_duration_min
            and (r["Total Fwd Packets"] + r["Total Backward Packets"]) <= t.bot_pkts_max
            and r["Average Packet Size"] >= t.bot_avg_pkt_min):
            return "Bot", 0.70
        return "BENIGN", 0.95

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, float]:
        start = time.perf_counter()
        labels = np.empty(len(X), dtype=object)
        probs = np.empty(len(X), dtype=np.float64)
        for i, (_, row) in enumerate(X.iterrows()):
            label, p = self._classify_row(row)
            labels[i] = label
            probs[i] = p if label != "BENIGN" else 1.0 - p
        return labels, probs, time.perf_counter() - start


# ---- ML baseline ------------------------------------------------------------

class MLBaseline:
    """Random Forest baseline wrapper that returns class probabilities."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 18,
        n_jobs: int = -1,
        random_seed: int = 42,
    ) -> None:
        self.encoder = LabelEncoder()
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=4,
            n_jobs=n_jobs,
            random_state=random_seed,
            class_weight="balanced_subsample",
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        y_enc = self.encoder.fit_transform(y.values)
        self.clf.fit(X.values, y_enc)

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return predicted labels (str), attack-probability, and latency."""
        start = time.perf_counter()
        y_enc = self.clf.predict(X.values)
        proba = self.clf.predict_proba(X.values)
        elapsed = time.perf_counter() - start
        labels = self.encoder.inverse_transform(y_enc)
        # Attack probability = 1 - P(BENIGN)
        try:
            benign_idx = list(self.encoder.classes_).index("BENIGN")
            p_attack = 1.0 - proba[:, benign_idx]
        except ValueError:
            p_attack = proba.max(axis=1)
        return labels, p_attack, elapsed

    def predict_proba_full(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Per-class probability matrix for use by the LLM/RAG simulator."""
        proba = self.clf.predict_proba(X.values)
        return proba, np.asarray(self.encoder.classes_)
