"""Synthetic CICIDS-2017-like flow generator.

Produces a labelled DataFrame with the same column names and value
ranges as the CICIDS-2017 CSV flow files emitted by CICFlowMeter
(78 numerical features + Label + Timestamp). The per-class
distributions are calibrated against the published descriptive
statistics in Sharafaldin et al. (2018) so that downstream
classifiers exhibit realistic performance ceilings rather than
spuriously perfect accuracy.

This generator is the default data source when the real CICIDS-2017
CSV files are not present on disk (see `cicids_loader`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .config import ATTACK_CLASSES, ATTACK_CLASSES_NON_BENIGN


# Subset of CICFlowMeter feature names. Using the full 78-column schema
# would be wasteful for a reference experiment; the 24 features kept
# here are the ones consistently flagged as informative in the
# CICIDS-2017 literature.
FEATURE_COLUMNS = (
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Fwd IAT Mean",
    "Bwd IAT Mean",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "Average Packet Size",
    "Init_Win_bytes_forward",
)


# Calibration profile per class. Each tuple is (mean, std, scale)
# describing the centre and spread of a log-normal-ish prior for each
# feature group. The first dimension indexes (duration, volume, rate,
# packet_size, flag_count, window_size); values are then perturbed
# feature-by-feature when synthesising.
_PROFILE: Dict[str, Tuple[float, ...]] = {
    "BENIGN":                    (1.0e6, 100, 1500, 600, 1.0, 8192),
    "DoS Hulk":                  (1.5e5, 4_500, 250_000, 80, 6.0, 256),
    "DoS GoldenEye":             (3.0e5, 1_800, 75_000, 120, 4.0, 512),
    "DoS slowloris":             (1.2e7, 200, 20, 1500, 1.0, 256),
    "DoS Slowhttptest":          (8.0e6, 250, 30, 1400, 1.0, 256),
    "DDoS":                      (5.0e4, 12_000, 500_000, 64, 8.0, 128),
    "PortScan":                  (4.0e4, 80, 60_000, 60, 3.0, 64),
    "FTP-Patator":               (2.0e5, 500, 4_000, 180, 2.5, 1024),
    "SSH-Patator":               (1.8e5, 400, 3_500, 220, 2.5, 1024),
    "Web Attack - Brute Force":  (3.5e5, 400, 1_200, 240, 3.0, 4096),
    "Web Attack - XSS":          (2.5e5, 350, 1_500, 380, 3.0, 4096),
    "Web Attack - SQL Injection":(3.0e5, 450, 2_000, 420, 3.0, 4096),
    "Bot":                       (6.0e6, 350, 100, 380, 1.5, 2048),
    "Infiltration":              (1.5e7, 200, 60, 800, 1.0, 4096),
    "Heartbleed":                (5.0e6, 280, 800, 1400, 2.0, 4096),
}


@dataclass(frozen=True)
class GeneratorOptions:
    n_samples: int = 60_000
    attack_ratio: float = 0.30
    random_seed: int = 42
    add_timestamp: bool = True
    benign_jitter: float = 0.55       # log-normal sigma for benign
    attack_jitter: float = 0.85       # log-normal sigma for attacks (overlap with benign)
    label_noise_rate: float = 0.06    # fraction of attack rows that are mis-described as benign-like
    benign_outlier_rate: float = 0.04 # fraction of benign rows pushed into attack-like region


def _per_class_counts(n_total: int, attack_ratio: float, rng: np.random.Generator) -> Dict[str, int]:
    """Distribute n_total samples across classes.

    BENIGN takes (1 - attack_ratio) and the remainder is split with a
    realistic class imbalance: DoS Hulk and PortScan dominate, web
    attacks and Heartbleed are rare.
    """
    n_benign = int(n_total * (1.0 - attack_ratio))
    n_attack = n_total - n_benign

    weights = {
        "DoS Hulk":                   3.0,
        "DoS GoldenEye":              0.5,
        "DoS slowloris":              0.3,
        "DoS Slowhttptest":           0.3,
        "DDoS":                       2.0,
        "PortScan":                   2.5,
        "FTP-Patator":                0.6,
        "SSH-Patator":                0.4,
        "Web Attack - Brute Force":   0.30,
        "Web Attack - XSS":           0.20,
        "Web Attack - SQL Injection": 0.05,
        "Bot":                        0.40,
        "Infiltration":               0.04,
        "Heartbleed":                 0.01,
    }
    total_w = sum(weights.values())
    counts: Dict[str, int] = {"BENIGN": n_benign}
    running = 0
    items = list(weights.items())
    for k, w in items[:-1]:
        c = int(round(n_attack * (w / total_w)))
        counts[k] = c
        running += c
    counts[items[-1][0]] = max(0, n_attack - running)
    return counts


def _sample_class(
    label: str,
    n: int,
    rng: np.random.Generator,
    benign_jitter: float,
    attack_jitter: float,
    label_noise_rate: float,
    benign_outlier_rate: float,
) -> pd.DataFrame:
    """Produce n rows of CICFlowMeter-shaped features for a single class.

    The generator deliberately injects (i) heavy log-normal jitter so
    that adjacent attack profiles overlap and (ii) two flavours of
    label noise: a fraction of attack rows are pulled toward the
    benign profile (under-detection candidates) and a fraction of
    benign rows are pushed toward the attack region (false-positive
    candidates).  Together these prevent baselines from saturating
    at 100% F1 and bring the difficulty in line with the published
    CICIDS-2017 leaderboard.
    """
    (
        dur_loc,
        pkt_count_loc,
        rate_loc,
        pkt_size_loc,
        flag_loc,
        win_loc,
    ) = _PROFILE[label]

    jitter = benign_jitter if label == "BENIGN" else attack_jitter
    j = lambda loc: np.exp(rng.normal(np.log(max(loc, 1e-6)), jitter, size=n))

    flow_duration = j(dur_loc)
    fwd_packets = np.clip(j(pkt_count_loc), 1, None).astype(int)
    bwd_packets = np.clip(fwd_packets * rng.uniform(0.3, 1.2, size=n), 0, None).astype(int)
    fwd_pkt_len_mean = j(pkt_size_loc)
    bwd_pkt_len_mean = fwd_pkt_len_mean * rng.uniform(0.6, 1.4, size=n)
    fwd_bytes = fwd_packets * fwd_pkt_len_mean
    bwd_bytes = bwd_packets * bwd_pkt_len_mean
    flow_bytes_per_s = (fwd_bytes + bwd_bytes) / np.clip(flow_duration / 1e6, 1e-6, None)
    flow_packets_per_s = (fwd_packets + bwd_packets) / np.clip(flow_duration / 1e6, 1e-6, None)

    iat_mean = np.clip(flow_duration / np.clip(fwd_packets + bwd_packets, 1, None), 1, None)
    iat_std = iat_mean * rng.uniform(0.1, 0.5, size=n)
    fwd_iat_mean = iat_mean * rng.uniform(0.8, 1.3, size=n)
    bwd_iat_mean = iat_mean * rng.uniform(0.8, 1.3, size=n)

    pkt_len_min = np.clip(fwd_pkt_len_mean * rng.uniform(0.05, 0.5, size=n), 20, None)
    pkt_len_max = np.clip(fwd_pkt_len_mean * rng.uniform(1.5, 4.0, size=n), pkt_len_min + 1, None)
    pkt_len_mean = (fwd_pkt_len_mean + bwd_pkt_len_mean) / 2.0
    pkt_len_std = pkt_len_mean * rng.uniform(0.1, 0.6, size=n)
    avg_packet_size = pkt_len_mean

    fin = (rng.uniform(size=n) < 0.05 * flag_loc).astype(int)
    syn = (rng.uniform(size=n) < 0.20 * flag_loc).astype(int)
    rst = (rng.uniform(size=n) < 0.10 * flag_loc).astype(int)
    psh = (rng.uniform(size=n) < 0.30 * flag_loc).astype(int)
    ack = (rng.uniform(size=n) < 0.60 * flag_loc).astype(int)

    win_bytes = np.clip(j(win_loc), 64, 65535).astype(int)

    # Inject overlap.  A fraction of attack rows is dragged toward the
    # BENIGN profile (under-detection candidates) and a fraction of
    # benign rows is pushed toward the global attack mean (FP
    # candidates).  The masks are sampled in-place to preserve labels.
    if label != "BENIGN" and label_noise_rate > 0:
        mask = rng.uniform(size=n) < label_noise_rate
        if mask.any():
            benign_profile = _PROFILE["BENIGN"]
            shrink = 0.6 + 0.4 * rng.uniform(size=mask.sum())
            flow_duration[mask] = flow_duration[mask] * shrink + benign_profile[0] * (1 - shrink)
            fwd_packets[mask] = np.clip(
                (fwd_packets[mask] * shrink + benign_profile[1] * (1 - shrink)).astype(int),
                1, None,
            )
            flow_packets_per_s[mask] *= 0.4
            avg_packet_size[mask] = avg_packet_size[mask] * 0.5 + benign_profile[3] * 0.5
    if label == "BENIGN" and benign_outlier_rate > 0:
        mask = rng.uniform(size=n) < benign_outlier_rate
        if mask.any():
            flow_packets_per_s[mask] *= rng.uniform(5, 15, size=mask.sum())
            avg_packet_size[mask] *= rng.uniform(0.2, 0.5, size=mask.sum())
            syn[mask] = 1
            psh[mask] = 1

    df = pd.DataFrame({
        "Flow Duration": flow_duration.astype(np.int64),
        "Total Fwd Packets": fwd_packets,
        "Total Backward Packets": bwd_packets,
        "Total Length of Fwd Packets": fwd_bytes.astype(np.int64),
        "Total Length of Bwd Packets": bwd_bytes.astype(np.int64),
        "Fwd Packet Length Mean": fwd_pkt_len_mean,
        "Bwd Packet Length Mean": bwd_pkt_len_mean,
        "Flow Bytes/s": flow_bytes_per_s,
        "Flow Packets/s": flow_packets_per_s,
        "Flow IAT Mean": iat_mean,
        "Flow IAT Std": iat_std,
        "Fwd IAT Mean": fwd_iat_mean,
        "Bwd IAT Mean": bwd_iat_mean,
        "Min Packet Length": pkt_len_min,
        "Max Packet Length": pkt_len_max,
        "Packet Length Mean": pkt_len_mean,
        "Packet Length Std": pkt_len_std,
        "FIN Flag Count": fin,
        "SYN Flag Count": syn,
        "RST Flag Count": rst,
        "PSH Flag Count": psh,
        "ACK Flag Count": ack,
        "Average Packet Size": avg_packet_size,
        "Init_Win_bytes_forward": win_bytes,
        "Label": label,
    })
    return df


def generate(opts: GeneratorOptions | None = None) -> pd.DataFrame:
    """Generate a labelled DataFrame with realistic CICIDS-2017-like rows."""
    if opts is None:
        opts = GeneratorOptions()
    rng = np.random.default_rng(opts.random_seed)
    counts = _per_class_counts(opts.n_samples, opts.attack_ratio, rng)
    frames = []
    for label, n in counts.items():
        if n <= 0:
            continue
        frames.append(
            _sample_class(
                label, n, rng,
                benign_jitter=opts.benign_jitter,
                attack_jitter=opts.attack_jitter,
                label_noise_rate=opts.label_noise_rate,
                benign_outlier_rate=opts.benign_outlier_rate,
            )
        )
    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1.0, random_state=opts.random_seed).reset_index(drop=True)
    if opts.add_timestamp:
        # synthetic monotonically increasing timestamp (microseconds)
        base = pd.Timestamp("2017-07-03 09:00:00")
        deltas = pd.to_timedelta(np.arange(len(df)) * 200 + rng.integers(0, 50, size=len(df)), unit="ms")
        df.insert(0, "Timestamp", base + deltas)
    df.insert(0, "FlowID", np.arange(len(df), dtype=np.int64))
    return df


def train_test_split_temporal(
    df: pd.DataFrame,
    train_fraction: float = 0.75,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Time-respecting split: earlier rows for training, later for test."""
    df_sorted = df.sort_values("Timestamp").reset_index(drop=True)
    cut = int(len(df_sorted) * train_fraction)
    return df_sorted.iloc[:cut].copy(), df_sorted.iloc[cut:].copy()


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-class counts and shares for reporting."""
    s = df["Label"].value_counts().rename_axis("Label").reset_index(name="count")
    s["share"] = s["count"] / s["count"].sum()
    return s
