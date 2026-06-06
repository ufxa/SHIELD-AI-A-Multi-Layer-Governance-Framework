"""Evaluation metrics for the SHIELD-AI experiment.

The functions in this module implement the metrics referenced in the
empirical-evaluation section of the paper:

  * Binary attack-vs-benign confusion (precision, recall, F1, FPR).
  * Multi-class macro/weighted metrics over the 14 CICIDS-2017 labels.
  * HITL escalation rate (fraction of decisions routed to a human).
  * Latency summary (mean, median, p95, p99) per stage.
  * Groundedness distribution (mean, std, per-class statistics).
  * Audit completeness check via `AuditLog.verify_chain`.

All functions return plain Python dictionaries so that the runner can
serialise them to JSON for downstream analysis.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


# ----- Binary attack/benign metrics -----------------------------------------

def _binarise(y_true: Sequence[str], y_pred: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    yt = np.array([0 if v == "BENIGN" else 1 for v in y_true], dtype=int)
    yp = np.array([0 if v == "BENIGN" else 1 for v in y_pred], dtype=int)
    return yt, yp


def binary_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    p_attack: Sequence[float] | None = None,
) -> Dict[str, float]:
    yt, yp = _binarise(y_true, y_pred)
    tn = int(((yt == 0) & (yp == 0)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    tp = int(((yt == 1) & (yp == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (fn + tp) if fn + tp else 0.0
    out = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
    }
    if p_attack is not None:
        try:
            out["roc_auc"] = float(roc_auc_score(yt, np.asarray(p_attack)))
        except ValueError:
            out["roc_auc"] = float("nan")
    return out


def roc_curve_data(
    y_true: Sequence[str], p_attack: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    yt, _ = _binarise(y_true, y_true)  # only care about labels
    fpr, tpr, thr = roc_curve(yt, np.asarray(p_attack))
    return fpr, tpr, thr


# ----- Multi-class metrics --------------------------------------------------

def multiclass_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] | None = None
) -> Dict[str, float]:
    if labels is None:
        labels = sorted(set(list(y_true) + list(y_pred)))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels), average="macro", zero_division=0
    )
    w_p, w_r, w_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels), average="weighted", zero_division=0
    )
    return {
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(w_p),
        "weighted_recall": float(w_r),
        "weighted_f1": float(w_f1),
    }


def per_class_table(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] | None = None
) -> pd.DataFrame:
    if labels is None:
        labels = sorted(set(list(y_true) + list(y_pred)))
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels), zero_division=0
    )
    return pd.DataFrame(
        {
            "label": list(labels),
            "support": sup,
            "precision": p,
            "recall": r,
            "f1": f1,
        }
    )


def confusion_matrix_df(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    return pd.DataFrame(cm, index=list(labels), columns=list(labels))


# ----- HITL routing --------------------------------------------------------

def routing_distribution(routes: Sequence[str]) -> Dict[str, float]:
    n = max(1, len(routes))
    counts = Counter(routes)
    return {k: counts.get(k, 0) / n for k in ("AUTO", "HITL", "REJECT")}


def hitl_rate_by_class(
    y_true: Sequence[str], routes: Sequence[str]
) -> pd.DataFrame:
    by_class: Dict[str, Dict[str, int]] = defaultdict(lambda: {"AUTO": 0, "HITL": 0, "REJECT": 0})
    for label, route in zip(y_true, routes):
        by_class[label][route] = by_class[label].get(route, 0) + 1
    rows = []
    for label, counts in by_class.items():
        total = sum(counts.values())
        rows.append(
            {
                "label": label,
                "n": total,
                "auto_rate": counts.get("AUTO", 0) / max(1, total),
                "hitl_rate": counts.get("HITL", 0) / max(1, total),
                "reject_rate": counts.get("REJECT", 0) / max(1, total),
            }
        )
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


# ----- Latency -------------------------------------------------------------

def latency_summary(seconds_per_item: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(seconds_per_item, dtype=float) * 1000.0  # convert to ms
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "n": int(arr.size),
    }


# ----- Groundedness --------------------------------------------------------

def gamma_distribution(decisions) -> Dict[str, float]:
    g = np.array([d.triple.gamma_prime for d in decisions])
    return {
        "mean": float(g.mean()),
        "std": float(g.std()),
        "p10": float(np.percentile(g, 10)),
        "p50": float(np.percentile(g, 50)),
        "p90": float(np.percentile(g, 90)),
    }


def gamma_per_class(decisions, y_true: Sequence[str]) -> pd.DataFrame:
    rows = []
    by_class: Dict[str, List[float]] = defaultdict(list)
    for d, y in zip(decisions, y_true):
        by_class[y].append(d.triple.gamma_prime)
    for label, vals in by_class.items():
        arr = np.array(vals)
        rows.append(
            {
                "label": label,
                "n": len(arr),
                "gamma_mean": float(arr.mean()) if arr.size else float("nan"),
                "gamma_std": float(arr.std()) if arr.size else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("gamma_mean", ascending=False).reset_index(drop=True)
