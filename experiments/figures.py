"""Generate the six experimental figures referenced in the paper.

Reads the predictions parquet/csv and metrics.json produced by
``run_experiment`` and renders the figures under
``results/figures/``.  Each figure is saved as both PDF (for the
LaTeX paper) and PNG (for the GitHub README).

Figures:
  E1 - ROC curves Rule vs ML vs SHIELD-AI (LLM/RAG)
  E2 - Confusion matrix of SHIELD-AI full
  E3 - Latency per layer (mean and p99)
  E4 - HITL escalation rate by attack class
  E5 - Distribution of gamma' across classes (violin)
  E6 - Reliability triple components (theta, sigma, gamma')
       stacked by class
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from shield_ai import config  # noqa: E402


sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)


def _load_predictions(output_dir: str) -> pd.DataFrame:
    parquet = os.path.join(output_dir, "test_predictions.parquet")
    csv = os.path.join(output_dir, "test_predictions.csv")
    if os.path.exists(parquet):
        try:
            return pd.read_parquet(parquet)
        except Exception:
            pass
    if os.path.exists(csv):
        return pd.read_csv(csv)
    raise FileNotFoundError("Neither test_predictions.parquet nor .csv was found.")


def _load_metrics(output_dir: str) -> Dict:
    path = os.path.join(output_dir, "metrics.json")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save(fig, base_path: str) -> None:
    pdf = base_path + ".pdf"
    png = base_path + ".png"
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight", dpi=200)
    fig.savefig(png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  wrote {pdf} and {png}")


# -----------------------------------------------------------------------------

def fig_e1_roc(preds: pd.DataFrame, out_base: str) -> None:
    """ROC curves for Rule, ML, and SHIELD-AI (LLM/RAG composite R)."""
    y_bin = (preds["true_label"] != "BENIGN").astype(int).values
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for name, score_col, colour, lw in (
        ("Rule-based",     "rule_score", "#9e9e9e", 1.4),
        ("Random Forest",  "ml_score",   "#1f77b4", 1.8),
        ("SHIELD-AI",      "llm_score",  "#d62728", 2.4),
    ):
        try:
            fpr, tpr, _ = roc_curve(y_bin, preds[score_col].values)
            from sklearn.metrics import auc
            au = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{name} (AUC={au:.3f})", color=colour, lw=lw)
        except Exception as exc:
            print(f"    ROC skipped for {name}: {exc}")
    ax.plot([0, 1], [0, 1], "--", color="lightgray", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Figure E1: Detection ROC on CICIDS-2017-like data")
    ax.legend(loc="lower right", frameon=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    _save(fig, out_base)


def fig_e2_confusion(preds: pd.DataFrame, out_base: str) -> None:
    labels = list(config.ATTACK_CLASSES)
    cm = confusion_matrix(preds["true_label"].values, preds["llm_pred"].values, labels=labels)
    # normalise by row
    row_sum = cm.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    cm_norm = cm / row_sum
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
        cbar_kws={"label": "Recall (row-normalised)"},
        annot_kws={"size": 7},
        ax=ax,
    )
    ax.set_xlabel("Predicted label (SHIELD-AI)")
    ax.set_ylabel("True label")
    ax.set_title("Figure E2: Row-normalised confusion matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    _save(fig, out_base)


def fig_e3_latency(metrics: Dict, out_base: str) -> None:
    """Per-layer latency (ms) for rule, ML and full SHIELD-AI."""
    rows = []
    for name, key in (
        ("Rule-based (L2)", "rule_based"),
        ("RF (L2)",         "ml_baseline"),
        ("SHIELD-AI L2",    "shield_ai_full"),
    ):
        lat = metrics[key]["latency"]
        rows.append({"system": name, "stat": "mean",   "ms": lat["mean_ms"]})
        rows.append({"system": name, "stat": "median", "ms": lat["median_ms"]})
        rows.append({"system": name, "stat": "p95",    "ms": lat["p95_ms"]})
        rows.append({"system": name, "stat": "p99",    "ms": lat["p99_ms"]})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    sns.barplot(
        data=df, x="system", y="ms", hue="stat",
        palette={"mean": "#1f77b4", "median": "#2ca02c", "p95": "#ff7f0e", "p99": "#d62728"},
        ax=ax,
    )
    ax.set_yscale("log")
    ax.set_ylabel("Per-alert latency (ms, log scale)")
    ax.set_xlabel("")
    ax.set_title("Figure E3: Layer-2 latency per alert")
    ax.legend(title="Statistic", loc="upper left", frameon=True)
    _save(fig, out_base)


def fig_e4_hitl(metrics: Dict, out_base: str) -> None:
    df = pd.DataFrame(metrics["shield_ai_full"]["hitl_by_class"])
    df = df.sort_values("n", ascending=False).head(12)  # cap at 12 classes for legibility
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    classes = df["label"].tolist()
    auto = df["auto_rate"].values
    hitl = df["hitl_rate"].values
    reject = df["reject_rate"].values
    width = 0.7
    ax.bar(classes, auto, width, label="AUTO",   color="#2ca02c")
    ax.bar(classes, hitl, width, bottom=auto, label="HITL",   color="#ff7f0e")
    ax.bar(classes, reject, width, bottom=auto + hitl, label="REJECT", color="#d62728")
    ax.set_ylabel("Share of decisions")
    ax.set_xlabel("")
    ax.set_title("Figure E4: PC3 routing distribution per class")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", frameon=True)
    plt.xticks(rotation=40, ha="right")
    _save(fig, out_base)


def fig_e5_gamma(preds: pd.DataFrame, out_base: str) -> None:
    """Violin plot of gamma' per class (top 10 most common in test)."""
    # Pick top-10 most common true labels
    top = preds["true_label"].value_counts().head(10).index.tolist()
    sub = preds[preds["true_label"].isin(top)].copy()
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    sns.violinplot(
        data=sub, x="true_label", y="gamma_prime",
        order=top,
        inner="quartile",
        cut=0,
        color="#5e9bd6",
        ax=ax,
    )
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("")
    ax.set_ylabel(r"Adversarial-aware groundedness $\gamma'$")
    ax.set_title(r"Figure E5: Distribution of $\gamma'$ per class")
    plt.xticks(rotation=40, ha="right")
    _save(fig, out_base)


def fig_e6_triple(preds: pd.DataFrame, out_base: str) -> None:
    top = preds["true_label"].value_counts().head(10).index.tolist()
    sub = preds[preds["true_label"].isin(top)].copy()
    agg = sub.groupby("true_label")[["theta", "sigma", "gamma_prime"]].mean().reindex(top)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(top))
    width = 0.27
    ax.bar(x - width, agg["theta"].values, width, label=r"$\theta$",         color="#1f77b4")
    ax.bar(x,         agg["sigma"].values, width, label=r"$\sigma$",         color="#2ca02c")
    ax.bar(x + width, agg["gamma_prime"].values, width, label=r"$\gamma'$",  color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(top, rotation=40, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Component mean")
    ax.set_title(r"Figure E6: Mean reliability triple components per class")
    ax.legend(loc="upper right", frameon=True)
    _save(fig, out_base)


# -----------------------------------------------------------------------------

def main() -> None:
    cfg = config.DEFAULT_EXPERIMENT
    print("Generating figures from", cfg.output_dir)
    preds = _load_predictions(cfg.output_dir)
    metrics = _load_metrics(cfg.output_dir)
    figures_dir = cfg.figures_dir
    fig_e1_roc(preds, os.path.join(figures_dir, "fig_e1_roc"))
    fig_e2_confusion(preds, os.path.join(figures_dir, "fig_e2_confusion"))
    fig_e3_latency(metrics, os.path.join(figures_dir, "fig_e3_latency"))
    fig_e4_hitl(metrics, os.path.join(figures_dir, "fig_e4_hitl"))
    fig_e5_gamma(preds, os.path.join(figures_dir, "fig_e5_gamma"))
    fig_e6_triple(preds, os.path.join(figures_dir, "fig_e6_triple"))


if __name__ == "__main__":
    main()
