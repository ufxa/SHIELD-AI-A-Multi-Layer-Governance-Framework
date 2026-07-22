"""
SHIELD-AI — Quantitative data figures
Generated: 2026-05-31 · Stage 4 supplement

Produces three figures:
  F6 — ROC comparison: rule-based vs classical ML vs LLM/RAG
       (ILLUSTRATIVE — synthetic data clearly labelled in caption)
  F7 — Heatmap of SHIELD-AI control × external-standard coverage
       (Data derived from Table 4 of the manuscript outline)
  F8 — Threat trend bar chart with verified 2023–2025 industry datapoints
       (Data sourced from B12–B23, D10 of verified_bibliography_v2.md)

Outputs go to ./build/ as both .pdf (for LaTeX inclusion) and .svg.
Run:  python3 figures_data_v1.py
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# ---------------------------------------------------------------------
# Output config
# ---------------------------------------------------------------------
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
os.makedirs(OUT_DIR, exist_ok=True)

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def save(fig, name: str) -> None:
    for ext in ("pdf", "svg", "png"):
        out = os.path.join(OUT_DIR, f"{name}.{ext}")
        fig.savefig(out, dpi=200 if ext == "png" else None)
        print(f"  wrote {out}")


# =====================================================================
# F6 — ROC comparison
# =====================================================================
def figure_f6_roc() -> None:
    """ILLUSTRATIVE ROC curves comparing three SOC detection approaches.

    Note: the curves are synthetic but parameterized to reflect qualitative
    patterns reported in the literature ([A11, A12] for LLM/RAG SOC empirical
    studies; [A6, A7, A8] CyberSecEval-family benchmarks). The figure caption
    in the manuscript must mark this explicitly.
    """
    np.random.seed(202651)
    fpr = np.linspace(0.0, 1.0, 200)

    # Beta-CDF-like shapes parameterized to encode relative performance:
    #   rule-based: high precision at very low FPR, but ceiling around 0.7 TPR.
    #   classical ML: balanced curve, AUC ≈ 0.85.
    #   LLM/RAG: higher AUC ceiling but higher variance.
    def curve(fpr_arr, alpha, beta):
        from scipy.stats import beta as Beta  # local import — soft dep
        return Beta.cdf(fpr_arr, alpha, beta)

    try:
        rule = curve(fpr, 0.5, 3.5)
        cml  = curve(fpr, 0.9, 2.0)
        llm  = curve(fpr, 1.1, 1.3)
    except ImportError:
        # Fallback: parametric S-curve without scipy
        def s(fpr_arr, k, x0):
            return 1.0 / (1.0 + np.exp(-k * (fpr_arr - x0)))
        rule = np.clip(s(fpr, 7, 0.25) * 0.72, 0, 1)
        cml  = np.clip(s(fpr, 5, 0.18) * 0.93, 0, 1)
        llm  = np.clip(s(fpr, 4, 0.12) * 0.97, 0, 1)

    def auc(x, y):
        return float(np.trapezoid(y, x))

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=0.8,
            label="Random (AUC=0.50)")
    ax.plot(fpr, rule, color="#1f77b4", linewidth=1.8,
            label=f"Rule-based (AUC={auc(fpr, rule):.2f})")
    ax.plot(fpr, cml,  color="#ff7f0e", linewidth=1.8,
            label=f"Classical ML (AUC={auc(fpr, cml):.2f})")
    ax.plot(fpr, llm,  color="#2ca02c", linewidth=1.8,
            label=f"LLM/RAG (AUC={auc(fpr, llm):.2f})")

    # Shaded variance band for LLM
    band = np.clip(np.minimum(llm + 0.06, 1.0), 0, 1)
    band_low = np.clip(llm - 0.06, 0, 1)
    ax.fill_between(fpr, band_low, band, color="#2ca02c", alpha=0.12,
                    label="LLM/RAG ±1σ (illustrative)")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("F6 · Detection ROC: rule-based vs classical ML vs LLM/RAG\n"
                 "(illustrative; parameterized to reflect qualitative literature trends)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True)
    save(fig, "F6_roc_comparison")
    plt.close(fig)


# =====================================================================
# F7 — Heatmap: SHIELD-AI controls × external standards
# =====================================================================
def figure_f7_mapping_heatmap() -> None:
    """Mapping density of Table 4 (SHIELD-AI controls × standards)."""
    controls = [
        "Input sanitization (L2)",
        "RAG-grounding verification",
        "Hallucination consistency check",
        "Tool allowlist + signing",
        "Output filtering",
        "Audit log (L4)",
        "HITL escalation (L3)",
        "PII / secret pre-scan",
        "LGPD compliance gate",
        "Identity / Zero-Trust",
    ]
    standards = [
        "MITRE\nATT&CK",
        "MITRE\nATLAS",
        "NIST\nCSF 2.0",
        "OWASP\nLLM Top 10",
        "OWASP\nAgentic",
        "ISO 27001\n:2022",
        "ISO 42001\n:2023",
        "LGPD",
    ]
    # Coverage matrix:
    #   0 = no alignment
    #   1 = partial / supporting
    #   2 = primary alignment
    matrix = np.array([
        # ATT&CK ATLAS CSF OWASP-LLM OWASP-A ISO27 ISO42 LGPD
        [0,      2,    1,    2,         1,      1,    1,    1],   # Input sanitization
        [0,      2,    1,    2,         0,      1,    2,    1],   # RAG-grounding
        [0,      2,    1,    2,         0,      1,    2,    1],   # Hallucination consistency
        [1,      2,    1,    2,         2,      2,    2,    1],   # Tool allowlist
        [0,      2,    1,    2,         1,      1,    1,    1],   # Output filtering
        [0,      0,    2,    1,         2,      2,    2,    2],   # Audit log
        [0,      1,    2,    2,         2,      1,    2,    2],   # HITL escalation
        [0,      0,    2,    2,         0,      2,    2,    2],   # PII pre-scan
        [0,      0,    2,    0,         0,      2,    2,    2],   # LGPD gate
        [1,      0,    2,    1,         2,      2,    2,    2],   # Identity / ZT
    ])

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    cmap = sns.color_palette("YlGnBu", as_cmap=True)
    sns.heatmap(matrix, ax=ax, cmap=cmap, cbar_kws={
                    "label": "Alignment depth (0=none · 1=partial · 2=primary)",
                    "ticks": [0, 1, 2]},
                xticklabels=standards, yticklabels=controls,
                linewidths=0.4, linecolor="white", annot=True, fmt="d",
                annot_kws={"fontsize": 8})

    ax.set_title("F7 · SHIELD-AI control × external-standard mapping (Table 4 density)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8.5)
    save(fig, "F7_mapping_heatmap")
    plt.close(fig)


# =====================================================================
# F8 — Threat trend bar chart with verified industry datapoints
# =====================================================================
def figure_f8_threat_trends() -> None:
    """Verified 2023–2025 industry datapoints from the bibliography."""
    # Each entry: (label, value%, source-ref tag for caption)
    data = [
        ("Vishing growth (H1→H2 2024)",          442, "B21 CrowdStrike GTR 2025"),
        ("Infostealer-delivery emails YoY",        84, "B17 IBM X-Force 2025"),
        ("Identity-based attacks (H1 2025)",       32, "B15 Microsoft DDR 2025"),
        ("Edge / VPN exploitation growth",         633, "B13 DBIR 2025 (3%→22%)"),
        ("Third-party breach involvement",         100, "B13 DBIR 2025 (15%→30%)"),
        ("Critical-infra share of investigated",   70, "B17 IBM X-Force 2025"),
        ("Malware-free intrusions (vs 40% 2019)",  79, "B21 CrowdStrike GTR 2025"),
        ("Orgs without AI gov. policy",            63, "D10 IBM Cost-of-Breach 2025"),
        ("Breached orgs lacking AI access ctrl",   97, "D10 IBM Cost-of-Breach 2025"),
    ]
    labels  = [d[0] for d in data]
    values  = [d[1] for d in data]
    sources = [d[2] for d in data]

    # Color-code by category
    cats = ["adversary cap.", "adversary cap.", "adversary cap.",
            "adversary cap.", "adversary cap.", "exposure",
            "stealth", "gov. gap", "gov. gap"]
    cat_colors = {
        "adversary cap.": "#d62728",
        "exposure":       "#ff7f0e",
        "stealth":        "#9467bd",
        "gov. gap":       "#1f77b4",
    }
    colors = [cat_colors[c] for c in cats]

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Reported value (% — see source tags)")
    ax.set_title("F8 · Verified industry datapoints (2024–2025)\n"
                 "All values traceable to specific report editions in the bibliography")
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)

    # Source labels at bar end
    for bar, src in zip(bars, sources):
        w = bar.get_width()
        ax.text(w + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                src, va="center", fontsize=7.5, color="black")

    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=cat_colors[c], label=c) for c in cat_colors]
    ax.legend(handles=handles, loc="lower right", frameon=True,
              title="Category", fontsize=8, title_fontsize=8)

    save(fig, "F8_threat_trends")
    plt.close(fig)


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    print("Generating SHIELD-AI quantitative figures...")
    figure_f6_roc()
    figure_f7_mapping_heatmap()
    figure_f8_threat_trends()
    print(f"\nAll figures written to: {OUT_DIR}")
