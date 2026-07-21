"""
SHIELD-AI -- F8 threat trends, versao IEEE TDSC
Dimensoes: largura de 2 colunas IEEE (7.16in x 3.2in)
Labels das fontes ficam DENTRO da barra (se cabem) ou colados ao lado esquerdo da margem direita.
Saida: build/F8_threat_trends_ieee.png e .pdf
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
os.makedirs(OUT_DIR, exist_ok=True)

# --- dados ---
data = [
    ("Vishing growth (H1→H2 2024)",         442, "CrowdStrike GTR 2025",  "adversary"),
    ("Infostealer emails YoY",                84, "IBM X-Force 2025",      "adversary"),
    ("Identity-based attacks (H1 2025)",      32, "Microsoft MDDR 2025",   "adversary"),
    ("Edge/VPN exploitation growth",         633, "DBIR 2025 (3%→22%)",   "adversary"),
    ("Third-party breach involvement",        100, "DBIR 2025 (15%→30%)", "adversary"),
    ("Critical-infra share investigated",     70, "IBM X-Force 2025",      "exposure"),
    ("Malware-free intrusions (2025)",        79, "CrowdStrike GTR 2025",  "stealth"),
    ("Orgs without AI gov. policy",           63, "IBM Cost-of-Breach 2025","gov_gap"),
    ("Breached orgs lacking AI access ctrl",  97, "IBM Cost-of-Breach 2025","gov_gap"),
]

labels  = [d[0] for d in data]
values  = [d[1] for d in data]
sources = [d[2] for d in data]
cats    = [d[3] for d in data]

palette = {
    "adversary": "#c0392b",
    "exposure":  "#e67e22",
    "stealth":   "#8e44ad",
    "gov_gap":   "#2980b9",
}
colors = [palette[c] for c in cats]

# --- estilo IEEE ---
mpl.rcParams.update({
    "font.family":      "serif",
    "font.size":        8,
    "axes.titlesize":   8.5,
    "axes.labelsize":   8,
    "xtick.labelsize":  7.5,
    "ytick.labelsize":  8,
    "legend.fontsize":  7.5,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.04,
})

# 7.16in = largura de 2 colunas IEEE com espaco entre elas
fig, ax = plt.subplots(figsize=(7.16, 3.1))

y = np.arange(len(labels))
bars = ax.barh(y, values, color=colors, edgecolor="white",
               linewidth=0.3, height=0.65)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=7.8)
ax.invert_yaxis()
ax.set_xlabel("Reported percentage change (%)", fontsize=8)
ax.set_xlim(0, 730)
ax.set_title(
    "AI-augmented threat indicators (2024–2025)  —  values traceable to cited reports",
    fontsize=8.5, pad=4
)
ax.grid(True, axis="x", linestyle=":", alpha=0.4, linewidth=0.6)
ax.spines[["top", "right"]].set_visible(False)

# Labels das fontes: dentro da barra se valor > 180, senao fora (mas dentro do xlim)
THRESHOLD = 180
for bar, src, val in zip(bars, sources, values):
    bw = bar.get_width()
    by = bar.get_y() + bar.get_height() / 2
    if val >= THRESHOLD:
        # dentro da barra, alinhado a direita
        ax.text(bw - 6, by, src, va="center", ha="right",
                fontsize=6.8, color="white", fontweight="bold")
    else:
        # fora da barra, mas limitado ao xlim
        ax.text(min(bw + 6, 680), by, src, va="center", ha="left",
                fontsize=6.8, color="#333333")

# Legenda
legend_handles = [
    Patch(facecolor=palette["adversary"], label="Adversary capability"),
    Patch(facecolor=palette["exposure"],  label="Exposure"),
    Patch(facecolor=palette["stealth"],   label="Stealth / evasion"),
    Patch(facecolor=palette["gov_gap"],   label="Governance gap"),
]
ax.legend(handles=legend_handles, loc="lower right", frameon=True,
          framealpha=0.9, edgecolor="#cccccc",
          title="Category", title_fontsize=7.5,
          fontsize=7.5, handlelength=1.2, handleheight=0.9)

plt.tight_layout(pad=0.3)

base = os.path.join(OUT_DIR, "F8_threat_trends_ieee")
fig.savefig(base + ".pdf")
fig.savefig(base + ".png")
plt.close(fig)
print(f"Salvo: {base}.pdf / .png")
