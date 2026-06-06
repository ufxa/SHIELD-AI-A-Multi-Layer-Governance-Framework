"""SHIELD-AI Layer 1: ingest, normalisation, alert synthesis, PII gate.

Layer 1 is responsible for:
  * Loading raw CICFlowMeter records from disk or the synthetic
    generator.
  * Producing a SIEM-style textual alert from each flow record so that
    Layer 2 can reason over natural-language input.
  * Running a PII scan over alerts before they leave the trust
    perimeter into Layer 2 (assumption A8 in the paper).

The PII gate is a deterministic regex-based scanner; on real
deployments it would be replaced by Presidio or a comparable
production-grade engine, but the contract (`pii_findings` list and
a redacted alert text) is identical.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from .synthetic_data import FEATURE_COLUMNS


# ----- PII scanner ----------------------------------------------------------

_PII_PATTERNS = {
    "EMAIL": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "CPF":   re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"),
    "CNPJ":  re.compile(r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b"),
    "IPV4":  re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "MAC":   re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b"),
    "CC":    re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b"),
}


@dataclass(frozen=True)
class PIIFinding:
    label: str
    span: str
    start: int
    end: int


def scan_pii(text: str) -> List[PIIFinding]:
    findings: List[PIIFinding] = []
    for label, pat in _PII_PATTERNS.items():
        for m in pat.finditer(text):
            findings.append(PIIFinding(label=label, span=m.group(0), start=m.start(), end=m.end()))
    return findings


def redact(text: str, findings: List[PIIFinding]) -> str:
    if not findings:
        return text
    # Apply replacements from rightmost to leftmost to keep offsets valid.
    out = text
    for f in sorted(findings, key=lambda x: x.start, reverse=True):
        out = out[: f.start] + f"<{f.label}>" + out[f.end :]
    return out


# ----- Alert text synthesis -------------------------------------------------

_ALERT_TEMPLATE = (
    "[SHIELD-AI L1] flow_id={flow_id} ts={ts}\n"
    "  duration_us={duration:,} fwd_pkts={fwd_pkts} bwd_pkts={bwd_pkts}\n"
    "  flow_pps={pps:.1f} flow_bps={bps:.1f}\n"
    "  pkt_len[min/mean/max]={lmin:.0f}/{lmean:.0f}/{lmax:.0f}\n"
    "  flags FIN={fin} SYN={syn} RST={rst} PSH={psh} ACK={ack}\n"
    "  init_win_fwd={win} avg_pkt_size={aps:.1f}\n"
    "  iat_mean_us={iat:.0f} iat_std_us={iats:.0f}"
)


def synthesize_alert(row: pd.Series) -> str:
    """Render a SIEM-style alert text for one flow row."""
    return _ALERT_TEMPLATE.format(
        flow_id=int(row["FlowID"]),
        ts=row["Timestamp"].isoformat() if hasattr(row["Timestamp"], "isoformat") else str(row["Timestamp"]),
        duration=int(row["Flow Duration"]),
        fwd_pkts=int(row["Total Fwd Packets"]),
        bwd_pkts=int(row["Total Backward Packets"]),
        pps=float(row["Flow Packets/s"]),
        bps=float(row["Flow Bytes/s"]),
        lmin=float(row["Min Packet Length"]),
        lmean=float(row["Packet Length Mean"]),
        lmax=float(row["Max Packet Length"]),
        fin=int(row["FIN Flag Count"]),
        syn=int(row["SYN Flag Count"]),
        rst=int(row["RST Flag Count"]),
        psh=int(row["PSH Flag Count"]),
        ack=int(row["ACK Flag Count"]),
        win=int(row["Init_Win_bytes_forward"]),
        aps=float(row["Average Packet Size"]),
        iat=float(row["Flow IAT Mean"]),
        iats=float(row["Flow IAT Std"]),
    )


# ----- High-level pipeline ---------------------------------------------------

@dataclass
class IngestResult:
    features: pd.DataFrame
    labels: pd.Series
    alert_texts: List[str]
    pii_audit: List[List[PIIFinding]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.features)


def run_ingest(df: pd.DataFrame, redact_pii: bool = True) -> IngestResult:
    """Apply Layer-1 ingest to a CICFlowMeter-shaped DataFrame.

    Returns numeric features (the FEATURE_COLUMNS subset), labels,
    SIEM-style alert texts, and per-row PII audit records.
    """
    features = df[list(FEATURE_COLUMNS)].copy()
    labels = df["Label"].copy()
    alert_texts: List[str] = []
    pii_audit: List[List[PIIFinding]] = []
    for _, row in df.iterrows():
        text = synthesize_alert(row)
        findings = scan_pii(text)
        if redact_pii:
            text = redact(text, findings)
        alert_texts.append(text)
        pii_audit.append(findings)
    return IngestResult(features=features, labels=labels, alert_texts=alert_texts, pii_audit=pii_audit)


def pii_summary(audit: List[List[PIIFinding]]) -> Dict[str, int]:
    """Aggregate PII findings by label across an audit list."""
    counts: Dict[str, int] = {}
    for findings in audit:
        for f in findings:
            counts[f.label] = counts.get(f.label, 0) + 1
    return counts
