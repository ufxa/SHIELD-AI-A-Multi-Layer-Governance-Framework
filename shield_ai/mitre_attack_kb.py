"""Minimal MITRE ATT&CK knowledge base used as the RAG corpus.

The KB is implemented as a small inline dictionary so that the
experiment is fully reproducible without depending on the ATT&CK STIX
bundle. Each entry contains the technique identifier, the technique
name, a short description, the tactic, and a list of CICIDS-2017
labels for which the technique is the canonical mapping.

The text fields are used by the simulated RAG retriever as candidate
chunks; the metadata is used by the groundedness score gamma to
measure whether the LLM-predicted attack class is consistent with the
retrieved technique.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import math
import re


@dataclass(frozen=True)
class AttckEntry:
    technique_id: str
    name: str
    tactic: str
    description: str
    cicids_labels: Tuple[str, ...]


_KB: List[AttckEntry] = [
    AttckEntry(
        technique_id="T1499",
        name="Endpoint Denial of Service",
        tactic="Impact",
        description=(
            "Adversaries may perform endpoint denial of service attacks to degrade or block availability of "
            "services to users. Common vectors include flooding the target host with large numbers of malformed "
            "or oversized HTTP requests, exhausting socket pools and worker threads."
        ),
        cicids_labels=("DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest"),
    ),
    AttckEntry(
        technique_id="T1499.001",
        name="OS Exhaustion Flood",
        tactic="Impact",
        description=(
            "Endpoint denial-of-service via operating system resource exhaustion using a high volume of small "
            "TCP/UDP packets, typically observed as elevated PSH/ACK ratios and packets-per-second."
        ),
        cicids_labels=("DoS Hulk", "DoS GoldenEye"),
    ),
    AttckEntry(
        technique_id="T1499.004",
        name="Application or System Exploitation",
        tactic="Impact",
        description=(
            "Slow-rate attacks against application servers such as slowloris and slowhttptest keep connections "
            "alive while holding worker threads, characterised by long flow durations with very low packet "
            "counts."
        ),
        cicids_labels=("DoS slowloris", "DoS Slowhttptest"),
    ),
    AttckEntry(
        technique_id="T1498",
        name="Network Denial of Service",
        tactic="Impact",
        description=(
            "Network-based DDoS focuses on consuming the bandwidth available to the target network rather than "
            "host resources, observable as bursts of high packets-per-second from many distinct sources."
        ),
        cicids_labels=("DDoS",),
    ),
    AttckEntry(
        technique_id="T1498.001",
        name="Direct Network Flood",
        tactic="Impact",
        description=(
            "Direct network flooding produces sustained bursts of small fixed-size packets with very high "
            "flow packets per second and minimal payload."
        ),
        cicids_labels=("DDoS",),
    ),
    AttckEntry(
        technique_id="T1046",
        name="Network Service Discovery",
        tactic="Discovery",
        description=(
            "Adversaries attempt to get a listing of services running on a remote host by issuing connection "
            "attempts against a range of ports. Observable as flows with many SYN packets and very few "
            "completed handshakes."
        ),
        cicids_labels=("PortScan",),
    ),
    AttckEntry(
        technique_id="T1110",
        name="Brute Force",
        tactic="Credential Access",
        description=(
            "Adversaries may use brute force techniques to gain access to accounts when passwords are unknown "
            "or when password hashes are obtained. Manifested at the network as repeated short flows with "
            "similar packet counts and durations."
        ),
        cicids_labels=("FTP-Patator", "SSH-Patator", "Web Attack - Brute Force"),
    ),
    AttckEntry(
        technique_id="T1110.001",
        name="Password Guessing",
        tactic="Credential Access",
        description=(
            "Password guessing against FTP and SSH services via repeated authentication attempts; flows are "
            "short and similar in length with characteristic ACK/PSH ratios."
        ),
        cicids_labels=("FTP-Patator", "SSH-Patator"),
    ),
    AttckEntry(
        technique_id="T1110.003",
        name="Password Spraying",
        tactic="Credential Access",
        description=(
            "Password spraying against web authentication forms by submitting a small set of common passwords "
            "against many user names; flows show structured POST request patterns and elevated PSH counts."
        ),
        cicids_labels=("Web Attack - Brute Force",),
    ),
    AttckEntry(
        technique_id="T1059.007",
        name="Command and Scripting Interpreter: JavaScript",
        tactic="Execution",
        description=(
            "Cross-site scripting payloads injected via HTTP parameters causing the victim browser to execute "
            "attacker-controlled JavaScript."
        ),
        cicids_labels=("Web Attack - XSS",),
    ),
    AttckEntry(
        technique_id="T1190",
        name="Exploit Public-Facing Application",
        tactic="Initial Access",
        description=(
            "Exploitation of a public-facing application such as a web server, including SQL injection and "
            "memory-disclosure vulnerabilities like Heartbleed."
        ),
        cicids_labels=("Web Attack - SQL Injection", "Heartbleed"),
    ),
    AttckEntry(
        technique_id="T1583.006",
        name="Acquire Infrastructure: Web Services",
        tactic="Resource Development",
        description=(
            "Adversaries may register accounts with web services that can be used during targeting, including "
            "the rental of command-and-control infrastructure used by bots to beacon home."
        ),
        cicids_labels=("Bot",),
    ),
    AttckEntry(
        technique_id="T1071.001",
        name="Application Layer Protocol: Web Protocols",
        tactic="Command and Control",
        description=(
            "Bot C2 traffic over HTTP/HTTPS with periodic beacons; flows display long durations punctuated by "
            "small fixed-size requests."
        ),
        cicids_labels=("Bot",),
    ),
    AttckEntry(
        technique_id="T1078",
        name="Valid Accounts",
        tactic="Initial Access",
        description=(
            "Adversaries may obtain and abuse credentials of existing accounts to gain initial access. "
            "Infiltration flows often appear as long-duration low-rate sessions consistent with legitimate "
            "remote access."
        ),
        cicids_labels=("Infiltration",),
    ),
]


def all_entries() -> Tuple[AttckEntry, ...]:
    return tuple(_KB)


def by_id(technique_id: str) -> AttckEntry | None:
    for entry in _KB:
        if entry.technique_id == technique_id:
            return entry
    return None


# --- Naive retriever: term frequency on tokenised description -----------------

_token_re = re.compile(r"[A-Za-z][A-Za-z\-]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _token_re.findall(text)]


def _tf(tokens: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not tokens:
        return out
    inv = 1.0 / len(tokens)
    for t in tokens:
        out[t] = out.get(t, 0.0) + inv
    return out


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a).intersection(b)
    num = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return num / (na * nb)


def retrieve(query: str, k: int = 3) -> List[Tuple[AttckEntry, float]]:
    """Return the top-k ATT&CK entries ranked by cosine on description."""
    q_tf = _tf(_tokenize(query))
    scored = []
    for entry in _KB:
        score = _cosine(q_tf, _tf(_tokenize(entry.description + " " + entry.name)))
        scored.append((entry, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
