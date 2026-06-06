"""SHIELD-AI Layer 4: append-only Merkle-chained audit log.

This module implements the engineering invariants E1 (append-only),
E2 (Merkle chaining), and E3 (time-monotone) described in
Section 7 of the paper.  Each record is signed with an HMAC simulating
HSM-backed signing; the records are also threaded into a hash chain
so that any tampering breaks downstream verification.  On every
`merkle_batch_size` records, a Merkle root is published; the
verifier `verify_record` re-derives the root from a leaf and a
sibling-hash proof, matching the public-verifier pseudocode of
Algorithm PC5.

The audit log is persisted as JSON lines on disk so that the
verifier can run as an independent CLI tool against the same file.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from .config import AuditConfig, DEFAULT_AUDIT
from .layer2_llm_rag import LLMDecision
from .layer3_hitl import RoutingOutcome, Route


# ---- Record schema ----------------------------------------------------------

@dataclass
class AuditRecord:
    sequence: int
    timestamp_ns: int
    flow_id: int
    true_label: str
    predicted_label: str
    route: str
    composite_R: float
    theta: float
    sigma: float
    gamma: float
    gamma_prime: float
    rationale: str
    retrieved_top_techniques: List[str]
    prev_hash: str
    record_hash: str = ""
    signature: str = ""

    def canonical_bytes(self) -> bytes:
        d = asdict(self)
        d.pop("record_hash", None)
        d.pop("signature", None)
        return json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")


# ---- Merkle helpers ---------------------------------------------------------

def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _hmac(secret: bytes, b: bytes) -> str:
    return hmac.new(secret, b, hashlib.sha256).hexdigest()


def _merkle_root(leaves: List[str]) -> str:
    if not leaves:
        return "0" * 64
    layer = [bytes.fromhex(h) for h in leaves]
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])  # duplicate last
        layer = [
            hashlib.sha256(layer[i] + layer[i + 1]).digest()
            for i in range(0, len(layer), 2)
        ]
    return layer[0].hex()


def _merkle_proof(leaves: List[str], index: int) -> List[Tuple[str, str]]:
    """Return list of (sibling_hash_hex, side) where side is 'L' or 'R'."""
    proof: List[Tuple[str, str]] = []
    if not leaves:
        return proof
    layer = [bytes.fromhex(h) for h in leaves]
    idx = index
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])
        if idx % 2 == 0:
            sibling = layer[idx + 1]
            proof.append((sibling.hex(), "R"))
        else:
            sibling = layer[idx - 1]
            proof.append((sibling.hex(), "L"))
        layer = [
            hashlib.sha256(layer[i] + layer[i + 1]).digest()
            for i in range(0, len(layer), 2)
        ]
        idx //= 2
    return proof


def _verify_proof(leaf_hex: str, proof: List[Tuple[str, str]], expected_root: str) -> bool:
    cur = bytes.fromhex(leaf_hex)
    for sib_hex, side in proof:
        sib = bytes.fromhex(sib_hex)
        if side == "R":
            cur = hashlib.sha256(cur + sib).digest()
        else:
            cur = hashlib.sha256(sib + cur).digest()
    return cur.hex() == expected_root


# ---- The audit log ----------------------------------------------------------

@dataclass
class MerkleBatch:
    batch_index: int
    start_sequence: int
    end_sequence: int
    leaf_hashes: List[str]
    root_hash: str
    timestamp_ns: int


@dataclass
class AuditLog:
    config: AuditConfig = field(default_factory=lambda: DEFAULT_AUDIT)
    records: List[AuditRecord] = field(default_factory=list)
    batches: List[MerkleBatch] = field(default_factory=list)
    _last_hash: str = "0" * 64
    _pending_leaves: List[str] = field(default_factory=list)
    _pending_start_seq: int = 0

    # ---- append --------------------------------------------------------

    def append(
        self,
        true_label: str,
        decision: LLMDecision,
        outcome: RoutingOutcome,
        flow_id: int,
    ) -> AuditRecord:
        seq = len(self.records)
        ts = time.time_ns()
        record = AuditRecord(
            sequence=seq,
            timestamp_ns=ts,
            flow_id=flow_id,
            true_label=true_label,
            predicted_label=decision.predicted_label,
            route=outcome.route.value if isinstance(outcome.route, Route) else str(outcome.route),
            composite_R=float(outcome.composite_R),
            theta=float(decision.triple.theta),
            sigma=float(decision.triple.sigma),
            gamma=float(decision.triple.gamma),
            gamma_prime=float(decision.triple.gamma_prime),
            rationale=outcome.rationale,
            retrieved_top_techniques=[e.technique_id for e, _ in decision.retrieved[:3]],
            prev_hash=self._last_hash,
        )
        canonical = record.canonical_bytes()
        record.record_hash = _sha256(canonical + self._last_hash.encode("ascii"))
        if self.config.sign_records:
            record.signature = _hmac(self.config.hmac_secret, record.record_hash.encode("ascii"))
        self.records.append(record)
        self._last_hash = record.record_hash
        self._pending_leaves.append(record.record_hash)
        if len(self._pending_leaves) >= self.config.merkle_batch_size:
            self._publish_root()
        return record

    def _publish_root(self) -> None:
        if not self._pending_leaves:
            return
        root = _merkle_root(self._pending_leaves)
        start_seq = self._pending_start_seq
        end_seq = start_seq + len(self._pending_leaves) - 1
        self.batches.append(
            MerkleBatch(
                batch_index=len(self.batches),
                start_sequence=start_seq,
                end_sequence=end_seq,
                leaf_hashes=list(self._pending_leaves),
                root_hash=root,
                timestamp_ns=time.time_ns(),
            )
        )
        self._pending_start_seq = end_seq + 1
        self._pending_leaves.clear()

    def flush(self) -> None:
        self._publish_root()

    # ---- chain-level verification --------------------------------------

    def verify_chain(self) -> Tuple[bool, Optional[int], str]:
        """Re-derive the hash chain from scratch and compare every record."""
        prev = "0" * 64
        for r in self.records:
            # check signature
            if self.config.sign_records:
                expected_sig = _hmac(self.config.hmac_secret, r.record_hash.encode("ascii"))
                if not hmac.compare_digest(expected_sig, r.signature):
                    return False, r.sequence, "bad signature"
            # check hash
            expected_hash = _sha256(r.canonical_bytes() + prev.encode("ascii"))
            if expected_hash != r.record_hash:
                return False, r.sequence, "hash mismatch"
            if r.prev_hash != prev:
                return False, r.sequence, "prev_hash mismatch"
            prev = r.record_hash
        return True, None, "OK"

    # ---- record-level verification (Algorithm PC5) ---------------------

    def verify_record(self, sequence: int) -> Tuple[bool, str]:
        if sequence < 0 or sequence >= len(self.records):
            return False, "out of range"
        record = self.records[sequence]
        # find batch
        batch = next((b for b in self.batches if b.start_sequence <= sequence <= b.end_sequence), None)
        if batch is None:
            return False, "no published root for this record yet"
        idx = sequence - batch.start_sequence
        proof = _merkle_proof(batch.leaf_hashes, idx)
        ok = _verify_proof(batch.leaf_hashes[idx], proof, batch.root_hash)
        # Also re-check signature and record_hash
        if self.config.sign_records:
            expected_sig = _hmac(self.config.hmac_secret, record.record_hash.encode("ascii"))
            if not hmac.compare_digest(expected_sig, record.signature):
                return False, "signature mismatch"
        if not ok:
            return False, "merkle proof failed"
        return True, "VALID"

    # ---- persistence ---------------------------------------------------

    def to_jsonl(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for r in self.records:
                fh.write(json.dumps(asdict(r), separators=(",", ":")) + "\n")

    def batches_to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump([asdict(b) for b in self.batches], fh, indent=2)
