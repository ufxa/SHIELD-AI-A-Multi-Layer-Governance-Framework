# SHIELD-AI: A Multi-Layer Governance Framework

Reference implementation, experimental harness, and reproducibility
artefacts for the paper:

> **SHIELD-AI: A Multi-Layer Governance Framework for Generative AI in
> Security Operations Centres.**

The repository contains the four-layer pipeline (L1 ingest, L2 LLM/RAG
with reliability triple, L3 decision-theoretic HITL routing, L4
append-only Merkle-chained audit log) and a CICIDS-2017-like
experimental harness that produces the metrics and figures reported
in Section 9 of the paper.

---

## Repository layout

```
shield_ai/
  config.py             reliability weights, thresholds, routing costs
  synthetic_data.py     CICIDS-2017-like generator (24 features, 14+1 classes)
  cicids_loader.py      drop-in loader for the real CICIDS-2017 CSV files
  mitre_attack_kb.py    MITRE ATT&CK techniques used as RAG corpus
  layer1_ingest.py      alert-text synthesis and PII gate (Layer 1)
  layer2_baselines.py   rule-based and Random Forest baselines
  layer2_llm_rag.py     LLM/RAG simulator computing <theta, sigma, gamma>
  layer3_hitl.py        PC3 decision-theoretic routing (AUTO / HITL / REJECT)
  layer4_audit.py       append-only signed Merkle audit log + PC5 verifier
  metrics.py            evaluation metrics used in the paper
experiments/
  run_experiment.py     end-to-end runner (writes metrics.json + audit log)
  figures.py            renders E1-E6 from the runner output
tests/
  test_audit_chain.py   E1/E2/E3 invariants + PC5 verifier
  test_reliability.py   weight sanity + PC3 routing zones
results/                runner outputs and figures (generated)
```

## Quick start

```bash
git clone https://github.com/ufxa/SHIELD-AI-A-Multi-Layer-Governance-Framework.git
cd SHIELD-AI-A-Multi-Layer-Governance-Framework
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run unit tests
python tests/test_audit_chain.py
python tests/test_reliability.py

# Run end-to-end experiment (about 1 minute on a recent laptop)
python -m experiments.run_experiment --n-samples 60000 --seed 42

# Render the six paper figures
python -m experiments.figures
```

Outputs are written under `results/`:

| File | Content |
|---|---|
| `results/metrics.json` | All metrics referenced in the paper |
| `results/per_class_metrics.csv` | Per-attack-class precision / recall / F1 |
| `results/hitl_by_class.csv` | AUTO / HITL / REJECT shares per attack class |
| `results/audit_log.jsonl` | Layer-4 signed Merkle-chained audit log |
| `results/merkle_batches.json` | Published Merkle roots (batch size 1024) |
| `results/figures/fig_e{1..6}.{pdf,png}` | Paper figures E1-E6 |

## Reproducing the paper numbers

```bash
python -m experiments.run_experiment \
    --n-samples 60000 \
    --seed 42 \
    --attack-ratio 0.30 \
    --train-fraction 0.75
```

Headline numbers obtained with seed 42:

| System | binary F1 | binary FPR | ROC AUC |
|---|---|---|---|
| Rule-based | 0.516 | 0.375 | 0.750 |
| Random Forest | 0.985 | 0.004 | 0.999 |
| SHIELD-AI L2 (LLM/RAG) | 0.984 | 0.004 | 0.998 |

PC3 routing on SHIELD-AI: AUTO 68.97 %, HITL 29.69 %, REJECT 1.35 %.

Layer-4 invariants: chain verification on 15 000 records passes;
50/50 random records also pass the record-level PC5 Merkle verifier.

## Using the real CICIDS-2017 CSV files

By default `synthetic_data.generate` is used so that the experiment is
runnable without external downloads. To replace it with the real
CICIDS-2017 CSVs, place the daily files (Monday-WorkingHours.pcap_ISCX.csv,
etc.) under `data/cicids-2017/` and switch the runner's first stage to
the loader:

```python
from shield_ai import cicids_loader
df = cicids_loader.load_directory("data/cicids-2017/")
```

The loader is a thin wrapper around `pandas.read_csv` that performs the
same column renaming and label normalisation expected by Layer 1.

## Architecture summary

The four layers map onto specific functions in this repository:

1. **L1 (`layer1_ingest`)** consumes raw CICFlowMeter records and
   produces SIEM-style alert text plus a PII audit list.  All
   subsequent layers reason over the alert text.
2. **L2 (`layer2_*`)** runs in three configurations: rule-based,
   classical-ML, and the LLM/RAG simulator.  The LLM path computes
   the reliability triple `<theta, sigma, gamma>` where gamma is
   decomposed into the sub-signals CC, ES, GD, RS as in Section 5 of
   the paper.
3. **L3 (`layer3_hitl`)** applies PC3 routing.  The AUTO threshold
   `r_auto` is derived from the cost model in `config.RoutingCosts`
   and equals 0.80 with the default parameters.
4. **L4 (`layer4_audit`)** appends each decision to an HMAC-signed
   append-only log.  Records are batched into Merkle trees and the
   public roots are written to `results/merkle_batches.json`.  The
   record-level verifier implements Algorithm PC5 of the paper.

## License

Apache License 2.0.  See `LICENSE`.

## Citation

```bibtex
@article{shieldai2026,
  title={{SHIELD-AI}: {A} Multi-Layer Governance Framework for Generative {AI} in Security Operations Centres},
  author={{Anonymous}},
  journal={Under review},
  year={2026},
  note={Reference implementation: https://github.com/ufxa/SHIELD-AI-A-Multi-Layer-Governance-Framework}
}
```
