"""End-to-end SHIELD-AI experiment runner.

Executes the full L1->L2->L3->L4 pipeline against a labelled
CICFlowMeter-shaped dataset, computes the metrics referenced in the
empirical-evaluation section of the paper, and serialises the
results as JSON / CSV under ``results/``.  Figures are produced by
the companion module ``experiments.figures``.

Usage:
    python -m experiments.run_experiment --n-samples 60000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Allow running as `python -m experiments.run_experiment` from repo root.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from shield_ai import (  # noqa: E402  - after sys.path mutation
    config,
    synthetic_data,
    layer1_ingest,
    layer2_baselines,
    layer2_llm_rag,
    layer3_hitl,
    layer4_audit,
    metrics,
)


# -----------------------------------------------------------------------------

def _ensure_dirs(cfg: config.ExperimentConfig) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.figures_dir, exist_ok=True)


def _print_banner(text: str) -> None:
    bar = "=" * max(60, len(text) + 4)
    print(bar)
    print(f"  {text}")
    print(bar)


def _to_serialisable(obj: Any) -> Any:
    """Convert numpy and pandas types to vanilla Python for json.dumps."""
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {k: _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serialisable(v) for v in obj]
    return obj


# -----------------------------------------------------------------------------

def run(cfg: config.ExperimentConfig) -> Dict[str, Any]:
    _ensure_dirs(cfg)
    results: Dict[str, Any] = {
        "config": asdict(cfg),
        "timings_seconds": {},
        "dataset": {},
        "rule_based": {},
        "ml_baseline": {},
        "shield_ai_full": {},
        "audit": {},
    }

    # ---- 1. Generate dataset ------------------------------------------
    _print_banner("Stage 1: synthetic CICIDS-2017-like data generation")
    t0 = time.perf_counter()
    df = synthetic_data.generate(
        synthetic_data.GeneratorOptions(
            n_samples=cfg.n_samples,
            attack_ratio=cfg.attack_ratio,
            random_seed=cfg.random_seed,
        )
    )
    results["timings_seconds"]["data_generation"] = time.perf_counter() - t0
    summ = synthetic_data.summary(df)
    print(summ.to_string(index=False))
    results["dataset"]["per_class"] = summ.to_dict(orient="records")
    results["dataset"]["n_rows"] = int(len(df))

    train_df, test_df = synthetic_data.train_test_split_temporal(df, cfg.train_fraction)
    results["dataset"]["train_rows"] = int(len(train_df))
    results["dataset"]["test_rows"] = int(len(test_df))

    # ---- 2. Layer 1: ingest ------------------------------------------
    _print_banner("Stage 2: Layer 1 ingest (alert synthesis + PII gate)")
    t0 = time.perf_counter()
    ingest_train = layer1_ingest.run_ingest(train_df, redact_pii=True)
    ingest_test = layer1_ingest.run_ingest(test_df, redact_pii=True)
    results["timings_seconds"]["layer1_ingest"] = time.perf_counter() - t0
    pii_counts = layer1_ingest.pii_summary(ingest_train.pii_audit + ingest_test.pii_audit)
    print(f"  PII findings across train+test: {pii_counts}")
    results["dataset"]["pii_findings"] = pii_counts

    # ---- 3. Layer 2 baselines: Rule-based + Random Forest -------------
    _print_banner("Stage 3a: Layer 2 rule-based baseline")
    rb = layer2_baselines.RuleBasedClassifier()
    rb_pred, rb_p_attack, rb_latency = rb.predict(ingest_test.features)
    rb_binary = metrics.binary_metrics(ingest_test.labels.values, rb_pred, rb_p_attack)
    rb_multi = metrics.multiclass_metrics(
        ingest_test.labels.values, rb_pred, labels=config.ATTACK_CLASSES,
    )
    rb_latency_per_item = [rb_latency / max(1, len(rb_pred))] * len(rb_pred)
    results["rule_based"] = {
        "binary": rb_binary,
        "multiclass": rb_multi,
        "latency": metrics.latency_summary(rb_latency_per_item),
        "total_seconds": rb_latency,
    }
    print(f"  Rule-based binary F1={rb_binary['f1']:.3f} FPR={rb_binary['fpr']:.3f}")

    _print_banner("Stage 3b: Layer 2 classical ML baseline (Random Forest)")
    ml = layer2_baselines.MLBaseline(n_jobs=cfg.n_jobs_classifier, random_seed=cfg.random_seed)
    t0 = time.perf_counter()
    ml.fit(ingest_train.features, ingest_train.labels)
    results["timings_seconds"]["ml_fit"] = time.perf_counter() - t0
    ml_pred, ml_p_attack, ml_latency = ml.predict(ingest_test.features)
    ml_proba_full, ml_classes = ml.predict_proba_full(ingest_test.features)
    ml_binary = metrics.binary_metrics(ingest_test.labels.values, ml_pred, ml_p_attack)
    ml_multi = metrics.multiclass_metrics(
        ingest_test.labels.values, ml_pred, labels=config.ATTACK_CLASSES,
    )
    ml_latency_per_item = [ml_latency / max(1, len(ml_pred))] * len(ml_pred)
    results["ml_baseline"] = {
        "binary": ml_binary,
        "multiclass": ml_multi,
        "latency": metrics.latency_summary(ml_latency_per_item),
        "total_seconds": ml_latency,
    }
    print(f"  ML binary F1={ml_binary['f1']:.3f} FPR={ml_binary['fpr']:.3f}")

    # ---- 4. Layer 2 LLM/RAG simulator ---------------------------------
    _print_banner("Stage 4: Layer 2 LLM/RAG simulator with reliability triple")
    t0 = time.perf_counter()
    llm = layer2_llm_rag.LLMRAGLayer()
    llm_decisions, llm_total_latency = llm.classify_batch(
        ingest_test.alert_texts,
        ml_proba_full,
        list(ml_classes),
    )
    results["timings_seconds"]["layer2_llm"] = time.perf_counter() - t0

    llm_pred = np.array([d.predicted_label for d in llm_decisions])
    llm_p_attack = np.array(
        [1.0 - (d.triple.theta if d.predicted_label == "BENIGN" else 0.0) for d in llm_decisions]
    )
    # Use a more meaningful score: composite R as attack-likelihood when predicted is attack
    composite_scores = np.array([d.triple.composite() for d in llm_decisions])
    pred_attack = np.array([d.predicted_label != "BENIGN" for d in llm_decisions])
    llm_score = np.where(pred_attack, composite_scores, 1.0 - composite_scores)

    llm_binary = metrics.binary_metrics(ingest_test.labels.values, llm_pred, llm_score)
    llm_multi = metrics.multiclass_metrics(
        ingest_test.labels.values, llm_pred, labels=config.ATTACK_CLASSES,
    )
    llm_latency_per_item = [d.latency_seconds for d in llm_decisions]
    llm_gamma = metrics.gamma_distribution(llm_decisions)
    print(f"  LLM binary F1={llm_binary['f1']:.3f} FPR={llm_binary['fpr']:.3f}")
    print(f"  gamma mean={llm_gamma['mean']:.3f} p50={llm_gamma['p50']:.3f}")

    # ---- 5. Layer 3 routing ---------------------------------------------
    _print_banner("Stage 5: Layer 3 decision-theoretic HITL routing")
    t0 = time.perf_counter()
    outcomes = layer3_hitl.route_batch(llm_decisions)
    results["timings_seconds"]["layer3_routing"] = time.perf_counter() - t0
    routes = [o.route.value for o in outcomes]
    route_dist = metrics.routing_distribution(routes)
    hitl_by_class = metrics.hitl_rate_by_class(ingest_test.labels.values, routes)
    print(f"  Route distribution: {route_dist}")

    # ---- 6. Layer 4 audit log ------------------------------------------
    _print_banner("Stage 6: Layer 4 Merkle-chained audit log")
    t0 = time.perf_counter()
    audit = layer4_audit.AuditLog()
    flow_ids = test_df["FlowID"].values
    for decision, outcome, flow_id, true_label in zip(llm_decisions, outcomes, flow_ids, ingest_test.labels.values):
        audit.append(true_label, decision, outcome, int(flow_id))
    audit.flush()
    audit_seconds = time.perf_counter() - t0
    chain_ok, bad_idx, chain_msg = audit.verify_chain()
    # Spot-check 50 random record-level verifications
    rng = np.random.default_rng(cfg.random_seed)
    sample_idx = rng.choice(len(audit.records), size=min(50, len(audit.records)), replace=False)
    spot_pass = 0
    for idx in sample_idx:
        ok, _ = audit.verify_record(int(idx))
        if ok:
            spot_pass += 1
    print(f"  Chain integrity: {chain_ok} ({chain_msg}) records={len(audit.records)}")
    print(f"  Spot-check verify {spot_pass}/{len(sample_idx)} records pass PC5 verifier")

    audit_path = os.path.join(cfg.output_dir, "audit_log.jsonl")
    batches_path = os.path.join(cfg.output_dir, "merkle_batches.json")
    audit.to_jsonl(audit_path)
    audit.batches_to_json(batches_path)
    results["audit"] = {
        "chain_ok": chain_ok,
        "chain_msg": chain_msg,
        "bad_idx": bad_idx,
        "records": len(audit.records),
        "batches": len(audit.batches),
        "merkle_batch_size": audit.config.merkle_batch_size,
        "audit_path": audit_path,
        "batches_path": batches_path,
        "audit_seconds": audit_seconds,
        "audit_latency_per_record_ms": (audit_seconds / max(1, len(audit.records))) * 1000.0,
        "spot_verify_pass": int(spot_pass),
        "spot_verify_total": int(len(sample_idx)),
    }

    # ---- 7. Save SHIELD-AI metrics + auxiliary data --------------------
    results["shield_ai_full"] = {
        "binary": llm_binary,
        "multiclass": llm_multi,
        "latency": metrics.latency_summary(llm_latency_per_item),
        "gamma_overall": llm_gamma,
        "gamma_per_class": metrics.gamma_per_class(llm_decisions, ingest_test.labels.values).to_dict(orient="records"),
        "route_distribution": route_dist,
        "hitl_by_class": hitl_by_class.to_dict(orient="records"),
        "total_seconds_llm": llm_total_latency,
    }

    # Persist per-row test predictions for figure generation
    test_predictions = pd.DataFrame(
        {
            "flow_id": test_df["FlowID"].values,
            "true_label": ingest_test.labels.values,
            "rule_pred": rb_pred,
            "rule_score": rb_p_attack,
            "ml_pred": ml_pred,
            "ml_score": ml_p_attack,
            "llm_pred": llm_pred,
            "llm_score": llm_score,
            "theta": [d.triple.theta for d in llm_decisions],
            "sigma": [d.triple.sigma for d in llm_decisions],
            "gamma": [d.triple.gamma for d in llm_decisions],
            "gamma_prime": [d.triple.gamma_prime for d in llm_decisions],
            "composite_R": [d.triple.composite() for d in llm_decisions],
            "route": routes,
            "llm_latency_seconds": [d.latency_seconds for d in llm_decisions],
        }
    )
    pred_path = os.path.join(cfg.output_dir, "test_predictions.parquet")
    try:
        test_predictions.to_parquet(pred_path, index=False)
    except Exception:
        # parquet engine may be unavailable; fall back to csv
        pred_path = os.path.join(cfg.output_dir, "test_predictions.csv")
        test_predictions.to_csv(pred_path, index=False)
    results["test_predictions_path"] = pred_path

    # ---- 8. Save metrics.json ------------------------------------------
    metrics_path = os.path.join(cfg.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(_to_serialisable(results), fh, indent=2)
    print(f"  metrics.json written to {metrics_path}")

    # ---- 9. Per-class table for the paper ------------------------------
    per_class_path = os.path.join(cfg.output_dir, "per_class_metrics.csv")
    metrics.per_class_table(
        ingest_test.labels.values, llm_pred, labels=config.ATTACK_CLASSES
    ).to_csv(per_class_path, index=False)
    print(f"  per_class_metrics.csv written to {per_class_path}")

    # Per-class HITL table
    hitl_path = os.path.join(cfg.output_dir, "hitl_by_class.csv")
    hitl_by_class.to_csv(hitl_path, index=False)

    return results


def parse_args() -> config.ExperimentConfig:
    p = argparse.ArgumentParser(description="SHIELD-AI experiment runner")
    p.add_argument("--n-samples", type=int, default=config.DEFAULT_EXPERIMENT.n_samples)
    p.add_argument("--seed", type=int, default=config.DEFAULT_EXPERIMENT.random_seed)
    p.add_argument("--attack-ratio", type=float, default=config.DEFAULT_EXPERIMENT.attack_ratio)
    p.add_argument("--output-dir", type=str, default=config.DEFAULT_EXPERIMENT.output_dir)
    p.add_argument("--figures-dir", type=str, default=config.DEFAULT_EXPERIMENT.figures_dir)
    p.add_argument("--train-fraction", type=float, default=config.DEFAULT_EXPERIMENT.train_fraction)
    args = p.parse_args()
    return config.ExperimentConfig(
        random_seed=args.seed,
        n_samples=args.n_samples,
        attack_ratio=args.attack_ratio,
        train_fraction=args.train_fraction,
        test_fraction=1.0 - args.train_fraction,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
    )


if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)
