"""
Offline eval harness.

Two separate questions this answers, deliberately kept distinct:

1. Pipeline accuracy (run_eval): does the pipeline's output match the
   golden dataset's ground truth? This is the regression check a prompt
   or model change should be run against before shipping.

2. Judge alignment (run_judge_eval): does the eval_agent's confidence
   actually correlate with whether the pipeline got it right? A judge
   that's confidently wrong is worse than no judge at all -- the retry
   loop trusts it. Counting how often retries fire (which the live
   metrics already do) says nothing about whether the judge's calls are
   *correct*; this does.

Run: python -m backend.eval.run_eval
Requires GROQ_API_KEY (all four pipelines are LLM-based) -- not runnable
in this sandbox, which has no network path to Groq's API. See
tests/test_eval_harness.py for coverage of the scoring/tracking logic
itself using injected fake pipelines, which needs no network access.
"""
import argparse
import json
import logging
import os
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from backend.eval.golden_dataset import ALL_EXAMPLES, GoldenExample

logger = logging.getLogger("eval-harness")

DEFAULT_RUN_LOG_PATH = os.path.join(os.path.dirname(__file__), "eval_runs.jsonl")


def run_pipeline_on_example(pipeline_fn: Callable, example: GoldenExample) -> dict:
    result = pipeline_fn(example.payload)
    predicted_fraud = len(result.get("violations", [])) > 0
    return {
        "id": example.id,
        "channel": example.channel,
        "ground_truth": example.is_fraud,
        "predicted": predicted_fraud,
        "correct": predicted_fraud == example.is_fraud,
        "notes": example.notes,
        "pipeline_result": result,
    }


def compute_metrics(results: List[dict]) -> dict:
    tp = sum(1 for r in results if r["ground_truth"] and r["predicted"])
    fp = sum(1 for r in results if not r["ground_truth"] and r["predicted"])
    fn = sum(1 for r in results if r["ground_truth"] and not r["predicted"])
    tn = sum(1 for r in results if not r["ground_truth"] and not r["predicted"])

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(results) if results else 0.0

    return {
        "n": len(results), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 3), "recall": round(recall, 3),
        "f1": round(f1, 3), "accuracy": round(accuracy, 3),
    }


def run_eval(pipelines: Dict[str, Callable], examples: Optional[List[GoldenExample]] = None) -> dict:
    examples = examples if examples is not None else ALL_EXAMPLES
    by_channel: Dict[str, List[dict]] = {}

    for example in examples:
        pipeline_fn = pipelines.get(example.channel)
        if pipeline_fn is None:
            logger.warning(f"No pipeline registered for channel '{example.channel}', skipping {example.id}")
            continue
        by_channel.setdefault(example.channel, []).append(run_pipeline_on_example(pipeline_fn, example))

    report = {"run_at": datetime.now(timezone.utc).isoformat(), "channels": {}}
    all_results: List[dict] = []
    for channel, results in by_channel.items():
        report["channels"][channel] = {"metrics": compute_metrics(results), "examples": results}
        all_results.extend(results)
    report["overall"] = compute_metrics(all_results)
    return report


def run_judge_eval(pipelines: Dict[str, Callable], eval_agent_module, examples: Optional[List[GoldenExample]] = None) -> dict:
    """
    For each example: is the judge's is_confident verdict ALIGNED with
    whether the pipeline actually got it right? Ideal judge: confident
    when the pipeline is correct, not confident when it's wrong (so the
    retry loop kicks in). A judge that's confident regardless of
    correctness provides no real signal, even if its retry-rate metrics
    look reasonable in production.
    """
    examples = examples if examples is not None else ALL_EXAMPLES
    rows = []

    for example in examples:
        pipeline_fn = pipelines.get(example.channel)
        if pipeline_fn is None:
            continue
        result = pipeline_fn(example.payload)
        predicted_fraud = len(result.get("violations", [])) > 0
        pipeline_correct = predicted_fraud == example.is_fraud

        judge_eval = eval_agent_module.evaluate_event(result, retrieved_rules=result.get("rag_sources"))
        rows.append({
            "id": example.id,
            "pipeline_correct": pipeline_correct,
            "judge_confident": judge_eval.is_confident,
            "judge_confidence_score": judge_eval.confidence_score,
            "judge_aligned": judge_eval.is_confident == pipeline_correct,
        })

    alignment_rate = sum(1 for r in rows if r["judge_aligned"]) / len(rows) if rows else 0.0
    return {"judge_alignment_rate": round(alignment_rate, 3), "n": len(rows), "rows": rows}


def append_to_run_log(report: dict, log_path: str = DEFAULT_RUN_LOG_PATH):
    """
    Flat-file experiment tracking: one JSON line per run, git-diffable,
    no infrastructure to stand up. This is intentionally NOT full MLflow
    -- for a golden set this size, a JSONL log you can `tail` or load
    into a notebook is the right amount of tooling. The interface
    (report in, nothing else needed) means swapping in real MLflow later
    is a change to this one function, not to run_eval() or the pipelines.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    summary = {
        "run_at": report["run_at"],
        "overall": report["overall"],
        "per_channel": {ch: data["metrics"] for ch, data in report["channels"].items()},
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    from backend.src.orchestrator.orchestrator import PIPELINES
    from backend.src.orchestrator import eval_agent as default_eval_agent

    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-judge-eval", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    report = run_eval(PIPELINES)
    append_to_run_log(report)
    print(json.dumps(report["overall"], indent=2))
    for channel, data in report["channels"].items():
        print(f"\n{channel}: {json.dumps(data['metrics'], indent=2)}")
        for ex in data["examples"]:
            if not ex["correct"]:
                print(f"  MISS: {ex['id']} - {ex['notes']}")

    if not args.skip_judge_eval:
        judge_report = run_judge_eval(PIPELINES, default_eval_agent)
        print(f"\njudge_alignment_rate: {judge_report['judge_alignment_rate']} (n={judge_report['n']})")
