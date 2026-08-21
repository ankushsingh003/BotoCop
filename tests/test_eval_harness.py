"""
Tests the eval harness's scoring, aggregation, and run-log logic using
injected fake pipelines/judges -- deliberately independent of GROQ_API_KEY
and network access, since this is pure logic that shouldn't need a real
LLM call to verify it's correct. Running the harness against the REAL
pipelines (backend/eval/run_eval.py __main__) requires a real API key
and isn't exercised here.
"""
import json

from backend.eval.golden_dataset import GoldenExample, ALL_EXAMPLES
from backend.eval.run_eval import compute_metrics, run_eval, run_judge_eval, append_to_run_log


def test_golden_dataset_has_both_classes_per_channel():
    by_channel = {}
    for ex in ALL_EXAMPLES:
        by_channel.setdefault(ex.channel, {True: 0, False: 0})[ex.is_fraud] += 1

    for channel, counts in by_channel.items():
        assert counts[True] > 0, f"{channel} golden set has no fraud examples"
        assert counts[False] > 0, f"{channel} golden set has no non-fraud examples"


def test_compute_metrics_perfect_predictions():
    results = [
        {"ground_truth": True, "predicted": True},
        {"ground_truth": False, "predicted": False},
    ]
    m = compute_metrics(results)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["accuracy"] == 1.0


def test_compute_metrics_with_false_positive_and_false_negative():
    results = [
        {"ground_truth": True, "predicted": True},    # TP
        {"ground_truth": False, "predicted": True},   # FP
        {"ground_truth": True, "predicted": False},   # FN
        {"ground_truth": False, "predicted": False},  # TN
    ]
    m = compute_metrics(results)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5


def test_run_eval_with_injected_pipeline_catches_false_positive():
    examples = [
        GoldenExample(id="a", channel="transaction", payload={}, is_fraud=True),
        GoldenExample(id="b", channel="transaction", payload={}, is_fraud=False),
    ]

    def always_flag(payload, retry_feedback=None):
        return {"violations": [{"severity": "high"}], "final_status": "warning"}

    report = run_eval({"transaction": always_flag}, examples=examples)

    assert report["overall"]["tp"] == 1
    assert report["overall"]["fp"] == 1  # flagged the non-fraud example too
    assert report["channels"]["transaction"]["metrics"]["recall"] == 1.0
    assert report["channels"]["transaction"]["metrics"]["precision"] == 0.5


def test_run_eval_skips_channels_with_no_registered_pipeline():
    examples = [GoldenExample(id="a", channel="video", payload={}, is_fraud=True)]
    report = run_eval({"transaction": lambda p, retry_feedback=None: {"violations": []}}, examples=examples)
    assert report["channels"] == {}  # no pipeline for "video" -> skipped, not an error


def test_append_to_run_log_writes_jsonl(tmp_path):
    log_path = str(tmp_path / "eval_runs.jsonl")
    report = {
        "run_at": "2026-01-01T00:00:00+00:00",
        "channels": {"transaction": {"metrics": {"f1": 0.9}, "examples": []}},
        "overall": {"f1": 0.9},
    }
    append_to_run_log(report, log_path=log_path)
    append_to_run_log(report, log_path=log_path)

    with open(log_path) as f:
        lines = f.readlines()
    assert len(lines) == 2  # each run appends, doesn't overwrite -- history is preserved
    parsed = json.loads(lines[0])
    assert parsed["overall"]["f1"] == 0.9


def test_judge_alignment_detects_overconfident_wrong_judge():
    """
    The judge that matters least is one that's always confident
    regardless of correctness -- this should show up as a LOW alignment
    rate whenever the pipeline is actually wrong, not get masked by the
    judge always saying "confident".
    """
    from backend.src.orchestrator.eval_agent import EventEvalModel

    examples = [
        GoldenExample(id="a", channel="transaction", payload={}, is_fraud=True),   # pipeline will miss this
        GoldenExample(id="b", channel="transaction", payload={}, is_fraud=False),  # pipeline will get this right
    ]

    def never_flag(payload, retry_feedback=None):
        return {"violations": [], "final_status": "success"}  # always predicts "not fraud"

    class OverconfidentJudge:
        @staticmethod
        def evaluate_event(result, retrieved_rules=None):
            return EventEvalModel(is_confident=True, confidence_score=0.95, feedback="")  # always confident

    report = run_judge_eval({"transaction": never_flag}, OverconfidentJudge, examples=examples)

    assert report["judge_alignment_rate"] == 0.5
