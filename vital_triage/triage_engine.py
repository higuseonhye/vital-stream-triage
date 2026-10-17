"""Score-and-entropy alarm gate.

Baseline pages when the deteriorate-class softmax exceeds 0.7. The fusion
rule also requires the 3-class entropy, in nats, to be under 0.3. On this
head that second cut is the binding one: H < 0.3 only happens when the
distribution is sharply peaked (the largest probability is about 0.93 or
higher). It is a confidence gate on the same softmax, not a second model.
"""

from __future__ import annotations

import numpy as np

from .generator import CLASS_DETERIORATE, CLASS_NAMES, IGNORE_LABEL

SCORE_THRESHOLD = 0.7
ENTROPY_THRESHOLD = 0.3


def softmax_entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-8, 1.0)
    clipped = clipped / clipped.sum(axis=-1, keepdims=True)
    return -(clipped * np.log(clipped)).sum(axis=-1)


def alarm_masks(
    probabilities: np.ndarray,
    score_threshold: float = SCORE_THRESHOLD,
    entropy_threshold: float = ENTROPY_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    score = probabilities[..., CLASS_DETERIORATE]
    entropy = softmax_entropy(probabilities)
    baseline = score > score_threshold
    fusion = baseline & (entropy < entropy_threshold)
    return score, entropy, baseline, fusion


def _binary_counts(prediction: np.ndarray, truth: np.ndarray) -> dict:
    prediction = prediction.astype(bool)
    truth = truth.astype(bool)
    true_positive = int((prediction & truth).sum())
    false_positive = int((prediction & ~truth).sum())
    false_negative = int((~prediction & truth).sum())
    true_negative = int((~prediction & ~truth).sum())
    precision = None if true_positive + false_positive == 0 else true_positive / (true_positive + false_positive)
    recall = None if true_positive + false_negative == 0 else true_positive / (true_positive + false_negative)
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "tn": true_negative,
        "precision": precision,
        "recall": recall,
        "alarms": true_positive + false_positive,
    }


def _class_rows(score: np.ndarray, entropy: np.ndarray, labels: np.ndarray, baseline: np.ndarray, fusion: np.ndarray) -> list[dict]:
    rows = []
    for class_index, name in enumerate(CLASS_NAMES):
        chosen = labels == class_index
        if int(chosen.sum()) == 0:
            continue
        rows.append(
            {
                "class": name,
                "hours": int(chosen.sum()),
                "mean_score": float(score[chosen].mean()),
                "p50_score": float(np.percentile(score[chosen], 50)),
                "p90_score": float(np.percentile(score[chosen], 90)),
                "mean_entropy": float(entropy[chosen].mean()),
                "baseline_alarm_rate": float(baseline[chosen].mean()),
                "fusion_alarm_rate": float(fusion[chosen].mean()),
            }
        )
    return rows


def evaluate_alarms(
    probabilities: np.ndarray,
    labels: np.ndarray,
    patient_ids: np.ndarray,
    score_threshold: float = SCORE_THRESHOLD,
    entropy_threshold: float = ENTROPY_THRESHOLD,
) -> dict:
    """Score held-out patient-hours. Ignored hours (no window, or after the event) are dropped."""
    if probabilities.shape[:2] != labels.shape:
        raise ValueError("probabilities and labels must share patient and time axes")
    score, entropy, baseline, fusion = alarm_masks(probabilities, score_threshold, entropy_threshold)
    valid = np.zeros(labels.shape, dtype=bool)
    valid[patient_ids] = True
    valid &= labels >= 0
    if int(valid.sum()) == 0:
        raise RuntimeError("alarm evaluation has no labeled hours")
    truth = labels == CLASS_DETERIORATE
    baseline_counts = _binary_counts(baseline[valid], truth[valid])
    fusion_counts = _binary_counts(fusion[valid], truth[valid])
    if baseline_counts["fp"] == 0:
        false_alarm_reduction = None
    else:
        false_alarm_reduction = (baseline_counts["fp"] - fusion_counts["fp"]) / baseline_counts["fp"]
    if baseline_counts["precision"] in (None, 0) or fusion_counts["precision"] is None:
        precision_defense = None
    else:
        precision_defense = fusion_counts["precision"] / baseline_counts["precision"]
    kept = fusion[valid] & baseline[valid]
    min_score_kept = None if int(kept.sum()) == 0 else float(score[valid][kept].min())
    return {
        "score_threshold": float(score_threshold),
        "entropy_threshold_nats": float(entropy_threshold),
        "entropy_definition": "-sum p_k ln p_k",
        "evaluated_hours": int(valid.sum()),
        "positive_hours": int(truth[valid].sum()),
        "baseline": baseline_counts,
        "fusion": fusion_counts,
        "false_alarm_reduction": false_alarm_reduction,
        "precision_defense": precision_defense,
        "min_score_among_fusion_alarms": min_score_kept,
        "by_class": _class_rows(score[valid], entropy[valid], labels[valid], baseline[valid], fusion[valid]),
    }
