"""Streaming imputation for an hourly EMR grid.

Zero-fill writes 0 into every gap. Forward-fill carries the last charted
value and writes 0 before the first charted value. Time-decay treats a stale
result as shrinking toward the population mean:

    x_hat(t) = mu + (x_last - mu) * exp(-dt / tau)

Tau is the analyte freshness constant declared on each feature. Before any
observation, time-decay uses mu and the other two methods use 0. Experiment 1
scores only gaps that already have a previous observation, so that leading-gap
policy is not the comparison.
"""

from __future__ import annotations

import math

import numpy as np

from .generator import FEATURES, LAB_INDEX, Cohort

METHODS = ("zero", "forward", "time_decay")


def decay_value(last: float, delta_hours: float, mu: float, tau_hours: float, method: str) -> float:
    if method == "zero":
        return 0.0
    if method == "forward":
        return float(last)
    if method == "time_decay":
        return float(mu + (last - mu) * math.exp(-delta_hours / tau_hours))
    raise ValueError(f"unknown imputation method {method!r}")


def impute(observed: np.ndarray, mask: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray]:
    """Impute a cohort tensor.

    Returns imputed values and hours since the last real observation.
    Unseen steps use +inf hours. Observed steps use 0 hours and keep the
    charted number.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if observed.shape != mask.shape or observed.ndim != 3:
        raise ValueError("observed and mask must share shape (patients, hours, features)")
    n_patients, n_hours, n_features = observed.shape
    if n_features != len(FEATURES):
        raise ValueError(f"expected {len(FEATURES)} features, got {n_features}")

    times = np.arange(n_hours, dtype=np.int64)
    observed_idx = np.where(mask, times[None, :, None], np.int64(-1))
    last_idx = np.maximum.accumulate(observed_idx, axis=1)
    seen = last_idx >= 0
    safe_idx = np.clip(last_idx, 0, n_hours - 1)
    filled = np.where(mask, np.nan_to_num(observed, nan=0.0), 0.0)
    last_val = np.take_along_axis(filled, safe_idx, axis=1)
    delta = np.where(seen, times[None, :, None] - last_idx, np.int64(0)).astype(np.float64)

    prior = np.array([feature.mu for feature in FEATURES], dtype=np.float64)
    tau = np.array([feature.tau_hours for feature in FEATURES], dtype=np.float64)
    if method == "zero":
        hat = np.zeros_like(filled)
    elif method == "forward":
        hat = np.where(seen, last_val, 0.0)
    else:
        decayed = prior + (last_val - prior) * np.exp(-delta / tau)
        hat = np.where(seen, decayed, prior)
    hat = np.where(mask, filled, hat)

    hours_since = np.where(seen, delta, np.inf)
    hours_since = np.where(mask, 0.0, hours_since)
    if not np.isfinite(hat).all():
        raise RuntimeError("imputation produced a non-finite value")
    return hat, hours_since


class TimeDecayBuffer:
    """One-bed streaming state. Matches `impute` on an hourly grid."""

    def __init__(self, method: str = "time_decay"):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self.method = method
        self.last_value = [None] * len(FEATURES)
        self.last_time = [None] * len(FEATURES)

    def update(self, hour: float, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if values.shape != (len(FEATURES),) or mask.shape != (len(FEATURES),):
            raise ValueError("values and mask must have one entry per feature")
        imputed = np.empty(len(FEATURES), dtype=np.float64)
        for index, feature in enumerate(FEATURES):
            if bool(mask[index]):
                charted = float(values[index])
                self.last_value[index] = charted
                self.last_time[index] = float(hour)
                imputed[index] = charted
                continue
            last = self.last_value[index]
            if last is None:
                imputed[index] = feature.mu if self.method == "time_decay" else 0.0
                continue
            delta = float(hour) - float(self.last_time[index])
            imputed[index] = decay_value(last, delta, feature.mu, feature.tau_hours, self.method)
        return imputed


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 3:
        return 0.0
    left_center = left - left.mean()
    right_center = right - right.mean()
    denom = float(np.sqrt(np.square(left_center).sum() * np.square(right_center).sum()))
    if denom < 1e-12:
        return 0.0
    return float(left_center.dot(right_center) / denom)


def score_imputation(cohort: Cohort, method: str) -> dict:
    """Standardized MSE and decay reflection on lab gaps after the first draw."""
    hat, _hours = impute(cohort.observed, cohort.mask, method)
    n_hours = cohort.n_hours
    times = np.arange(n_hours, dtype=np.int64)
    observed_idx = np.where(cohort.mask, times[None, :, None], np.int64(-1))
    last_idx = np.maximum.accumulate(observed_idx, axis=1)
    seen = last_idx >= 0
    safe_idx = np.clip(last_idx, 0, n_hours - 1)
    filled = np.where(cohort.mask, np.nan_to_num(cohort.observed, nan=0.0), 0.0)
    last_val = np.take_along_axis(filled, safe_idx, axis=1)
    delta = np.where(seen, times[None, :, None] - last_idx, np.int64(0)).astype(np.float64)

    per_feature = []
    for index in LAB_INDEX:
        feature = FEATURES[int(index)]
        evaluated = (~cohort.mask[:, :, index]) & seen[:, :, index]
        if int(evaluated.sum()) < 10:
            raise RuntimeError(f"{feature.key} has too few scored gaps")
        residual = (hat[:, :, index] - cohort.values[:, :, index]) / feature.sigma
        standardized_mse = float(np.square(residual[evaluated]).mean())
        raw_mse = float(np.square((hat[:, :, index] - cohort.values[:, :, index])[evaluated]).mean())
        reflection = _pearson(delta[:, :, index][evaluated], np.abs(hat[:, :, index] - last_val[:, :, index])[evaluated])
        gap = last_val[:, :, index] - feature.mu
        pull_mask = evaluated & (np.abs(gap) > 0.25 * feature.sigma)
        if int(pull_mask.sum()) == 0:
            mean_pull = 0.0
        else:
            mean_pull = float(((last_val[:, :, index] - hat[:, :, index]) / gap)[pull_mask].mean())
        per_feature.append(
            {
                "name": feature.key,
                "label": feature.label,
                "mse": raw_mse,
                "standardized_mse": standardized_mse,
                "decay_reflection": reflection,
                "mean_prior_pull": mean_pull,
                "evaluated_cells": int(evaluated.sum()),
            }
        )

    missing_rate = float(1.0 - cohort.mask[:, :, LAB_INDEX].mean())
    return {
        "method": method,
        "missing_rate": missing_rate,
        "standardized_mse": float(np.mean([row["standardized_mse"] for row in per_feature])),
        "decay_reflection": float(np.mean([row["decay_reflection"] for row in per_feature])),
        "mean_prior_pull": float(np.mean([row["mean_prior_pull"] for row in per_feature])),
        "per_feature": per_feature,
    }


def self_check() -> None:
    """Lock the decay formula and the streaming/batch agreement."""
    lactate = next(index for index, feature in enumerate(FEATURES) if feature.key == "lactate")
    feature = FEATURES[lactate]
    observed = np.full((1, 5, len(FEATURES)), np.nan, dtype=np.float64)
    mask = np.zeros((1, 5, len(FEATURES)), dtype=bool)
    observed[0, 1, lactate] = 10.0
    mask[0, 1, lactate] = True
    hat, hours = impute(observed, mask, "time_decay")
    expected = feature.mu + (10.0 - feature.mu) * math.exp(-1.0 / feature.tau_hours)
    if abs(hat[0, 0, lactate] - feature.mu) > 1e-8:
        raise AssertionError("time-decay should use the population mean before the first draw")
    if abs(hat[0, 1, lactate] - 10.0) > 1e-8:
        raise AssertionError("charted values must pass through")
    if abs(hat[0, 2, lactate] - expected) > 1e-8:
        raise AssertionError("one-hour decay does not match the closed form")
    if hours[0, 0, lactate] != np.inf or abs(hours[0, 2, lactate] - 1.0) > 1e-8:
        raise AssertionError("hours-since-last was not tracked")

    forward, _ = impute(observed, mask, "forward")
    zero, _ = impute(observed, mask, "zero")
    if forward[0, 0, lactate] != 0.0 or forward[0, 4, lactate] != 10.0:
        raise AssertionError("forward-fill did not carry the last value")
    if zero[0, 4, lactate] != 0.0 or zero[0, 1, lactate] != 10.0:
        raise AssertionError("zero-fill did not clear the gap")

    buffer = TimeDecayBuffer("time_decay")
    streamed = np.zeros_like(hat)
    for hour in range(5):
        streamed[0, hour] = buffer.update(hour, np.nan_to_num(observed[0, hour], nan=0.0), mask[0, hour])
    if not np.allclose(streamed, hat):
        raise AssertionError("streaming buffer diverged from the batch imputer")
