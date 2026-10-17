"""Synthetic hourly EMR stream for 19 VitalCare-style variables.

Six vitals are charted almost every hour. Eleven labs share two blood-panel
draws per 48 h, so each lab is missing on more than 90% of the grid. Age and
GCS are static. Latent labs are an Ornstein-Uhlenbeck process around a
population mean; that is the world time-decay imputation is built for.

Nothing here is an AITRICS or VUNO export. The four labs that complete the
eleven (hematocrit, BUN, sodium, potassium) are routine chemistries, not a
claim about a vendor feature contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WARNING_HOURS = 6
TRANSIENT_HOURS = 3
POST_EVENT_IGNORE = True
VITAL_MISSING_RATE = 0.04
FRESH_LAB_PROBABILITY = 0.50
TRANSIENT_RATE = 0.85
MIMIC_FRACTION = 0.70


@dataclass(frozen=True)
class Feature:
    key: str
    korean: str
    group: str
    unit: str
    mu: float
    sigma: float
    tau_hours: float
    floor: float
    ceil: float
    event_shift: float
    mimic_shift: float
    spike_shift: float
    meas_scale: float


FEATURES: tuple[Feature, ...] = (
    Feature("sbp", "수축기 혈압", "vital", "mmHg", 122.0, 10.0, 3.0, 70.0, 210.0, -34.0, -22.0, 0.0, 0.10),
    Feature("dbp", "이완기 혈압", "vital", "mmHg", 74.0, 8.0, 3.0, 40.0, 130.0, -16.0, -10.0, 0.0, 0.10),
    Feature("hr", "맥박", "vital", "bpm", 82.0, 8.0, 2.0, 35.0, 180.0, 38.0, 26.0, 30.0, 0.10),
    Feature("rr", "호흡수", "vital", "/min", 16.0, 2.0, 2.0, 8.0, 44.0, 9.0, 6.0, 7.0, 0.10),
    Feature("bt", "체온", "vital", "°C", 36.7, 0.25, 8.0, 34.5, 41.0, 1.25, 0.55, 0.0, 0.10),
    Feature("spo2", "산소포화도", "vital", "%", 97.2, 1.1, 2.0, 78.0, 100.0, -5.5, -2.8, -0.3, 0.08),
    Feature("wbc", "백혈구", "lab", "10^3/µL", 8.0, 1.6, 24.0, 1.0, 40.0, 6.0, 0.0, 0.0, 0.20),
    Feature("hb", "헤모글로빈", "lab", "g/dL", 12.8, 1.3, 72.0, 6.0, 18.0, -1.4, 0.0, 0.0, 0.15),
    Feature("hct", "헤마토크릿", "lab", "%", 38.0, 3.5, 72.0, 18.0, 55.0, -4.0, 0.0, 0.0, 0.15),
    Feature("platelet", "혈소판", "lab", "10^3/µL", 230.0, 40.0, 48.0, 20.0, 700.0, -80.0, 0.0, 0.0, 0.15),
    Feature("bilirubin", "빌리루빈", "lab", "mg/dL", 0.70, 0.22, 48.0, 0.10, 12.0, 0.80, 0.0, 0.0, 0.20),
    Feature("creatinine", "크레아티닌", "lab", "mg/dL", 0.90, 0.18, 18.0, 0.30, 8.0, 0.70, 0.0, 0.0, 0.20),
    Feature("bun", "BUN", "lab", "mg/dL", 16.0, 3.5, 24.0, 4.0, 80.0, 12.0, 0.0, 0.0, 0.20),
    Feature("crp", "CRP", "lab", "mg/dL", 1.10, 0.55, 36.0, 0.05, 30.0, 7.0, 0.0, 0.0, 0.20),
    Feature("lactate", "젖산", "lab", "mmol/L", 1.15, 0.30, 6.0, 0.30, 12.0, 2.40, 0.0, 0.0, 0.25),
    Feature("sodium", "나트륨", "lab", "mEq/L", 139.0, 2.2, 24.0, 125.0, 155.0, -4.0, 0.0, 0.0, 0.15),
    Feature("potassium", "칼륨", "lab", "mEq/L", 4.10, 0.30, 12.0, 2.6, 6.5, 0.55, 0.0, 0.0, 0.20),
)

FEATURE_INDEX = {feature.key: index for index, feature in enumerate(FEATURES)}
VITAL_INDEX = np.array([index for index, feature in enumerate(FEATURES) if feature.group == "vital"], dtype=np.int64)
LAB_INDEX = np.array([index for index, feature in enumerate(FEATURES) if feature.group == "lab"], dtype=np.int64)
STATIC_KEYS = ("age", "gcs")

CLASS_STABLE = 0
CLASS_TRANSIENT = 1
CLASS_DETERIORATE = 2
CLASS_NAMES = ("stable", "transient", "deteriorate")
IGNORE_LABEL = -100


@dataclass
class Cohort:
    values: np.ndarray
    observed: np.ndarray
    mask: np.ndarray
    age: np.ndarray
    gcs: np.ndarray
    labels: np.ndarray
    event_time: np.ndarray
    transient_start: np.ndarray
    transient_kind: np.ndarray
    lab_draws: np.ndarray
    seed: int

    @property
    def n_patients(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_hours(self) -> int:
        return int(self.values.shape[1])


def feature_index(key: str) -> int:
    try:
        return FEATURE_INDEX[key]
    except KeyError as exc:
        known = ", ".join(FEATURE_INDEX)
        raise KeyError(f"unknown feature {key!r}; expected one of {known}") from exc


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def _timeline(n_hours: int) -> tuple[int, int, int, int]:
    if n_hours < 24:
        raise ValueError("n_hours must be at least 24 so a 6 h warning and a transient can both fit")
    event_lo = max(WARNING_HOURS + 2, (2 * n_hours) // 3)
    event_hi = n_hours
    earliest_warning = event_lo - WARNING_HOURS
    transient_lo = max(1, n_hours // 8)
    transient_hi = earliest_warning - TRANSIENT_HOURS
    if transient_hi <= transient_lo or event_hi <= event_lo:
        raise ValueError(f"n_hours={n_hours} cannot separate transients from deterioration windows")
    return transient_lo, transient_hi, event_lo, event_hi


def sample_cohort(n_patients: int = 400, n_hours: int = 48, seed: int = 0) -> Cohort:
    """Draw latent trajectories, noisy charted values, and deterioration labels."""
    if n_patients < 32:
        raise ValueError("n_patients must be at least 32")
    transient_lo, transient_hi, event_lo, event_hi = _timeline(n_hours)
    rng = np.random.default_rng(seed)
    n_features = len(FEATURES)
    times = np.arange(n_hours, dtype=np.int64)

    age = np.clip(rng.normal(64.0, 14.0, size=n_patients), 18.0, 95.0)
    gcs_roll = rng.random(n_patients)
    gcs = np.full(n_patients, 15.0)
    gcs[gcs_roll > 0.75] = 14.0
    gcs[gcs_roll > 0.90] = 13.0
    gcs[gcs_roll > 0.97] = 12.0

    event_logit = -1.35 + 0.02 * (age - 64.0) + 0.22 * (15.0 - gcs)
    has_event = rng.random(n_patients) < _sigmoid(event_logit)
    event_time = np.full(n_patients, -1, dtype=np.int64)
    event_time[has_event] = rng.integers(event_lo, event_hi, size=int(has_event.sum()))

    has_transient = rng.random(n_patients) < TRANSIENT_RATE
    is_mimic = has_transient & (rng.random(n_patients) < MIMIC_FRACTION)
    is_spike = has_transient & ~is_mimic
    transient_start = np.full(n_patients, -1, dtype=np.int64)
    transient_start[has_transient] = rng.integers(transient_lo, transient_hi, size=int(has_transient.sum()))
    transient_kind = np.zeros(n_patients, dtype=np.int64)
    transient_kind[is_spike] = 1
    transient_kind[is_mimic] = 2

    draw = np.zeros((n_patients, 2), dtype=np.int64)
    draw[:, 0] = rng.integers(2, max(3, transient_lo + 2), size=n_patients)
    draw[:, 1] = rng.integers(n_hours // 2, event_lo, size=n_patients)
    fresh = has_event & (rng.random(n_patients) < FRESH_LAB_PROBABILITY)
    if np.any(fresh):
        offset = rng.integers(1, WARNING_HOURS, size=int(fresh.sum()))
        draw[fresh, 1] = event_time[fresh] - offset
    clash = draw[:, 0] == draw[:, 1]
    draw[clash, 1] = np.minimum(draw[clash, 0] + 5, n_hours - 1)

    mu = np.array([feature.mu for feature in FEATURES], dtype=np.float64)
    sigma = np.array([feature.sigma for feature in FEATURES], dtype=np.float64)
    tau = np.array([feature.tau_hours for feature in FEATURES], dtype=np.float64)
    floor = np.array([feature.floor for feature in FEATURES], dtype=np.float64)
    ceil = np.array([feature.ceil for feature in FEATURES], dtype=np.float64)
    event_shift = np.array([feature.event_shift for feature in FEATURES], dtype=np.float64)
    mimic_shift = np.array([feature.mimic_shift for feature in FEATURES], dtype=np.float64)
    spike_shift = np.array([feature.spike_shift for feature in FEATURES], dtype=np.float64)
    meas_sigma = np.array([feature.meas_scale * feature.sigma for feature in FEATURES], dtype=np.float64)

    transient_shift = np.zeros((n_patients, n_features), dtype=np.float64)
    transient_shift[is_spike] = spike_shift
    transient_shift[is_mimic] = mimic_shift
    transient_on = np.zeros((n_patients, n_hours), dtype=bool)
    active = has_transient & (transient_start >= 0)
    if np.any(active):
        start = transient_start[active][:, None]
        grid = times[None, :]
        transient_on[active] = (grid >= start) & (grid < start + TRANSIENT_HOURS)

    ramp = np.zeros((n_patients, n_hours), dtype=np.float64)
    if np.any(has_event):
        # First labeled hour already carries 1/6 of the shift, and the last
        # labeled hour sits on the full shift. A zero-shift first hour would
        # be a positive label with no signal.
        start = (event_time[has_event] - WARNING_HOURS)[:, None]
        steps = times[None, :] - start + 1
        ramp[has_event] = np.clip(steps / float(WARNING_HOURS), 0.0, 1.0)

    target_extra = ramp[:, :, None] * event_shift[None, None, :]
    target_extra = target_extra + transient_on[:, :, None] * transient_shift[:, None, :]

    decay = np.exp(-1.0 / tau)
    noise_scale = sigma * np.sqrt(1.0 - np.exp(-2.0 / tau))
    state = mu + rng.normal(0.0, 1.0, size=(n_patients, n_features)) * sigma
    values = np.empty((n_patients, n_hours, n_features), dtype=np.float64)
    for hour in range(n_hours):
        target = mu + target_extra[:, hour, :]
        shock = rng.normal(0.0, 1.0, size=(n_patients, n_features)) * noise_scale
        state = target + (state - target) * decay + shock
        state = np.clip(state, floor, ceil)
        values[:, hour, :] = state

    mask = np.zeros((n_patients, n_hours, n_features), dtype=bool)
    vital_keep = rng.random((n_patients, n_hours, VITAL_INDEX.size)) >= VITAL_MISSING_RATE
    mask[:, :, VITAL_INDEX] = vital_keep
    for draw_slot in range(2):
        for lab in LAB_INDEX:
            mask[np.arange(n_patients), draw[:, draw_slot], lab] = True

    noisy = values + rng.normal(0.0, 1.0, size=values.shape) * meas_sigma
    noisy = np.clip(noisy, floor, ceil)
    observed = np.where(mask, noisy, np.nan)

    labels = np.zeros((n_patients, n_hours), dtype=np.int64)
    labels[transient_on] = CLASS_TRANSIENT
    warning = np.zeros((n_patients, n_hours), dtype=bool)
    post = np.zeros((n_patients, n_hours), dtype=bool)
    if np.any(has_event):
        event_rows = event_time[has_event][:, None]
        warning[has_event] = (times[None, :] >= event_rows - WARNING_HOURS) & (times[None, :] < event_rows)
        post[has_event] = times[None, :] >= event_rows
    if np.any(warning & transient_on):
        raise RuntimeError("transient window overlapped a deterioration label")
    labels[warning] = CLASS_DETERIORATE
    if POST_EVENT_IGNORE:
        labels[post] = IGNORE_LABEL

    return Cohort(
        values=values,
        observed=observed,
        mask=mask,
        age=age.astype(np.float64),
        gcs=gcs.astype(np.float64),
        labels=labels,
        event_time=event_time,
        transient_start=transient_start,
        transient_kind=transient_kind,
        lab_draws=draw,
        seed=seed,
    )
