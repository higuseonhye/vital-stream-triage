"""Causal 1D head and a single-worker micro-batch queue.

The head reads a 6-hour window of time-decay-imputed vitals and labs, plus
how many hours each lab has gone without a new draw. Static age and GCS are
concatenated after the temporal stack. Inference uses one softmax. Its
entropy is the entropy of that 3-class distribution, in nats.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .generator import FEATURES, IGNORE_LABEL, LAB_INDEX, STATIC_KEYS, Cohort

WINDOW = 6
CHANNEL_COUNT = len(FEATURES) + len(LAB_INDEX)


class DeteriorationHead(nn.Module):
    def __init__(self, channels: int = CHANNEL_COUNT, static_dim: int = len(STATIC_KEYS)):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, 32, kernel_size=3)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3)
        self.drop = nn.Dropout(0.10)
        self.fc = nn.Linear(32 * (WINDOW - 4) + static_dim, 3)

    def forward(self, window: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.conv1(window))
        hidden = self.drop(torch.relu(self.conv2(hidden)))
        flat = hidden.reshape(hidden.shape[0], -1)
        return self.fc(torch.cat([flat, static], dim=1))


@dataclass
class Normalizer:
    channel_mean: np.ndarray
    channel_std: np.ndarray
    static_mean: np.ndarray
    static_std: np.ndarray

    def channels(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.channel_mean) / self.channel_std).astype(np.float32)

    def static(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.static_mean) / self.static_std).astype(np.float32)


@dataclass
class FitResult:
    model: DeteriorationHead
    normalizer: Normalizer
    probabilities: np.ndarray
    hour_index: np.ndarray
    labels: np.ndarray
    train_loss: float
    train_ids: np.ndarray
    test_ids: np.ndarray


def assemble_channels(imputed: np.ndarray, hours_since: np.ndarray, n_hours: int) -> np.ndarray:
    staleness = np.minimum(hours_since[:, :, LAB_INDEX], float(n_hours))
    return np.concatenate([imputed, staleness], axis=-1).astype(np.float64)


def _windows(channels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_patients, n_hours, n_channels = channels.shape
    frames = []
    for hour in range(WINDOW - 1, n_hours):
        frames.append(channels[:, hour - WINDOW + 1 : hour + 1, :])
    stacked = np.stack(frames, axis=1)
    windows = np.transpose(stacked, (0, 1, 3, 2)).astype(np.float32)
    hour_index = np.arange(WINDOW - 1, n_hours, dtype=np.int64)
    if windows.shape[2] != n_channels:
        raise RuntimeError("window channel axis was transposed incorrectly")
    return windows, hour_index


def _split(n_patients: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.random.default_rng(seed + 17).permutation(n_patients)
    n_train = max(WINDOW, int(round(n_patients * 0.70)))
    if n_patients - n_train < 8:
        raise ValueError("test split would be smaller than 8 patients")
    return order[:n_train], order[n_train:]


def fit_head(
    cohort: Cohort,
    imputed: np.ndarray,
    hours_since: np.ndarray,
    epochs: int = 20,
    batch_size: int = 512,
    seed: int = 0,
) -> FitResult:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    channels = assemble_channels(imputed, hours_since, cohort.n_hours)
    train_ids, test_ids = _split(cohort.n_patients, seed)
    train_channels = channels[train_ids].reshape(-1, channels.shape[-1])
    channel_mean = train_channels.mean(axis=0)
    channel_std = np.maximum(train_channels.std(axis=0), 1e-3)
    static = np.stack([cohort.age, cohort.gcs], axis=1)
    static_mean = static[train_ids].mean(axis=0)
    static_std = np.maximum(static[train_ids].std(axis=0), 1e-3)
    normalizer = Normalizer(channel_mean, channel_std, static_mean, static_std)

    windows, hour_index = _windows(normalizer.channels(channels))
    labeled = cohort.labels.copy()
    labeled[:, : WINDOW - 1] = IGNORE_LABEL
    labels = labeled[:, hour_index]
    static_norm = normalizer.static(static)

    y_train = labels[train_ids].reshape(-1)
    counted = y_train[y_train >= 0]
    counts = np.bincount(counted, minlength=3).astype(np.float64)
    if np.any(counts == 0):
        raise RuntimeError("training split is missing a class; increase the cohort")
    weights = counts.sum() / (counts.size * counts)

    torch.manual_seed(seed)
    model = DeteriorationHead()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32), ignore_index=IGNORE_LABEL)
    x_train = np.ascontiguousarray(windows[train_ids].reshape(-1, CHANNEL_COUNT, WINDOW))
    s_train = np.repeat(static_norm[train_ids], labels.shape[1], axis=0)
    order = np.arange(x_train.shape[0])
    shuffle = np.random.default_rng(seed + 3)
    model.train()
    train_loss = 0.0
    for _epoch in range(epochs):
        shuffle.shuffle(order)
        total = 0.0
        seen = 0
        for start in range(0, order.size, batch_size):
            batch = order[start : start + batch_size]
            target = torch.from_numpy(y_train[batch].astype(np.int64))
            if int((target >= 0).sum()) == 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            logits = model(torch.from_numpy(x_train[batch]), torch.from_numpy(s_train[batch]))
            loss = loss_fn(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError("training loss became non-finite")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * int(batch.size)
            seen += int(batch.size)
        train_loss = total / max(seen, 1)

    probabilities = _predict(model, windows, static_norm)
    return FitResult(
        model=model,
        normalizer=normalizer,
        probabilities=probabilities,
        hour_index=hour_index,
        labels=labels,
        train_loss=float(train_loss),
        train_ids=train_ids,
        test_ids=test_ids,
    )


def _predict(model: DeteriorationHead, windows: np.ndarray, static_norm: np.ndarray) -> np.ndarray:
    n_patients, n_times, channels, length = windows.shape
    flat_x = np.ascontiguousarray(windows.reshape(-1, channels, length))
    flat_s = np.repeat(static_norm, n_times, axis=0)
    out = np.empty((flat_x.shape[0], 3), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, flat_x.shape[0], 4096):
            stop = min(flat_x.shape[0], start + 4096)
            logits = model(torch.from_numpy(flat_x[start:stop]), torch.from_numpy(flat_s[start:stop]))
            out[start:stop] = torch.softmax(logits, dim=-1).numpy()
    return out.reshape(n_patients, n_times, 3)


class MicroBatchQueue:
    """One worker. max_batch=1 is the sequential baseline; larger batches share a forward."""

    def __init__(self, max_batch: int):
        if max_batch < 1:
            raise ValueError("max_batch must be positive")
        self.max_batch = int(max_batch)

    def serve(
        self,
        model: DeteriorationHead,
        windows: torch.Tensor,
        static: torch.Tensor,
    ) -> tuple[np.ndarray, float]:
        count = int(windows.shape[0])
        if count < 8:
            raise ValueError("need at least 8 beds so a P99 is defined")
        if static.shape[0] != count:
            raise ValueError("static rows must match beds")
        model.eval()
        warmup_n = min(self.max_batch, count)
        with torch.inference_mode():
            model(windows[:1], static[:1])
            model(windows[:warmup_n], static[:warmup_n])
            latencies = np.empty(count, dtype=np.float64)
            arrival = time.perf_counter()
            cursor = 0
            while cursor < count:
                stop = min(count, cursor + self.max_batch)
                model(windows[cursor:stop], static[cursor:stop])
                done = time.perf_counter()
                latencies[cursor:stop] = (done - arrival) * 1000.0
                cursor = stop
            wall_s = time.perf_counter() - arrival
        return latencies, wall_s


def _percentiles(latencies_ms: np.ndarray) -> dict[str, float]:
    return {
        "p50_ms": float(np.percentile(latencies_ms, 50)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
        "p99_ms": float(np.percentile(latencies_ms, 99)),
        "mean_ms": float(latencies_ms.mean()),
        "max_ms": float(latencies_ms.max()),
    }


def benchmark_serving(
    model: DeteriorationHead,
    windows: np.ndarray,
    static: np.ndarray,
    max_batch: int = 128,
    sla_ms: float = 50.0,
) -> dict:
    """Burst-latency of one CPU thread. Every bed arrives at the same instant."""
    if windows.shape[0] != static.shape[0]:
        raise ValueError("window and static batches differ")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        tensor_x = torch.from_numpy(np.ascontiguousarray(windows))
        tensor_s = torch.from_numpy(np.ascontiguousarray(static))
        rows = []
        for mode, batch in (("sequential", 1), ("microbatch", max_batch)):
            latencies, wall_s = MicroBatchQueue(batch).serve(model, tensor_x, tensor_s)
            stats = _percentiles(latencies)
            stats.update(
                {
                    "mode": mode,
                    "max_batch": int(batch),
                    "beds": int(windows.shape[0]),
                    "wall_ms": float(wall_s * 1000.0),
                    "throughput_beds_per_sec": float(windows.shape[0] / wall_s),
                    "sla_ms": float(sla_ms),
                    "sla_met": bool(stats["p99_ms"] < sla_ms),
                }
            )
            rows.append(stats)
    finally:
        torch.set_num_threads(previous_threads)
    return {
        "device": "cpu",
        "threads": 1,
        "arrival": "simultaneous_burst",
        "latency_definition": "completion_minus_common_arrival",
        "sla_rule": "p99_ms < sla_ms",
        "rows": rows,
    }


def sample_serving_batch(
    windows: np.ndarray,
    static: np.ndarray,
    patient_ids: np.ndarray,
    n_beds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw bed windows from the held-out patients."""
    chosen = windows[patient_ids]
    n_patients, n_times, channels, length = chosen.shape
    flat_x = chosen.reshape(-1, channels, length)
    flat_s = np.repeat(static[patient_ids], n_times, axis=0)
    if flat_x.shape[0] < n_beds:
        replace = True
    else:
        replace = False
    pick = np.random.default_rng(seed + 29).choice(flat_x.shape[0], size=n_beds, replace=replace)
    return np.ascontiguousarray(flat_x[pick]), np.ascontiguousarray(flat_s[pick])


def prepared_windows(cohort: Cohort, imputed: np.ndarray, hours_since: np.ndarray, normalizer: Normalizer):
    channels = normalizer.channels(assemble_channels(imputed, hours_since, cohort.n_hours))
    windows, hour_index = _windows(channels)
    static = normalizer.static(np.stack([cohort.age, cohort.gcs], axis=1))
    return windows, static, hour_index
