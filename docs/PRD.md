# Product requirements

This document specifies what a ward scoring service built on `vital-stream-triage` must do. The repository itself is the synthetic benchmark, not that service and not a clinical device. Measured values are the seed-0 run in [BENCHMARK_RESULTS.md](../BENCHMARK_RESULTS.md).

## 1. Problem

Ward charts are hourly. Vitals are usually present. Labs arrive once or twice a day, so a model that expects a dense vector spends most hours on a stale or empty chemistry panel. A score that crosses 0.7 on a flat class distribution pages the floor for hours that are not the labeled deterioration window. A 1,000-bed census that arrives together has to finish on the CPU budget of one inference worker.

## 2. Users

- The bedside service that decides whether an hour is a page, a watchlist row, or silence.
- The platform service that owns imputation, the scoring head, and the queue.

## 3. In scope

| Item | Requirement |
| --- | --- |
| Channels | 6 vitals, 11 labs, age, GCS, on a 1-hour grid |
| Imputation | Time-decay toward the population mean, compared with zero-fill and forward-fill |
| Score | 3-class softmax: stable, transient, deteriorate |
| Page | Deteriorate probability $> 0.7$ and $H(p) < 0.3$ nats |
| Watchlist | Deteriorate probability $> 0.7$ and $H(p) \ge 0.3$ nats |
| Silence | Deteriorate probability $\le 0.7$ |
| Serving | 1,000 windows of shape $(28, 6)$ plus a static vector of length 2, one CPU thread, P99 $< 50$ ms |

The deterioration label is the six hours before a synthetic event. It is not a cardiac-arrest code. Hours after the event are ignored in the alarm counts.

## 4. Out of scope

- A filed clinical device, a sensitivity claim, or a hospital policy review.
- A 1 Hz waveform, a waveform archive, or a chart-review UI.
- Learned decay weights, MC dropout, and deep ensembles.
- Executing a FHIR server or a Triton inference server. Both appear only as the integration contract below.

## 5. Success criteria

Targets below are the benchmark's. The seed-0 column is what this repository measured.

| Criterion | Target | Seed 0 |
| --- | --- | --- |
| Standardized MSE order | time-decay $<$ forward-fill $<$ zero-fill | 0.723 $<$ 0.870 $<$ 412.978 |
| Decay reflection | time-decay correlation above forward-fill | met (forward-fill is 0 by construction) |
| False-alarm reduction | $\ge 35\%$ | 57.6% (92 to 39) |
| Precision ratio | fusion / baseline $\ge 1$, recall not zero | 1.363; recall 0.278 to 0.200 |
| Serving | micro-batch P99 $< 50$ ms and higher throughput than batch 1 | 6.10 ms; 163,580 vs 5,254 beds/s |

Latency is the quantity that moves on a repeat run. Imputation error and alarm counts are stable at seed 0.

## 6. Page and watchlist

The fusion rule in the benchmark deletes the hour from the alarm count. The product does not. An hour with score $> 0.7$ and $H \ge 0.3$ stays on a watchlist. An hour with score $> 0.7$ and $H < 0.3$ is a page. Everything else is silent.

On the recorded run the fusion rule dropped 67 hours: 14 true positives and 53 false positives. The watchlist is where those 67 hours go. That queue is specified here and is not in the code. The board recomputes the two cuts; it does not render two destinations.

## 7. Integration contract

Not executed.

| Boundary | Contract |
| --- | --- |
| Chart read | FHIR `Patient`, `Encounter`, and `Observation`, hourly buckets |
| Score write | One deteriorate probability, one entropy in nats, one of page / watchlist / silence |
| Serving | A dynamic batcher with a maximum batch and a queue delay budget. The reference behavior is Triton dynamic batching. This process does not start that server. Documentation: [dynamic batcher](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html) |

The local stand-in is `MicroBatchQueue`: one worker, latency measured from a shared arrival time, `max_batch` 128.

## 8. Documents

| File | Role |
| --- | --- |
| [BENCHMARK_RESULTS.md](../BENCHMARK_RESULTS.md) | The measured run |
| [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) | What the code implements |
| [PAPER.md](PAPER.md) | Technical note for the same run |
