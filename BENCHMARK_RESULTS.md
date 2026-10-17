# Vital stream triage — benchmark results

Synthetic hourly ward stream. This file is one CPU run of the repository, not a clinical-device measurement.

- 400 patients, 48-hour grid, seed 0
- Alarms are counted on 120 held-out patients. Imputation error uses every lab gap in the cohort.
- Lab missingness 95.8%. Scoring uses gaps after the first draw only.

## 1. Imputation

Standardized MSE is the mean squared error against the latent trajectory, divided by that analyte's population variance. Decay reflection is the Pearson correlation between hours since the last draw and distance from that draw. Forward-fill keeps the distance at zero, so the correlation is zero.

| Analyte | key | zero | forward | time-decay | decay correlation |
| --- | --- | --- | --- | --- | --- |
| WBC | wbc | 26.660 | 0.885 | 0.682 | 0.500 |
| Hemoglobin | hb | 98.044 | 0.307 | 0.279 | 0.529 |
| Hematocrit | hct | 118.596 | 0.312 | 0.282 | 0.515 |
| Platelet | platelet | 33.302 | 0.450 | 0.375 | 0.528 |
| Bilirubin | bilirubin | 11.498 | 0.461 | 0.399 | 0.520 |
| Creatinine | creatinine | 26.394 | 0.914 | 0.725 | 0.506 |
| BUN | bun | 23.176 | 0.676 | 0.575 | 0.513 |
| CRP | crp | 6.211 | 0.873 | 0.789 | 0.513 |
| Lactate | lactate | 19.858 | 2.807 | 2.486 | 0.305 |
| Sodium | sodium | 3989.082 | 0.702 | 0.555 | 0.491 |
| Potassium | potassium | 189.940 | 1.181 | 0.802 | 0.420 |
| 11-lab mean | macro | 412.978 | 0.870 | 0.723 | 0.485 |

Order target (time-decay < forward < zero): met.
Decay-correlation target (time-decay > forward): met.
Lactate standardized MSE is zero 19.858, forward 2.807, time-decay 2.486.

## 2. Alarm policy

The baseline fires when the deteriorate-class softmax exceeds 0.7. The fusion rule also requires 3-class entropy below 0.3 nats. That entropy is the entropy of the same softmax, so values under 0.3 nats occur only when the largest probability is about 0.93 or higher.

| Class | Hours | Mean score | Score P90 | Mean entropy | Baseline alarms | Fusion alarms |
| --- | --- | --- | --- | --- | --- | --- |
| stable | 4391 | 0.035 | 0.030 | 0.194 | 1.9% | 0.8% |
| transient | 309 | 0.062 | 0.155 | 0.259 | 3.2% | 0.6% |
| deteriorate | 180 | 0.325 | 0.998 | 0.280 | 27.8% | 20.0% |

| Rule | TP | FP | FN | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| softmax > 0.7 | 50 | 92 | 130 | 0.352 | 0.278 |
| softmax > 0.7 and H < 0.3 | 36 | 39 | 144 | 0.480 | 0.200 |

False-alarm reduction 57.6% (target at least 35%): met.
Precision ratio 1.363 (fusion precision / baseline precision, target at least 1): met.
Lowest deteriorate score among fusion alarms 0.928.
Recall moves the other way. The false-negative column is the cost of the tighter rule.

## 3. Simultaneous 1,000-bed arrival

Every bed window arrives at one timestamp on a single CPU-thread queue. One window has shape (28, 6), with a separate static vector of length 2. Latency is completion time minus that shared arrival. Sequential serving is the same queue with max batch 1.

| Mode | max batch | P50 ms | P95 ms | P99 ms | beds/s | P99 < 50 ms |
| --- | --- | --- | --- | --- | --- | --- |
| sequential | 1 | 93.12 | 179.98 | 188.68 | 5253.9 | no |
| microbatch | 128 | 2.19 | 6.10 | 6.10 | 163580.4 | yes |

Micro-batch P99 6.10 ms: met.
Throughput 163580.4 versus sequential 5253.9 beds/s.

## Outside this run

- Not a clinical-device claim, a sensitivity filing, or a fit assessment for a hospital alarm policy.
- The entropy gate is one softmax entropy, not epistemic uncertainty from MC dropout or an ensemble.
- Not a 1 Hz monitor waveform. The grid is hourly, as in a ward chart.

