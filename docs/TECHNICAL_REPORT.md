# Technical report

Engineering record for one seed-0 run of `vital-stream-triage`. The cohort is synthetic. The numbers below are that run, not a clinical-device measurement.

## 1. Scope

400 patients, 48 hourly steps, seed 0. Six vitals are present on 96% of hours. The other 4% are independent gaps, forward-filled inside the model input. The generator does not place nursing observations on a four-hour or eight-hour round. Eleven labs share two scheduled panels, so lab missingness is 95.8%. Half of the patients who deteriorate also receive one lab draw inside the warning window. That is a generator schedule, not a model of a physician's decision to order a test.

Age and GCS are static. The head sees 17 dynamic channels plus 11 lab staleness channels (hours since the last draw, capped at 48), concatenated in `assemble_channels`. One scored hour is a tensor of shape $(28, 6)$, the current hour and the five before it, plus a static vector of length 2. The labels on those hours are stable, transient, and deteriorate. A transient is a three-hour spike or a deterioration-like mimic, and it is not the positive alarm class.

## 2. Time-decay, and what GRU-D trains instead

The implemented imputor is a fixed exponential return to the population mean:

$$
\hat{x}(t) = \mu + (x_{\mathrm{last}} - \mu)\, \exp(-\Delta t / \tau).
$$

$\tau$ is the analyte's generator time constant. It is not learned. There is no hidden-state decay. Before the first observation, zero-fill and forward-fill both write 0. Scoring starts after that first observation, so the empty prefix does not dominate standardized MSE. Standardized MSE divides squared error by the analyte's population variance, which keeps sodium from defining the macro number alone. The per-analyte table is still reported.

GRU-D (Che et al., *Scientific Reports* 2018) learns an input decay

$$
\gamma_t = \exp\bigl(-\max(0, W_\gamma \delta_t + b_\gamma)\bigr)
$$

and mixes the last observation with a mean:

$$
\hat{x}_t = m_t x_t + (1 - m_t)\bigl(\gamma_t x_{\mathrm{last}} + (1 - \gamma_t)\tilde{x}\bigr).
$$

This repository does not train $W_\gamma$ or $b_\gamma$. A public PyTorch port of GRU-D exists; it is not the authors' repository, and this bench does not vendor it.

On the recorded run, macro standardized MSE was 412.978 (zero), 0.870 (forward), and 0.723 (time-decay). All 11 labs favored time-decay. Lactate remained the largest residual: 19.858 / 2.807 / 2.486. An unobserved shock is in neither $x_{\mathrm{last}}$ nor $\mu$. Forward-fill decay reflection is 0 because the distance from the last draw is identically zero.

`TimeDecayBuffer` matches the vectorized imputer. `self_check` checks the lactate closed form.

## 3. Entropy gate

The head is two causal `Conv1d` layers (32 channels, kernel 3), dropout 0.1 in training only, then a linear map from the window and the static pair. The window is 6 hours, so the convolution length is 2 and the classifier input is 66. Training uses 280 patients, AdamW at $2\times 10^{-3}$, batch 512, 20 epochs, class-weighted cross-entropy, and `ignore_index` $-100$ for hours before the window and hours after the event. The warning label is $[t_{\mathrm{event}}-6, t_{\mathrm{event}})$. The ramp reaches $1/6$ of the shift on the first labeled hour.

Alarm counts use held-out patient-hours only (120 patients, 86 events in the full cohort of 400). The baseline is

$$
s > 0.7.
$$

Fusion is

$$
s > 0.7 \quad \text{and} \quad H(p) < 0.3,
$$

where

$$
H(p) = -\sum_{k=1}^{3} p_k \ln p_k
$$

is the entropy of that one softmax, in nats. $H < 0.3$ requires a maximum probability of about 0.93 or higher. A deteriorate probability of 0.75 is not enough: for $p = (0.10, 0.15, 0.75)$,

$$
H(p) \approx 0.73,
$$

which stays above the cut. It is not MC dropout and not an ensemble.

| Rule | TP | FP | FN | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| $s > 0.7$ | 50 | 92 | 130 | 0.352 | 0.278 |
| $s > 0.7$ and $H < 0.3$ | 36 | 39 | 144 | 0.480 | 0.200 |

False-alarm reduction is 57.6%. The precision ratio is 1.363. The lowest fusion score is 0.928, so the entropy cut is close to a higher score cut on this head. The 67 dropped hours are 14 true positives and 53 false positives. Dropped-set precision is $14/67 = 0.209$. Recall falls, and the report keeps that fall.

Most baseline false positives were stable-hour tails (baseline alarm rate 1.9% on 4,391 stable hours), not the transient class (3.2% to 0.6% on 309 hours). On the 180 deterioration hours the mean score is 0.325 and the 90th percentile is 0.998, so only the late part of the window crosses 0.7.

Sendelbach and Funk, *AACN Advanced Critical Care* 2013;24(4):378–386, report that 72% to 99% of clinical alarms are false. Paine et al. report that 74–99% of physiologic-monitor alarms were not actionable. Neither statistic is this bench's 92 or 39.

Ovadia et al., NeurIPS 2019, show that softmax confidence is often overconfident under dataset shift. This bench does not run a shift experiment. The entropy gate is an operating point on one synthetic distribution.

## 4. Serving

`MicroBatchQueue` uses one worker. Latency is completion minus a common arrival. `torch.set_num_threads(1)` applies only during the serving measurement. Each forward takes `window` of shape `(batch, 28, 6)` and `static` of shape `(batch, 2)`. Batch 128 is 128 of those windows, not a tensor of shape `(128, 19, 48)`. The device is CPU.

| Mode | Max batch | P99 | Throughput |
| --- | ---: | ---: | ---: |
| Sequential | 1 | 188.68 ms | 5,253.9 beds/s |
| Micro-batch | 128 | 6.10 ms | 163,580.4 beds/s |

The ratio is 31.1×. An earlier measurement of the same seed reported a micro-batch P99 of 4.9 ms; the file checked in with this report is 6.10 ms. Quote the file, not a single latency as a fixed proof.

Triton dynamic batching was not executed. The behavior this queue stands in for is documented at <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html> and in the server tree at `docs/user_guide/batcher.md`.

## 5. Operating point the product still owes

Page when $s > 0.7$ and $H < 0.3$. Watchlist when $s > 0.7$ and $H \ge 0.3$. Silence otherwise. The benchmark counts only the page rule against the score rule. The watchlist is a requirement in [PRD.md](PRD.md) and has no implementation.

FHIR `Observation`, `Patient`, and `Encounter` are the specified read contract. No FHIR client runs here.

## 6. References

- Che Z, Purushotham S, Cho K, Sontag D, Liu Y. Recurrent neural networks for multivariate time series with missing values. *Sci Rep.* 2018;8:6085. <https://doi.org/10.1038/s41598-018-24271-9>
- Reyna MA, et al. Early prediction of sepsis from clinical data: the PhysioNet/Computing in Cardiology Challenge 2019. *Crit Care Med.* 2020;48(2):210–217. Challenge set: <https://physionet.org/content/challenge-2019/1.0.0/>. Forty hourly variables, labels shifted by 6 hours, a utility that penalizes late and false alarms. This bench does not use that data or that utility.
- Sendelbach S, Funk M. Alarm fatigue: a patient safety concern. *AACN Adv Crit Care.* 2013;24(4):378–386. PMID 24153215.
- Paine CW, et al. Systematic review of physiologic monitor alarm characteristics and pragmatic interventions to reduce alarm frequency. *J Hosp Med.* 2016. PMC4778561.
- Ovadia Y, et al. Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift. NeurIPS 2019. arXiv:1906.02530.
- Maharaj R, Raffaele I, Wendon J. Rapid response systems: a systematic review and meta-analysis. *Crit Care.* 2015;19:254. Literature only. This bench does not simulate a rapid-response team.
- Triton dynamic batching user guide, as linked in section 4. Not executed.
