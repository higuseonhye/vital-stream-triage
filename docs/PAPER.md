# Technical note: decay imputation, entropy alarms, and micro-batching on a synthetic ward stream

This note restates one repository run in technical form. It is not a journal article and it has not been peer reviewed. It does not claim a hospital result.

## Abstract

Three comparisons were run on a 400-patient synthetic chart with an hourly grid. Eleven labs are observed twice in 48 hours, so lab missingness is 95.8%. A fixed time-decay toward the population mean has lower standardized MSE than forward-fill (0.723 versus 0.870) and far lower than zero-fill (412.978). Lactate follows the same order (2.486 versus 2.807) and keeps the largest residual. Requiring 3-class entropy below 0.3 nats on top of a deteriorate probability above 0.7 cuts held-out false positives from 92 to 39 and raises precision from 0.352 to 0.480. Recall falls from 0.278 to 0.200. On one CPU thread, a simultaneous arrival of 1,000 windows has P99 188.68 ms at batch 1 and 6.10 ms at batch 128, a 31.1× throughput change. Learned decay weights, a dataset shift, FHIR, and Triton are not in the experiment.

## 1. Introduction

Laboratory values in a chart are written at the time of the order, not every hour. Che et al. treat a long gap as evidence that the missing value has moved toward a default, and they train that decay inside GRU-D rather than fixing it. A separate line of work treats false alarms as the operational problem: reviews of physiologic-monitor alarms find that most are false or not actionable. This note measures a smaller version of each mechanism. The input decay time constant is not trained. Entropy is the entropy of one softmax. Serving changes only the batch size of one process.

## 2. Methods

### 2.1 Generator

Dynamic channels are 6 vitals and 11 labs. Static channels are age and GCS. Vitals are missing independently on 4% of hours. The latent state is an Ornstein–Uhlenbeck process with time constant $\tau$. During a deterioration the mean is pulled by a fixed shift. A transient is a separate three-hour spike, or a deterioration-like mimic that does not overlap the event window. The probability of deterioration depends on age and GCS. This cohort has 86 events among 400 patients. Class labels are stable, transient, and deteriorate. The positive label at event time $t^\star$ is the half-open interval $[t^\star - 6, t^\star)$. Hours at and after $t^\star$ are ignored.

Each patient has two lab timestamps. For half of the patients who deteriorate, the second timestamp falls inside the positive interval. Vital missingness is an independent 4%. Observations add a small measurement noise. Error is scored against the latent trajectory, not against that noise.

### 2.2 Imputation

| Method | Before the first observation | After the first observation |
| --- | --- | --- |
| Zero-fill | 0 | 0 |
| Forward-fill | 0 | last observation |
| Time-decay | population mean $\mu$ | $\mu + (x_{\mathrm{last}} - \mu)\exp(-\Delta t / \tau)$ |

$\tau$ matches the generator. Scored cells are missing hours after the first observation. The reported figures are the mean of per-analyte standardized MSE, and the correlation of $\Delta t$ with $|\hat{x} - x_{\mathrm{last}}|$.

### 2.3 Head and alarms

The input at one hour is a window of shape $(28, 6)$, seventeen imputed channels plus eleven lab-staleness channels, and a static vector $(2,)$ for age and GCS. The 48-hour stay is sliced into those windows. It is not presented as shape $(19, 48)$. The network is two 1D convolutions of width 32 and kernel 3, then a linear map onto stable, transient, and deteriorate. Training uses 70% of patients, 20 epochs, and AdamW. Hours after the event, and hours before the window is full, are ignored in the loss.

Alarms are counted on valid hours of held-out patients. The baseline is a deteriorate probability above 0.7. Fusion adds entropy below 0.3 nats:

$$
H(p) = -\sum_k p_k \ln p_k.
$$

### 2.4 Serving

The trained head is set to eval and intra-op threads are set to 1. The device is CPU. One thousand windows of shape $(28, 6)$ enter one queue, each with its length-2 static vector. Only `max_batch` changes, from 1 to 128. Latency is residence time from a shared arrival.

## 3. Results

### 3.1 Imputation

Mean standardized MSE across 11 labs was 412.978, 0.870, and 0.723. Decay correlation was 0.485 for time-decay and 0 for forward-fill. Per-analyte values are in the results file. Lactate alone is 19.858, 2.807, and 2.486, with decay correlation 0.305 against about 0.49–0.53 for the other labs. A short $\tau$ and a large deterioration shift move $|x_{\mathrm{last}} - \mu|$ inside one gap, so elapsed time alone correlates weakly with that distance.

### 3.2 Alarms

On 180 deterioration hours the baseline rule has 50 true positives, 92 false positives, recall 0.278, and precision 0.352. Those 92 false positives are already the count at $s > 0.7$. Fusion has 36 true positives, 39 false positives, recall 0.200, and precision 0.480. False-alarm reduction is 57.6%. The lowest fusion score is 0.928. A distribution such as $(0.10, 0.15, 0.75)$ has $H(p) \approx 0.73$ nats, so it fires the baseline and fails the fusion rule.

On 4,391 stable hours the baseline alarm rate is 1.9% and the fusion rate is 0.8%. On 309 transient hours the rate falls from 3.2% to 0.6%. Most false positives are the tail of stable hours, not the transient label.

The 67 hours present in the baseline and absent from fusion are 14 true positives and 53 false positives. Dropping them lowers recall. Keeping them on a watchlist would hold page precision at the fusion level and leave those 14 hours visible on the same shift. The code does not build that queue.

### 3.3 Latency

Sequential P50 / P99 is 93.12 / 188.68 ms, at 5,253.9 beds/s. Batch 128 is 2.19 / 6.10 ms, at 163,580.4 beds/s. The P99 ratio is about 31 and the throughput ratio is 31.1. There is no experiment in which windows arrive spread over time under a delay budget.

## 4. Discussion

Time-decay beats forward-fill because the latent process reverts to the population mean and the metric scores that reversion. A world in which each patient carries a permanent offset from $\mu$ was not measured, and forward-fill would be favored there. The lactate residual is that limit in numbers. The staleness channel is an input so the head does not read an old lactate as a fresh one. It does not synthesize the missing rise.

The entropy cut is an operating point that keeps only peaked distributions. It is not a new uncertainty estimator. Nothing in this run shows that the operating point survives a distribution shift. Ovadia et al. are cited as the warning in the other direction.

The serving gain is the joint effect of fewer Python calls and a batched GEMM. It does not generalize to a GPU or to Triton's dynamic delay window.

## 5. Limitations

The cohort is synthetic. The label is not Sepsis-3, a major adverse event, or a cardiac arrest. MEWS and NEWS are not computed. Lead times of a rapid-response call are not tabulated. The entropy cut of 0.3 nats was chosen when the task was specified, not tuned to a recall target on the test set. It is still one seed of that test set, so selection and reporting are not fully separated.

## References

1. Che Z, Purushotham S, Cho K, Sontag D, Liu Y. Recurrent neural networks for multivariate time series with missing values. *Sci Rep.* 2018;8:6085. https://doi.org/10.1038/s41598-018-24271-9
2. Reyna MA, Josef CS, Jeter R, et al. Early prediction of sepsis from clinical data: the PhysioNet/Computing in Cardiology Challenge 2019. *Crit Care Med.* 2020;48(2):210–217. https://physionet.org/content/challenge-2019/1.0.0/
3. Sendelbach S, Funk M. Alarm fatigue: a patient safety concern. *AACN Adv Crit Care.* 2013;24(4):378–386. https://pubmed.ncbi.nlm.nih.gov/24153215/
4. Paine CW, Goel VV, Ely E, et al. Systematic review of physiologic monitor alarm characteristics and pragmatic interventions to reduce alarm frequency. *J Hosp Med.* 2016. https://pmc.ncbi.nlm.nih.gov/articles/PMC4778561/
5. Ovadia Y, Fertig E, Ren J, et al. Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift. NeurIPS. 2019. https://arxiv.org/abs/1906.02530
6. Maharaj R, Raffaele I, Wendon J. Rapid response systems: a systematic review and meta-analysis. *Crit Care.* 2015;19:254.
7. HL7 FHIR. https://hl7.org/fhir/
8. Triton Inference Server, dynamic batching. https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html
