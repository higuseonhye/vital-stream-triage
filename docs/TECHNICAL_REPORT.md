# 기술 보고: 합성 병동 스트림

한 CPU 실행의 공학 기록이다. 코드가 잰 것, 수식이 논문의 어디를 베끼고 어디를 베끼지 않았는지, 어떤 운영점에서 숫자가 성립하는지를 적는다. 논문 게재본이 아니고, 임상 성적의 근거가 아니다. 숫자의 원본은 [BENCHMARK_RESULTS.md](../BENCHMARK_RESULTS.md)다.

## 1. 범위

환자는 400명, 격자는 48시간의 1시간 칸, 시드 0이다. 검증 분할은 환자 120명이다. 알람은 그 120명만 센다. 복원 오차는 400명 전체의 검사 공백을 센다. 서빙은 검증 창에서 뽑은 1,000개를 한 시각에 넣는다.

잠재 검사는 인구 평균으로 되돌아가는 Ornstein-Uhlenbeck 과정이다. 바이탈은 거의 매 시간 있고 결측은 4%다. 검사 11종은 환자당 채혈 두 번이라 결측률 95.8%다. 악화 라벨은 이벤트 6시간 전부터 이벤트 시각 직전까지다. 이벤트 이후는 평가에서 뺀다. 일과성 스파이크와, 악화와 겉모양이 비슷한 3시간 구간이 따로 있다.

병원 추출, MIMIC, PhysioNet 2019 파일, FHIR 서버, Triton은 열지 않았다.

## 2. 시간 감쇄는 GRU-D의 입력 감쇄 중 고정 τ다

Che 등의 GRU-D는 결측 입력을 마지막 관측에서 경험적 평균으로 보낸다. 감쇄는 변수마다 두고, 데이터에서 학습한다.

```text
γ_t = exp( − max(0, W_γ δ_t + b_γ) )
x̂_t = m_t x_t + (1 − m_t) ( γ_t x_last + (1 − γ_t) x̃ )
```

`m_t`는 관측 마스크, `δ_t`는 마지막 관측 이후 시간, `x̃`는 경험적 평균이다. 같은 논문은 은닉 상태에도 감쇄를 두고, 마스크를 GRU 게이트에 넣는다. 공개 PyTorch 포트 하나가 [zhiyongc/GRU-D](https://github.com/zhiyongc/GRU-D)다. 저자 공식 저장소라고 말하지 않는다.

이 버퍼가 구현한 것은 그중 입력 감쇄뿐이다. `W_γ`와 `b_γ`는 없다. 검사별 τ를 코드에 고정했고, 그 τ는 생성기의 OU 시정수와 같다. 병원 데이터에 맞춘 값이 아니다.

```text
x̂(t) = μ + (x_last − μ) exp(−Δt / τ)
```

첫 채혈 전에는 time-decay만 μ를 쓰고, zero-fill과 forward-fill은 0을 쓴다. 표준화 MSE는 그 앞부분을 빼서, 세 방법 모두 마지막 관측이 있는 뒤의 빈 칸만 비교한다. 표준화는 잠재값과의 오차를 그 검사의 σ로 나눈 제곱의 평균이다. 감쇄 상관은 `|x̂ − x_last|`와 `Δt`의 피어슨 상관이다. Forward-fill은 그 거리가 0이라 상관이 0이다.

11종 평균 표준화 MSE는 zero-fill 412.978, forward-fill 0.870, time-decay 0.723이다. 11종 모두 time-decay가 forward-fill보다 작다. 나트륨의 zero-fill 3989는 0에 가깝지 않은 값을 0으로 둔 결과다. 평균을 그 한 검사가 끌어올린다.

젖산은 19.858, 2.807, 2.486이다. τ가 6시간으로 짧고 악화 이동이 σ 대비 크다. 채혈 사이에 잠재값이 올라가면 마지막 값도 인구 평균도 그 상승을 모른다. 감쇄가 forward-fill보다 나은 것과, 급변 지표의 잔차가 남는 것은 같이 성립한다.

헤드는 보정값 17채널 뒤에 검사 11종의 경과 시간을 붙인다. 상한은 시계열 길이인 48시간이다. `model.assemble_channels`가 그 결합이다. 바이탈 결측의 이진 마스크를 GRU 게이트에 넣는 GRU-D 입력은 아니다.

생성기의 결측은 임의 결측(MCAR)로 뽑지 않았다. 검사는 정해진 두 시각의 패널이다. 악화 환자의 50%는 두 번째 채혈이 경보 구간 안에 들어가게 놓는다. 나머지 악화는 그 구간이 비어 있다. 의사가 걱정할 때만 처방한다는 MNAR 이야기와 같은 구조가 아니다. 정보성 결측의 아주 좁은 축소다.

## 3. 엔트로피 게이트는 한 번의 소프트맥스다

클래스는 안정, 일과성, 악화다. 점수 `s`는 악화 클래스 확률이다. 엔트로피는 자연로그다.

```text
H(p) = − Σ_k p_k ln p_k
기준 알람: s > 0.7
융합 알람: s > 0.7 그리고 H(p) < 0.3
```

3-class에서 H < 0.3 nat는 최대 확률이 대략 0.93 이상일 때만 나온다. 이번 융합 알람의 최저 악화 점수는 0.928이었다. 0.7은 사실상 느슨한 선행 조건이고, 엔트로피 컷이 결정을 가른다. MC dropout, deep ensemble, temperature scaling은 돌리지 않았다.

검증 4,880시간 중 악화 180시간이다.

| 규칙 | TP | FP | FN | 정밀도 | 재현율 |
| --- | --- | --- | --- | --- | --- |
| s > 0.7 | 50 | 92 | 130 | 0.352 | 0.278 |
| s > 0.7 그리고 H < 0.3 | 36 | 39 | 144 | 0.480 | 0.200 |

위양성 감소 (92 − 39) / 92 = 0.576이다. 정밀도 비 0.480 / 0.352 = 1.363이다. 융합이 기준에서 뺀 67시간은 참 14, 거짓 53이다.

클래스별 평균 악화 점수는 안정 0.035, 일과성 0.062, 악화 0.325다. 악화 시간의 P90은 0.998이라 꼬리만 0.7을 넘는다. 6시간 라벨의 앞부분은 이동이 작아 점수가 낮다. 재현율 0.200은 그 창 전체를 분모로 둔 값이다.

Ovadia 등은 학습 분포 밖에서는 소프트맥스 최대 확률과 그 엔트로피가 과신하기 쉽고, 깊은 앙상블이 그 조건에서 더 견디었다고 보고했다. 이 벤치는 같은 생성 분포의 검증 분할만 본다. 그 논문의 분포 이동 실험을 재현하지 않았고, 소프트맥스 엔트로피가 분포 이동에 견딘다는 근거로 쓸 수 없다.

Sendelbach와 Funk의 72–99%는 임상 모니터 알람이 거짓이었다는 문헌 정리다. Paine 등의 고찰은 생리 모니터 알람의 74–99%가 조치 가능하지 않았다는 다른 문장이다. 어느 쪽도 이 표의 92건, 39건이 아니다.

VitalCare 논문의 AUROC, AUPRC, 위양성/참 비는 그 연구의 병원 코호트 성적이다. 이 표에 더하지 않는다. Sim 등의 전향 검증은 일반병동 6,039 방문에서 주요 이상반응 6시간 전 VC-MAES의 AUROC 0.918, NEWS 0.797, MEWS 0.722, 참 하나당 거짓 양성을 최대 71% 낮췄다고 보고한다. Kim 등의 프리프린트는 VC-MAES를 6시간 내 악화, VC-SEPS를 4시간 내 패혈증으로 두고 MEWS, NEWS, SOFA와 비교한다. 입력은 시간 단위로 맞춘 바이탈·검사·나이·GCS다. 우리 헤드는 폭 32의 1D 합성곱 두 층이고, 그 네트워크가 아니다.

PhysioNet/CinC 2019는 ICU 시간 단위 40변수(바이탈 8, 검사 26, 인구학 6)로 패혈증을 6시간 앞서 예측하고, 늦은 예측과 거짓 알람을 깎는 유틸리티로 채점한다. 이 벤치는 그 데이터도 그 유틸리티도 쓰지 않는다. 6시간 창은 그 챌린지와 맞춘 라벨 정의다.

## 4. 서빙은 같은 큐의 배치 크기 차이다

`MicroBatchQueue`는 워커 하나다. 순차는 `max_batch = 1`, 마이크로배치는 128이다. 1,000개 창의 도착 시각은 같다. 지연은 그 시각부터 해당 창이 들어 있는 배치의 forward가 끝날 때까지다. 배치 안 창은 같은 완료 시각을 공유하므로 P95와 P99가 한 값으로 붙을 수 있다. 이번 마이크로배치 P99 6.10 ms가 그렇다. PyTorch 스레드는 이 구간만 1로 둔다.

| 모드 | max batch | P99 | 처리량 |
| --- | --- | --- | --- |
| sequential | 1 | 188.68 ms | 5,253.9 병상/초 |
| microbatch | 128 | 6.10 ms | 163,580.4 병상/초 |

처리량 비는 163580.4 / 5253.9 = 31.1이다. 순차 P99는 50 ms를 넘고, 마이크로배치 P99는 넘지 않는다. 이전 실행의 마이크로배치 P99는 4.9 ms였다. 알람 건수와 MSE는 같았고 지연만 움직였다. 지연 문장은 그 실행의 자릿수로 말하고, 소수점까지 고정된 SLA 증명으로 말하지 않는다.

Triton dynamic batcher는 상태가 없는 모델에서 대기 중인 요청으로 배치를 만들고, 선호 배치가 안 모이면 `max_queue_delay_microseconds`까지 기다렸다가 있는 만큼 보낸다. 문서: [Batchers](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html), [batcher.md](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/batcher.md). 이 프로세스는 Triton 스케줄러를 호출하지 않는다. 배치 128은 코드에 박은 값이지, 지연 상한으로 닫힌 동적 배치가 아니다.

## 5. 운영점

복원 기본값은 time-decay다. 11종 평균과 젖산 모두에서 forward-fill보다 MSE가 작고, 경과 시간에 따라 마지막 값에서 멀어진다. 급변 검사는 경과 시간 채널 없이 감쇄만으로 끝내지 않는다. 그 채널은 이미 헤드 입력에 있다.

알람의 측정된 즉시 호출은 융합 게이트다. 재현율을 기준 수준으로 유지하려면 융합이 버린 67시간을 관찰 큐로 남겨야 한다. 그 큐는 요구사항이고 코드 경로는 아니다. [PRD.md](PRD.md) 4절.

서빙의 측정된 기본값은 스레드 하나, 배치 128이다. 1,000창 버스트의 P99가 50 ms 아래였다. 창이 1초에 걸쳐 흩어져 들어오는 부하는 재지 않았다.

## 6. 참고

- Che Z, Purushotham S, Cho K, Sontag D, Liu Y. Recurrent neural networks for multivariate time series with missing values. *Sci Rep.* 2018;8:6085. <https://doi.org/10.1038/s41598-018-24271-9>
- Reyna MA, et al. Early prediction of sepsis from clinical data: the PhysioNet/Computing in Cardiology Challenge 2019. *Crit Care Med.* 2020;48(2):210–217. 데이터: <https://physionet.org/content/challenge-2019/1.0.0/>
- Sendelbach S, Funk M. Alarm fatigue: a patient safety concern. *AACN Adv Crit Care.* 2013;24(4):378–386. <https://pubmed.ncbi.nlm.nih.gov/24153215/>
- Paine CW, et al. Systematic review of physiologic monitor alarm characteristics and pragmatic interventions to reduce alarm frequency. <https://pmc.ncbi.nlm.nih.gov/articles/PMC4778561/>
- Ovadia Y, et al. Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift. NeurIPS 2019. <https://arxiv.org/abs/1906.02530>
- Sung MD, et al. Event prediction model considering time and input error using electronic medical records in the intensive care unit. *JMIR Med Inform.* 2021;9(11):e26426. <https://medinform.jmir.org/2021/11/e26426/> — ICU 이벤트와 입력 오류 실험이다. 일반병동 VitalCare 성적이 아니다.
- Sim T, et al. Prospective external validation of a deep-learning-based early-warning system for major adverse events in general wards in South Korea. *Acute Crit Care.* 2025;40(2):197–208. <https://doi.org/10.4266/acc.000525>
- Kim YJ, et al. Development and validation of VC-MAES and VC-SEPS. medRxiv. 2025. <https://doi.org/10.1101/2025.08.20.25334022> — 프리프린트다.
- HL7 FHIR `Observation`, `Patient`, `Encounter`. <https://hl7.org/fhir/>
- NVIDIA Triton dynamic batching. 위 4절의 두 URL.
- Maharaj R, Raffaele I, Wendon J. Rapid response systems: a systematic review and meta-analysis. *Crit Care.* 2015;19:254. 신속대응팀 문헌이다. 이 벤치의 엔드포인트가 아니다.
