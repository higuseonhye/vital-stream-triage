# 기술 노트: 합성 병동 EMR에서 감쇄 보정, 엔트로피 알람, 마이크로배치

이 글은 저장소의 한 번 실행을 논문 형식으로 옮긴 기술 노트다. 학술지 논문이 아니고, 동료 심사를 거치지 않았다. 병원 성적을 주장하지 않는다.

## 초록

1시간 격자 합성 EMR 400명에서 세 가지를 비교했다. 검사 11종은 48시간 중 두 번만 관측되어 결측 95.8%였다. 인구 평균으로의 고정 시간 감쇄는 forward-fill보다 표준화 MSE가 작았고(0.723 대 0.870), zero-fill(412.978)과는 차이가 컸다. 젖산은 2.486 대 2.807로 방향은 같았으나 잔차가 가장 컸다. 악화 확률 0.7 초과를, 3-class 엔트로피 0.3 nat 미만과 함께 쓰면 검증 집합의 위양성이 92건에서 39건으로 줄고 정밀도는 0.352에서 0.480으로 올랐다. 재현율은 0.278에서 0.200으로 내려갔다. 한 CPU 스레드에서 1,000창을 동시에 넣으면 배치 1의 P99는 188.7 ms, 배치 128은 6.1 ms였고 처리량은 31.1배였다. 감쇄 가중치, 분포 이동, FHIR, Triton은 실험에 없다.

## 1. 서론

전자의무기록의 검사는 매 시간이 아니라 처방된 시각에 찍힌다. Che 등은 오래 비어 있는 값은 기본값에 가깝고, 오래 비어 있으면 그 값의 영향도 옅어진다고 보고 GRU-D를 제안했다. 결측 패턴 자체도 예측에 쓰일 수 있어, 감쇄를 고정하지 않고 학습하는 쪽을 택했다. 병동 조기경보 문헌은 다른 축에서 위양성을 문제로 둔다. 모니터 알람 고찰은 거짓이거나 조치로 이어지지 않는 알람이 대부분이라고 적고, 일반병동 딥러닝 점수는 MEWS·NEWS 대비 참 하나당 거짓 호출을 줄였다고 보고한다. 그 성적과 이 노트의 건수는 다른 데이터다.

이 노트는 그 메커니즘 가운데 측정 가능한 축소만 고정한다. 입력 감쇄의 τ를 학습하지 않는다. 엔트로피는 한 소프트맥스의 엔트로피만 쓴다. 서빙은 Triton 없이 한 프로세스의 배치 크기만 바꾼다.

## 2. 방법

### 2.1 생성

동적 채널은 바이탈 6종과 검사 11종이다. 정적 채널은 나이와 GCS다. 잠재 상태는 시간 상수 τ의 OU 과정이고, 악화 구간에서는 정해진 이동이 평균을 끌어간다. 악화 여부는 나이와 GCS에 따라 확률이 달라진다. 이번 코호트 400명 중 이벤트는 86명이다. 라벨의 양성은 이벤트 시각 `t*`에 대해 `[t*−6, t*)`다. `t*` 이후는 무시한다.

검사 마스크는 환자당 두 시각이다. 악화가 있는 환자의 절반은 두 번째 시각이 양성 구간 안에 들어가게 둔다. 바이탈 결측은 독립적으로 4%다. 관측값에는 작은 측정 잡음이 더해지고, 오차의 기준은 그 잡음이 아니라 잠재 궤적이다.

이 19종은 VC-MAES 프리프린트가 나열한 19개 입력과 개수만 같다. 그 목록에는 동맥혈 pH와 중탄산염이 있고, 이 생성기에는 헤모글로빈과 BUN이 있다.

### 2.2 보정

빈 칸의 세 가지 추정:

| 방법 | 첫 관측 전 | 첫 관측 후 |
| --- | --- | --- |
| zero-fill | 0 | 0 |
| forward-fill | 0 | 마지막 관측 |
| time-decay | 인구 평균 μ | μ + (x_last − μ) exp(−Δt / τ) |

τ는 생성기와 같다. 평가 칸은 첫 관측 이후의 결측뿐이다. 지표는 검사별 표준화 MSE의 산술 평균, 그리고 Δt와 `|x̂ − x_last|`의 상관이다.

### 2.3 헤드와 알람

입력은 과거 6시간의 보정 채널과 검사 경과 시간, 그리고 나이·GCS다. 네트워크는 커널 3의 1D 합성곱 두 층(폭 32)과 선형 3-class 출력이다. 학습은 환자 70%, 20 epoch, AdamW다. 이벤트 이후와 창이 차지 않는 앞 시간은 손실에서 무시한다.

알람은 검증 환자의 유효 시간에만 집계한다. 기준은 악화 확률 > 0.7이다. 융합은 같은 조건에 엔트로피 < 0.3 nat를 더한다.

### 2.4 서빙

학습이 끝난 헤드를 평가 모드로 두고 intra-op 스레드를 1로 한다. 1,000개 창을 한 큐에 넣고 `max_batch`만 1 또는 128로 바꾼다. 지연은 공통 도착 시각 기준의 체류 시간이다.

## 3. 결과

### 3.1 보정

11종 평균 표준화 MSE는 412.978, 0.870, 0.723이었다. 감쇄 상관은 time-decay 0.485, forward-fill 0이었다. 종별 값은 결과 파일을 따른다. 젖산만 보면 19.858, 2.807, 2.486이고 감쇄 상관은 0.305로 다른 검사(대략 0.49–0.53)보다 낮다. 짧은 τ와 큰 악화 이동이 한 칸 안의 `|x_last − μ|`를 흔들어, 경과 시간만으로 거리의 상관이 약해진다.

### 3.2 알람

악화 180시간에서 기준 규칙은 참 50, 거짓 92, 재현율 0.278, 정밀도 0.352였다. 융합은 참 36, 거짓 39, 재현율 0.200, 정밀도 0.480이었다. 위양성 감소는 57.6%다. 융합 알람의 최저 점수는 0.928이라, 엔트로피 조건은 0.7보다 훨씬 높은 확률 질량을 요구했다.

안정 4,391시간의 기준 알람 비율은 1.9%, 융합은 0.8%였다. 일과성 309시간은 3.2%에서 0.6%로 줄었다. 위양성의 대부분은 일과성 라벨이 아니라 안정 시간의 두꺼운 꼬리다.

기준에는 있고 융합에는 없는 67시간은 참 14, 거짓 53이다. 이 집합을 버리면 재현율이 내려가고, 관찰 큐로 남기면 즉시 호출의 정밀도는 융합 수준을 유지한 채 그 14시간을 같은 근무에서 다시 볼 수 있다. 코드는 아직 큐를 만들지 않는다.

### 3.3 지연

순차 P50/P99는 93.1/188.7 ms, 처리량 5,254 병상/초였다. 배치 128은 P50/P99 2.2/6.1 ms, 처리량 163,580 병상/초였다. P99 비는 약 31배, 처리량 비는 31.1배다. 창이 시간차를 두고 도착하는 지연 상한 실험은 없다.

## 4. 고찰

감쇄가 forward-fill을 이긴 이유는 잠재 과정이 인구 평균으로 되돌아가고, 평가가 그 되돌아감을 재기 때문이다. 환자 고유의 기저값을 영구 오프셋으로 두면 forward-fill이 유리해지는 세계를 이 노트는 측정하지 않았다. 젖산 잔차는 그 한계를 수치로 보여 준다. 경과 시간 채널은 모델이 오래된 젖산을 새 젖산처럼 읽지 않게 하려는 입력이고, 상승 자체를 만들어 내지는 못한다.

엔트로피 컷의 이득은 불확실성의 새 추정기가 아니라 더 뾰족한 확률만 남긴 운영점이다. 분포가 이동하면 이 운영점이 유지된다는 증거는 없다. Ovadia 등의 결과는 그 반대 방향의 경고로 인용한다.

서빙 이득은 파이썬 호출 횟수와 배치 GEMM이 함께 만든 것이다. GPU나 Triton의 동적 지연 창으로 일반화하지 않는다.

## 5. 한계

합성 코호트다. Sepsis-3, 주요 이상반응, 코드 블루를 라벨로 쓰지 않았다. MEWS와 NEWS를 계산하지 않았다. 신속대응팀 호출의 리드 타임 분포를 집계하지 않았다. 엔트로피 0.3 nat는 작업 시작 때의 컷이지, 검증 집합에서 재현율을 맞추려고 고른 컷이 아니다. 그래도 검증 집합의 한 시드이므로 컷 선택과 보고가 완전히 분리되어 있지는 않다.

## 참고문헌

1. Che Z, Purushotham S, Cho K, Sontag D, Liu Y. Recurrent neural networks for multivariate time series with missing values. Sci Rep. 2018;8:6085. https://doi.org/10.1038/s41598-018-24271-9
2. Reyna MA, Josef CS, Jeter R, et al. Early prediction of sepsis from clinical data: the PhysioNet/Computing in Cardiology Challenge 2019. Crit Care Med. 2020;48(2):210-217. https://physionet.org/content/challenge-2019/1.0.0/
3. Sendelbach S, Funk M. Alarm fatigue: a patient safety concern. AACN Adv Crit Care. 2013;24(4):378-386. https://pubmed.ncbi.nlm.nih.gov/24153215/
4. Paine CW, Goel VV, Ely E, et al. Systematic review of physiologic monitor alarm characteristics and pragmatic interventions to reduce alarm frequency. https://pmc.ncbi.nlm.nih.gov/articles/PMC4778561/
5. Ovadia Y, Fertig E, Ren J, et al. Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift. NeurIPS. 2019. https://arxiv.org/abs/1906.02530
6. Sung MD, Hahn S, Han CH, et al. Event prediction model considering time and input error using electronic medical records in the intensive care unit: retrospective study. JMIR Med Inform. 2021;9(11):e26426. https://medinform.jmir.org/2021/11/e26426/
7. Sim T, Cho EY, Kim J, et al. Prospective external validation of a deep-learning-based early-warning system for major adverse events in general wards in South Korea. Acute Crit Care. 2025;40(2):197-208. https://doi.org/10.4266/acc.000525
8. Kim YJ, Hahn S, Kim KJ, et al. Development and validation of VC-MAES and VC-SEPS: deep learning-based early warning systems for hospitalized patients. medRxiv. 2025. https://doi.org/10.1101/2025.08.20.25334022
9. HL7 FHIR. https://hl7.org/fhir/
10. NVIDIA Triton Inference Server. Dynamic batching. https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html
11. Maharaj R, Raffaele I, Wendon J. Rapid response systems: a systematic review and meta-analysis. Crit Care. 2015;19:254.
