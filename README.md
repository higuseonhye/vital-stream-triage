# vital-stream-triage

합성 병동 EMR에서 세 가지를 재는 벤치마크다. 혈액검사가 하루 한두 번만 찍힐 때 빈 칸을 어떻게 메울지, 악화 점수가 0.7을 넘어도 분포가 뾰족하지 않으면 알람을 끌지, 1,000병상 창이 한 번에 들어올 때 한 CPU 스레드의 P99가 50 ms 안에 있는지를 본다.

병원 추출 파일은 읽지 않는다. 에이아이트릭스 바이탈케어나 뷰노 모델을 재현하지 않으며, 임상 장치가 아니다. 숫자는 [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md)의 한 번 실행이다. 그 실행을 제품 요구, 구현과의 차이, 기술 노트로 나눈 글이 있다.

- [docs/PRD.md](docs/PRD.md) — 범위, 성공 기준, 아직 없는 호출/관찰 분기, 면접에서 말할 수 있는 세 문장
- [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md) — GRU-D, 알람 문헌, VitalCare 논문, FHIR, Triton 중 코드가 실제로 구현한 부분
- [docs/PAPER.md](docs/PAPER.md) — 같은 실행의 기술 노트. 학술지 논문이 아니다

## 로드맵

이 저장소가 1순위다. 2순위 `dicom-stream-triage`, 3순위 `surgical-scene-triage`는 아직 만들지 않는다.

| 순서 | 저장소 | 지금 |
| --- | --- | --- |
| 1 | `vital-stream-triage` | 이 저장소 |
| 2 | `dicom-stream-triage` | 착수 전 |
| 3 | `surgical-scene-triage` | 착수 전 |

## 19개 항목

고빈도 바이탈 6종: 수축기 혈압, 이완기 혈압, 맥박, 호흡수, 체온, 산소포화도.

저빈도 검사 11종: 백혈구, 헤모글로빈, 헤마토크릿, 혈소판, 빌리루빈, 크레아티닌, BUN, CRP, 젖산, 나트륨, 칼륨. 명세에 이름을 적지 않은 네 가지(헤마토크릿, BUN, 나트륨, 칼륨)는 통상 화학·혈구 검사로 11종을 채운 것이다.

정적 2종: 나이, GCS.

격자는 1시간이다. 48시간 동안 혈액 패널은 환자당 두 번이라 검사 결측은 90%를 넘긴다. 모니터의 1초 파형은 아니다.

시간 감쇠는 모집단 평균으로 돌아간다.

```text
x_hat(t) = mu + (x_last - mu) * exp(-dt / tau)
```

`tau`는 검사별 신선도 상수이고, 생성기의 OU 시정수와 같다. 병원 데이터에 맞춘 값이 아니다.

## 실험

| 실험 | 비교 | 목표 |
| --- | --- | --- |
| 결측 복원 | zero-fill, forward-fill, time-decay | time-decay의 표준화 MSE가 가장 작고, 경과 시간에 따라 마지막 값에서 멀어진다 |
| 알람 | 악화 소프트맥스 > 0.7 vs 그 조건과 엔트로피 < 0.3 nat | 위양성 35% 이상 감소, 정밀도는 기준 이상 |
| 서빙 | 같은 큐의 batch 1 vs micro-batch | 한 CPU 스레드에서 1,000병상 동시 유입의 P99 < 50 ms |

엔트로피는 한 번의 3-class 소프트맥스에 대한 `-sum p_k ln p_k` 다. 0.3 nat 이하는 분포가 한 클래스에 매우 쏠린 경우뿐이라, 점수 0.7보다 더 엄격한 신뢰도 컷이다. MC dropout이 아니다.

악화 라벨은 이벤트 6시간 전부터 이벤트 시각 직전까지다. 이벤트가 난 뒤의 시간은 위양성으로 세지 않도록 평가에서 뺀다. 알람 지표는 환자 단위 30% 검증 분할만 쓴다.

## 실행

Python 3.10+ 와 `requirements.txt`. CPU면 된다.

```bash
python -m pip install -r requirements.txt
python -m vital_triage.runner --synthetic
streamlit run app.py
```

대시보드는 `http://127.0.0.1:8503` 이다. `--synthetic` 이 없으면 러너는 중단된다.

기본값: 환자 400, 48시간, 학습 20 epoch, 병상 1,000, micro-batch 128, 시드 0.

그 기본 실행에서 검사 11종의 표준화 MSE는 zero-fill 412.98, forward-fill 0.870, time-decay 0.723이었다. 11종 모두 time-decay가 forward-fill보다 작았다. 젖산은 19.86, 2.81, 2.49로 방향은 같지만 잔차가 가장 크다. 채혈 사이에 생긴 상승은 어느 쪽도 복원하지 못한다.

같은 실행에서 엔트로피 게이트는 위양성을 92건에서 39건으로 줄였다(57.6%). 정밀도는 0.352에서 0.480으로 올랐고, 재현율은 0.278에서 0.200으로 내려갔다. 융합이 뺀 67건은 참 14건과 거짓 53건이라, 제품으로 가면 그 67건은 삭제하지 않고 관찰 큐에 둔다. 그 큐는 아직 코드에 없다. 1,000병상 동시 유입의 P99는 순차 188.7 ms, micro-batch 6.1 ms였고 처리량은 5,254에서 163,580 병상/초로 31.1배였다. 지연은 다시 재면 조금씩 달라지고, 알람과 복원 오차는 시드가 같으면 같다. Triton과 FHIR는 실행하지 않았다.

근거로 쓰는 1차 문헌은 GRU-D(Che et al., Sci Rep 2018), PhysioNet/CinC 2019, Sendelbach와 Funk의 알람 피로 정리(AACN Adv Crit Care 2013), Ovadia 등의 분포 이동 불확실성(NeurIPS 2019), VitalCare의 일반병동 검증(Sim et al., Acute Crit Care 2025)과 VC-MAES/VC-SEPS 프리프린트(Kim et al., medRxiv 2025)다. 각 문헌의 숫자와 이 저장소의 숫자를 섞지 않는 이유는 기술 보고에 적혀 있다. Ovadia 등의 결론은 소프트맥스 신뢰도가 분포 이동에서 과신하기 쉽다는 쪽이지, 이 엔트로피 컷이 그 이동에 견딘다는 뜻이 아니다.

## 레이아웃

```text
vital_triage/          생성, 감쇄 버퍼, 1D 헤드, 알람, 러너
app.py                 병상 곡선과 알람 슬라이더
docs/PRD.md            제품 요구와 말할 수 있는 문장
docs/TECHNICAL_REPORT.md
docs/PAPER.md          기술 노트
```
