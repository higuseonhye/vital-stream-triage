"""Run the three synthetic vital-stream benchmarks and write the report."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch

from . import __version__
from .buffer import METHODS, impute, score_imputation, self_check
from .generator import LAB_INDEX, sample_cohort
from .model import benchmark_serving, fit_head, prepared_windows, sample_serving_batch
from .triage_engine import evaluate_alarms

ROOT = Path(__file__).resolve().parents[1]
FA_REDUCTION_TARGET = 0.35
SLA_MS_DEFAULT = 50.0


def collect_environment() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "cuda_available": bool(torch.cuda.is_available()),
        "serving_device": "cpu",
        "serving_threads": 1,
    }


def _pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.{digits}f}%"


def _num(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def render_markdown(report: dict) -> str:
    imputation = {row["method"]: row for row in report["imputation"]["methods"]}
    alarms = report["alert_fatigue"]
    serving = {row["mode"]: row for row in report["serving"]["rows"]}
    decay = imputation["time_decay"]
    forward = imputation["forward"]
    zero = imputation["zero"]
    feature_rows = []
    forward_by_name = {row["name"]: row for row in forward["per_feature"]}
    zero_by_name = {row["name"]: row for row in zero["per_feature"]}
    for row in decay["per_feature"]:
        feature_rows.append(
            [
                row["korean"],
                row["name"],
                _num(zero_by_name[row["name"]]["standardized_mse"], 3),
                _num(forward_by_name[row["name"]]["standardized_mse"], 3),
                _num(row["standardized_mse"], 3),
                _num(row["decay_reflection"], 3),
            ]
        )
    feature_rows.append(
        [
            "11종 평균",
            "macro",
            _num(zero["standardized_mse"], 3),
            _num(forward["standardized_mse"], 3),
            _num(decay["standardized_mse"], 3),
            _num(decay["decay_reflection"], 3),
        ]
    )
    class_rows = [
        [
            row["class"],
            str(row["hours"]),
            _num(row["mean_score"], 3),
            _num(row["p90_score"], 3),
            _num(row["mean_entropy"], 3),
            _pct(row["baseline_alarm_rate"]),
            _pct(row["fusion_alarm_rate"]),
        ]
        for row in alarms["by_class"]
    ]
    serve_rows = [
        [
            row["mode"],
            str(row["max_batch"]),
            _num(row["p50_ms"], 2),
            _num(row["p95_ms"], 2),
            _num(row["p99_ms"], 2),
            _num(row["throughput_beds_per_sec"], 1),
            "yes" if row["sla_met"] else "no",
        ]
        for row in report["serving"]["rows"]
    ]
    targets = report["targets"]
    lines = [
        "# Vital stream triage — benchmark results",
        "",
        "합성 시간별 EMR이다. 병원 기록, 에이아이트릭스 바이탈케어, 뷰노 모델을 읽거나 재현하지 않는다. 수치는 이 파일과 함께 저장된 한 번의 CPU 실행이다.",
        "",
        f"- 환자 {report['cohort']['patients']}명, {report['cohort']['hours']}시간 격자, 시드 {report['seed']}",
        f"- 검증 환자 {report['cohort']['test_patients']}명으로 알람을 센다. 결측 복원 오차는 코호트 전체의 검사 공백이다.",
        f"- 혈액검사 결측률 {100.0 * report['imputation']['lab_missing_rate']:.1f}%. 평가는 첫 채혈 이후의 빈 칸만 쓴다.",
        "",
        "## Exp 1. 결측치 복원",
        "",
        "표준화 MSE는 잠재 궤적 대비 오차를 그 검사의 정상 표준편차로 나눈 제곱의 평균이다. 감쇄 반영도는 마지막 측정과의 거리와 경과 시간의 피어슨 상관이다. Forward-fill은 거리를 0으로 두므로 상관이 0이다.",
        "",
        _md_table(
            ["검사", "key", "zero", "forward", "time-decay", "time-decay 감쇄 상관"],
            feature_rows,
        ),
        "",
        f"순서 목표(time-decay < forward < zero): {'충족' if targets['imputation_order'] else '미충족'}.",
        f"감쇄 상관 목표(time-decay > forward): {'충족' if targets['decay_reflection'] else '미충족'}.",
        (
            f"젖산 표준화 MSE는 zero {_num(zero_by_name['lactate']['standardized_mse'], 3)}, "
            f"forward {_num(forward_by_name['lactate']['standardized_mse'], 3)}, "
            f"time-decay {_num(next(row['standardized_mse'] for row in decay['per_feature'] if row['name']=='lactate'), 3)}."
        ),
        "",
        "## Exp 2. 알람 피로",
        "",
        "베이스라인은 악화 클래스 소프트맥스 > 0.7. 융합 규칙은 여기에 3-class 엔트로피 < 0.3 nat를 더한다. 이 엔트로피는 그 소프트맥스 분포의 엔트로피라서, 0.3 nat 아래는 최대 확률이 대략 0.93 이상인 뾰족한 분포만 통과한다.",
        "",
        _md_table(
            ["클래스", "시간", "평균 점수", "점수 P90", "평균 엔트로피", "기준 알람", "융합 알람"],
            class_rows,
        ),
        "",
        _md_table(
            ["규칙", "TP", "FP", "FN", "정밀도", "재현율"],
            [
                [
                    "softmax > 0.7",
                    str(alarms["baseline"]["tp"]),
                    str(alarms["baseline"]["fp"]),
                    str(alarms["baseline"]["fn"]),
                    _num(alarms["baseline"]["precision"], 3),
                    _num(alarms["baseline"]["recall"], 3),
                ],
                [
                    "softmax > 0.7 and H < 0.3",
                    str(alarms["fusion"]["tp"]),
                    str(alarms["fusion"]["fp"]),
                    str(alarms["fusion"]["fn"]),
                    _num(alarms["fusion"]["precision"], 3),
                    _num(alarms["fusion"]["recall"], 3),
                ],
            ],
        ),
        "",
        f"위양성 감소 {_pct(alarms['false_alarm_reduction'])} (목표 35% 이상): {'충족' if targets['false_alarm_reduction'] else '미충족'}.",
        f"정밀도 방어율 {_num(alarms['precision_defense'], 3)} (융합 정밀도 / 기준 정밀도, 목표 1 이상): {'충족' if targets['precision_defense'] else '미충족'}.",
        f"융합 알람 중 최저 악화 점수 {_num(alarms['min_score_among_fusion_alarms'], 3)}.",
        "재현율은 이 컷의 반대편이다. 융합 규칙이 재현율을 얼마나 깎는지는 위 표의 FN으로 본다.",
        "",
        "## Exp 3. 1,000병상 동시 유입",
        "",
        "병상 창이 같은 시각에 한 CPU 스레드 큐로 들어온다. 지연은 공통 도착 시각부터 그 요청의 배치가 끝날 때까지다. 순차 처리는 같은 큐의 max_batch=1이다.",
        "",
        _md_table(
            ["모드", "max batch", "P50 ms", "P95 ms", "P99 ms", "병상/초", f"P99 < {serving['microbatch']['sla_ms']:.0f} ms"],
            serve_rows,
        ),
        "",
        f"마이크로배치 P99 {_num(serving['microbatch']['p99_ms'], 2)} ms: {'충족' if targets['serving_p99'] else '미충족'}.",
        f"처리량 {serving['microbatch']['throughput_beds_per_sec']:.1f} vs 순차 {serving['sequential']['throughput_beds_per_sec']:.1f} 병상/초.",
        "",
        "## 이 실행이 아닌 것",
        "",
        "- 임상 장치 성능, 민감도 청구, 또는 특정 병원 경보 정책의 적합성 평가가 아니다.",
        "- 엔트로피 게이트는 인식적 불확실성(MC dropout, 앙상블)이 아니라 한 번의 소프트맥스 엔트로피다.",
        "- 1초 모니터 파형이 아니다. 격자는 병동 EMR처럼 1시간이다.",
        "",
    ]
    return "\n".join(lines) + "\n"


def run_benchmark(
    n_patients: int = 400,
    n_hours: int = 48,
    epochs: int = 20,
    batch_size: int = 512,
    n_beds: int = 1000,
    max_batch: int = 128,
    sla_ms: float = SLA_MS_DEFAULT,
    seed: int = 0,
) -> tuple[dict, dict]:
    self_check()
    np.random.seed(seed)
    torch.manual_seed(seed)
    cohort = sample_cohort(n_patients, n_hours, seed)
    lab_missing = float(1.0 - cohort.mask[:, :, LAB_INDEX].mean())
    if lab_missing < 0.90:
        raise RuntimeError(f"lab missing rate {lab_missing:.3f} is below 90%")

    imputation = [score_imputation(cohort, method) for method in METHODS]
    imputed, hours_since = impute(cohort.observed, cohort.mask, "time_decay")
    fit = fit_head(cohort, imputed, hours_since, epochs=epochs, batch_size=batch_size, seed=seed)
    alarms = evaluate_alarms(fit.probabilities, fit.labels, fit.test_ids)
    windows, static, _hour_index = prepared_windows(cohort, imputed, hours_since, fit.normalizer)
    bed_x, bed_s = sample_serving_batch(windows, static, fit.test_ids, n_beds, seed)
    serving = benchmark_serving(fit.model, bed_x, bed_s, max_batch=max_batch, sla_ms=sla_ms)

    by_method = {row["method"]: row for row in imputation}
    micro = next(row for row in serving["rows"] if row["mode"] == "microbatch")
    sequential = next(row for row in serving["rows"] if row["mode"] == "sequential")
    reduction = alarms["false_alarm_reduction"]
    defense = alarms["precision_defense"]
    fusion_recall = alarms["fusion"]["recall"]
    targets = {
        "imputation_order": bool(
            by_method["time_decay"]["standardized_mse"]
            < by_method["forward"]["standardized_mse"]
            < by_method["zero"]["standardized_mse"]
        ),
        "decay_reflection": bool(
            by_method["time_decay"]["decay_reflection"] > by_method["forward"]["decay_reflection"]
        ),
        "false_alarm_reduction": bool(reduction is not None and reduction >= FA_REDUCTION_TARGET),
        "precision_defense": bool(defense is not None and defense >= 1.0 and fusion_recall not in (None, 0)),
        "serving_p99": bool(micro["sla_met"]),
        "serving_throughput": bool(micro["throughput_beds_per_sec"] > sequential["throughput_beds_per_sec"]),
    }
    report = {
        "suite": "vital-stream-triage",
        "version": __version__,
        "synthetic": True,
        "seed": seed,
        "environment": collect_environment(),
        "cohort": {
            "patients": n_patients,
            "hours": n_hours,
            "train_patients": int(fit.train_ids.size),
            "test_patients": int(fit.test_ids.size),
            "events": int((cohort.event_time >= 0).sum()),
            "train_loss": fit.train_loss,
            "parameters": int(sum(parameter.numel() for parameter in fit.model.parameters())),
            "epochs": epochs,
        },
        "imputation": {"lab_missing_rate": lab_missing, "methods": imputation},
        "alert_fatigue": alarms,
        "serving": serving,
        "targets": targets,
    }
    other = {method: impute(cohort.observed, cohort.mask, method)[0] for method in METHODS if method != "time_decay"}
    bundle = {
        "values": cohort.values[fit.test_ids],
        "observed": cohort.observed[fit.test_ids],
        "mask": cohort.mask[fit.test_ids],
        "imp_zero": other["zero"][fit.test_ids],
        "imp_forward": other["forward"][fit.test_ids],
        "imp_time_decay": imputed[fit.test_ids],
        "probs": fit.probabilities[fit.test_ids],
        "labels": fit.labels[fit.test_ids],
        "hour_index": fit.hour_index,
        "event_time": cohort.event_time[fit.test_ids],
        "transient_start": cohort.transient_start[fit.test_ids],
        "transient_kind": cohort.transient_kind[fit.test_ids],
        "age": cohort.age[fit.test_ids],
        "gcs": cohort.gcs[fit.test_ids],
    }
    return report, bundle


def write_outputs(report: dict, bundle: dict, output: Path, markdown_path: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    np.savez_compressed(output.parent / "dashboard.npz", **bundle)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthetic vital-stream imputation, alarm, and serving benchmark.")
    parser.add_argument("--synthetic", action="store_true", help="Required. Use the simulated cohort.")
    parser.add_argument("--patients", type=int, default=400)
    parser.add_argument("--hours", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--beds", type=int, default=1000)
    parser.add_argument("--max-batch", type=int, default=128)
    parser.add_argument("--sla-ms", type=float, default=SLA_MS_DEFAULT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=str(ROOT / "results" / "benchmark_report.json"))
    parser.add_argument("--report", default=str(ROOT / "BENCHMARK_RESULTS.md"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.synthetic:
        print("Pass --synthetic. This suite does not read hospital EMR extracts.", file=sys.stderr)
        return 2
    print(
        f"vital_triage {__version__}: {args.patients} patients, {args.epochs} epochs, {args.beds} beds.",
        flush=True,
    )
    report, bundle = run_benchmark(
        n_patients=args.patients,
        n_hours=args.hours,
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_beds=args.beds,
        max_batch=args.max_batch,
        sla_ms=args.sla_ms,
        seed=args.seed,
    )
    write_outputs(report, bundle, Path(args.output), Path(args.report))
    targets = report["targets"]
    alarms = report["alert_fatigue"]
    decay = next(row for row in report["imputation"]["methods"] if row["method"] == "time_decay")
    micro = next(row for row in report["serving"]["rows"] if row["mode"] == "microbatch")
    print(f"Exp1 standardized MSE (time-decay) {decay['standardized_mse']:.4f} order={targets['imputation_order']}", flush=True)
    reduction = alarms["false_alarm_reduction"]
    reduction_text = "n/a" if reduction is None else f"{100.0 * reduction:.1f}%"
    print(
        f"Exp2 false-alarm reduction {reduction_text} precision_defense={alarms['precision_defense']} met={targets['false_alarm_reduction']}",
        flush=True,
    )
    print(f"Exp3 microbatch P99 {micro['p99_ms']:.2f} ms sla={targets['serving_p99']}", flush=True)
    print(f"Wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
