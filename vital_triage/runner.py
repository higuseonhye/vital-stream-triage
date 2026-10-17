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
                row.get("label", row["name"]),
                row["name"],
                _num(zero_by_name[row["name"]]["standardized_mse"], 3),
                _num(forward_by_name[row["name"]]["standardized_mse"], 3),
                _num(row["standardized_mse"], 3),
                _num(row["decay_reflection"], 3),
            ]
        )
    feature_rows.append(
        [
            "11-lab mean",
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
        "Synthetic hourly ward stream. This file is one CPU run of the repository, not a clinical-device measurement.",
        "",
        f"- {report['cohort']['patients']} patients, {report['cohort']['hours']}-hour grid, seed {report['seed']}",
        f"- Alarms are counted on {report['cohort']['test_patients']} held-out patients. Imputation error uses every lab gap in the cohort.",
        f"- Lab missingness {100.0 * report['imputation']['lab_missing_rate']:.1f}%. Scoring uses gaps after the first draw only.",
        "",
        "## 1. Imputation",
        "",
        "Standardized MSE is the mean squared error against the latent trajectory, divided by that analyte's population variance. Decay reflection is the Pearson correlation between hours since the last draw and distance from that draw. Forward-fill keeps the distance at zero, so the correlation is zero.",
        "",
        _md_table(
            ["Analyte", "key", "zero", "forward", "time-decay", "decay correlation"],
            feature_rows,
        ),
        "",
        f"Order target (time-decay < forward < zero): {'met' if targets['imputation_order'] else 'not met'}.",
        f"Decay-correlation target (time-decay > forward): {'met' if targets['decay_reflection'] else 'not met'}.",
        (
            f"Lactate standardized MSE is zero {_num(zero_by_name['lactate']['standardized_mse'], 3)}, "
            f"forward {_num(forward_by_name['lactate']['standardized_mse'], 3)}, "
            f"time-decay {_num(next(row['standardized_mse'] for row in decay['per_feature'] if row['name']=='lactate'), 3)}."
        ),
        "",
        "## 2. Alarm policy",
        "",
        "The baseline fires when the deteriorate-class softmax exceeds 0.7. The fusion rule also requires 3-class entropy below 0.3 nats. That entropy is the entropy of the same softmax, so values under 0.3 nats occur only when the largest probability is about 0.93 or higher.",
        "",
        _md_table(
            ["Class", "Hours", "Mean score", "Score P90", "Mean entropy", "Baseline alarms", "Fusion alarms"],
            class_rows,
        ),
        "",
        _md_table(
            ["Rule", "TP", "FP", "FN", "Precision", "Recall"],
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
        f"False-alarm reduction {_pct(alarms['false_alarm_reduction'])} (target at least 35%): {'met' if targets['false_alarm_reduction'] else 'not met'}.",
        f"Precision ratio {_num(alarms['precision_defense'], 3)} (fusion precision / baseline precision, target at least 1): {'met' if targets['precision_defense'] else 'not met'}.",
        f"Lowest deteriorate score among fusion alarms {_num(alarms['min_score_among_fusion_alarms'], 3)}.",
        "Recall moves the other way. The false-negative column is the cost of the tighter rule.",
        "",
        "## 3. Simultaneous 1,000-bed arrival",
        "",
        "Every bed window arrives at one timestamp on a single CPU-thread queue. Latency is completion time minus that shared arrival. Sequential serving is the same queue with max batch 1.",
        "",
        _md_table(
            ["Mode", "max batch", "P50 ms", "P95 ms", "P99 ms", "beds/s", f"P99 < {serving['microbatch']['sla_ms']:.0f} ms"],
            serve_rows,
        ),
        "",
        f"Micro-batch P99 {_num(serving['microbatch']['p99_ms'], 2)} ms: {'met' if targets['serving_p99'] else 'not met'}.",
        f"Throughput {serving['microbatch']['throughput_beds_per_sec']:.1f} versus sequential {serving['sequential']['throughput_beds_per_sec']:.1f} beds/s.",
        "",
        "## Outside this run",
        "",
        "- Not a clinical-device claim, a sensitivity filing, or a fit assessment for a hospital alarm policy.",
        "- The entropy gate is one softmax entropy, not epistemic uncertainty from MC dropout or an ensemble.",
        "- Not a 1 Hz monitor waveform. The grid is hourly, as in a ward chart.",
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
