"""Local board for the synthetic vital stream. Serves on 127.0.0.1:8503."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from vital_triage.generator import FEATURES, feature_index
from vital_triage.triage_engine import SCORE_THRESHOLD, ENTROPY_THRESHOLD, alarm_masks, evaluate_alarms

ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "results" / "benchmark_report.json"
DASHBOARD_PATH = ROOT / "results" / "dashboard.npz"

st.set_page_config(page_title="Vital Stream Triage", layout="wide")
plt.rcParams["axes.unicode_minus"] = False


@st.cache_resource
def load_board():
    if not REPORT_PATH.exists() or not DASHBOARD_PATH.exists():
        return None
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    with np.load(DASHBOARD_PATH) as packed:
        board = {key: packed[key] for key in packed.files}
    return report, board


def _metric_value(report: dict, key: str) -> str:
    alarms = report["alert_fatigue"]
    decay = next(row for row in report["imputation"]["methods"] if row["method"] == "time_decay")
    micro = next(row for row in report["serving"]["rows"] if row["mode"] == "microbatch")
    if key == "mse":
        return f"{decay['standardized_mse']:.3f}"
    if key == "fa":
        reduction = alarms["false_alarm_reduction"]
        return "n/a" if reduction is None else f"{100.0 * reduction:.1f}%"
    if key == "p99":
        return f"{micro['p99_ms']:.1f} ms"
    return ""


def _plot_series(board: dict, patient: int, key: str) -> plt.Figure:
    index = feature_index(key)
    feature = FEATURES[index]
    hours = np.arange(board["values"].shape[1])
    observed = board["observed"][patient, :, index]
    figure, axis = plt.subplots(figsize=(8.2, 3.4))
    axis.plot(hours, board["values"][patient, :, index], color="#111827", linewidth=1.6, label="Latent")
    axis.plot(hours, board["imp_zero"][patient, :, index], color="#9ca3af", linewidth=1.1, label="Zero-fill")
    axis.plot(hours, board["imp_forward"][patient, :, index], color="#d97706", linewidth=1.2, label="Forward-fill")
    axis.plot(hours, board["imp_time_decay"][patient, :, index], color="#0f766e", linewidth=1.6, label="Time-decay")
    charted = np.isfinite(observed)
    axis.scatter(hours[charted], observed[charted], color="#111827", s=28, zorder=4, label="Charted")
    axis.set_ylabel(f"{feature.label} ({feature.unit})")
    axis.set_xlabel("Hour")
    axis.legend(loc="upper right", fontsize=8, ncol=2)
    figure.tight_layout()
    return figure


def _plot_alarms(board: dict, patient: int, score_threshold: float, entropy_threshold: float) -> plt.Figure:
    probs = board["probs"][patient]
    score, entropy, baseline, fusion = alarm_masks(probs[None, ...], score_threshold, entropy_threshold)
    hours = board["hour_index"]
    figure, axis = plt.subplots(figsize=(8.2, 3.4))
    axis.plot(hours, score[0], color="#1d4ed8", linewidth=1.6, label="Deterioration score")
    axis.plot(hours, entropy[0], color="#7c3aed", linewidth=1.2, label="Entropy (nat)")
    axis.axhline(score_threshold, color="#1d4ed8", linestyle="--", linewidth=0.8)
    axis.axhline(entropy_threshold, color="#7c3aed", linestyle="--", linewidth=0.8)
    axis.scatter(hours[baseline[0]], np.full(int(baseline[0].sum()), -0.05), marker="|", s=80, color="#d97706", label="Baseline alarm")
    axis.scatter(hours[fusion[0]], np.full(int(fusion[0].sum()), -0.12), marker="|", s=80, color="#be123c", label="Fusion alarm")
    event = int(board["event_time"][patient])
    if event >= 0:
        axis.axvspan(event - 6, event, color="#fecdd3", alpha=0.45, label="Deterioration window")
    axis.set_ylim(-0.2, 1.05)
    axis.set_xlabel("Hour")
    axis.legend(loc="upper right", fontsize=8, ncol=2)
    figure.tight_layout()
    return figure


def main() -> None:
    st.title("Vital Stream Triage")
    st.caption("Synthetic hourly ward stream. Not for clinical use. The board replays the saved benchmark cohort.")
    loaded = load_board()
    if loaded is None:
        st.warning("No result files. From the repository root, run `python -m vital_triage.runner --synthetic` first.")
        return
    report, board = loaded
    targets = report["targets"]
    left, middle, right = st.columns(3)
    left.metric("Time-decay standardized MSE", _metric_value(report, "mse"))
    middle.metric(
        "False-alarm reduction",
        _metric_value(report, "fa"),
        f"35% target {'met' if targets['false_alarm_reduction'] else 'not met'}",
    )
    right.metric(
        "Micro-batch P99",
        _metric_value(report, "p99"),
        f"50 ms target {'met' if targets['serving_p99'] else 'not met'}",
    )
    st.caption(
        f"Imputation order {'met' if targets['imputation_order'] else 'not met'} · "
        f"precision ratio {'met' if targets['precision_defense'] else 'not met'} · "
        f"{report['cohort']['test_patients']} held-out patients · seed {report['seed']}"
    )

    n_patients = int(board["probs"].shape[0])
    labels = [f"Bed {index + 1:03d}" for index in range(n_patients)]
    event_patients = [index for index in range(n_patients) if int(board["event_time"][index]) >= 0]
    default_index = event_patients[0] if event_patients else 0
    patient = st.selectbox(
        "Held-out bed",
        options=list(range(n_patients)),
        index=default_index,
        format_func=lambda index: labels[index],
    )
    kind = {
        0: "none",
        1: "heart-rate and respiratory spike",
        2: "deterioration-like transient",
    }[int(board["transient_kind"][patient])]
    event = int(board["event_time"][patient])
    event_text = "none" if event < 0 else f"hour {event}"
    st.caption(
        f"Age {board['age'][patient]:.0f} · GCS {board['gcs'][patient]:.0f} · "
        f"deterioration {event_text} · transient {kind}"
    )

    feature_labels = {feature.key: f"{feature.label} ({feature.key})" for feature in FEATURES}
    chosen = st.selectbox(
        "Imputation series",
        options=[feature.key for feature in FEATURES],
        index=feature_index("lactate"),
        format_func=lambda key: feature_labels[key],
    )
    score_threshold, entropy_threshold = st.columns(2)
    with score_threshold:
        score_cut = st.slider("Deterioration score cut", 0.50, 0.99, SCORE_THRESHOLD, 0.01)
    with entropy_threshold:
        entropy_cut = st.slider("Entropy cut (nat)", 0.05, 1.00, ENTROPY_THRESHOLD, 0.01)

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.pyplot(_plot_series(board, patient, chosen), clear_figure=True)
    with chart_right:
        st.pyplot(_plot_alarms(board, patient, score_cut, entropy_cut), clear_figure=True)

    live = evaluate_alarms(
        board["probs"],
        board["labels"],
        np.arange(n_patients),
        score_threshold=score_cut,
        entropy_threshold=entropy_cut,
    )
    official = score_cut == SCORE_THRESHOLD and abs(entropy_cut - ENTROPY_THRESHOLD) < 1e-9
    st.subheader("Held-out alarms" + (" · recorded cuts" if official else " · slider cuts"))
    one, two, three, four = st.columns(4)
    one.metric("Baseline false positives", live["baseline"]["fp"])
    two.metric("Fusion false positives", live["fusion"]["fp"])
    three.metric("Fusion precision", "n/a" if live["fusion"]["precision"] is None else f"{live['fusion']['precision']:.3f}")
    four.metric("Fusion recall", "n/a" if live["fusion"]["recall"] is None else f"{live['fusion']['recall']:.3f}")

    decay = next(row for row in report["imputation"]["methods"] if row["method"] == "time_decay")
    forward = {row["name"]: row for row in next(row for row in report["imputation"]["methods"] if row["method"] == "forward")["per_feature"]}
    zero = {row["name"]: row for row in next(row for row in report["imputation"]["methods"] if row["method"] == "zero")["per_feature"]}
    labels_by_key = {feature.key: feature.label for feature in FEATURES}
    st.subheader("Lab standardized MSE")
    st.table(
        [
            {
                "Analyte": labels_by_key.get(row["name"], row["name"]),
                "zero": round(zero[row["name"]]["standardized_mse"], 3),
                "forward": round(forward[row["name"]]["standardized_mse"], 3),
                "time-decay": round(row["standardized_mse"], 3),
            }
            for row in decay["per_feature"]
        ],
        hide_index=True,
        width="stretch",
    )


main()
