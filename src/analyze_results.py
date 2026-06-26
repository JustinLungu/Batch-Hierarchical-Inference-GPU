from pathlib import Path

import pandas as pd


INPUT_RESULTS_CSV = Path("results/EdgeDevice_results.csv")
OUTPUT_DIR = Path("results/analysis")

TIMING_COLUMNS = [
    "ts_sml_inference_start",
    "ts_sml_inference_end",
    "ts_offload_decision_made",
    "ts_results_saved_not_offloaded",
    "ts_sample_sent_to_offloading",
    "ts_sample_sent_to_edge_server",
    "ts_sample_received_at_edge_server",
    "ts_lml_inference_start",
    "ts_lml_inference_end",
    "ts_results_sent_to_edge_device",
    "ts_results_received_from_edge_server",
    "ts_results_received_from_offloading_module",
    "ts_threshold_updated",
]


def seconds_between(df: pd.DataFrame, end: str, start: str) -> pd.Series:
    if end not in df.columns or start not in df.columns:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="Float64")
    return df[end] - df[start]


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for column in TIMING_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    derived = df.copy()

    derived["sml_inference_s"] = seconds_between(
        derived, "ts_sml_inference_end", "ts_sml_inference_start"
    )
    derived["offload_decision_s"] = seconds_between(
        derived, "ts_offload_decision_made", "ts_sml_inference_end"
    )
    derived["edge_buffer_wait_s"] = seconds_between(
        derived, "ts_sample_sent_to_edge_server", "ts_sample_sent_to_offloading"
    )
    derived["edge_to_server_network_s"] = seconds_between(
        derived, "ts_sample_received_at_edge_server", "ts_sample_sent_to_edge_server"
    )
    derived["server_queue_or_preprocess_s"] = seconds_between(
        derived, "ts_lml_inference_start", "ts_sample_received_at_edge_server"
    )
    derived["lml_inference_s"] = seconds_between(
        derived, "ts_lml_inference_end", "ts_lml_inference_start"
    )
    derived["server_postprocess_s"] = seconds_between(
        derived, "ts_results_sent_to_edge_device", "ts_lml_inference_end"
    )
    derived["server_to_edge_network_s"] = seconds_between(
        derived, "ts_results_received_from_edge_server", "ts_results_sent_to_edge_device"
    )
    derived["offload_roundtrip_s"] = seconds_between(
        derived, "ts_results_received_from_edge_server", "ts_sample_sent_to_edge_server"
    )
    derived["edge_receive_to_saved_s"] = seconds_between(
        derived,
        "ts_results_received_from_offloading_module",
        "ts_results_received_from_edge_server",
    )

    offloaded_total = seconds_between(
        derived,
        "ts_results_received_from_offloading_module",
        "ts_sml_inference_start",
    )
    local_total = seconds_between(
        derived,
        "ts_results_saved_not_offloaded",
        "ts_sml_inference_start",
    )
    derived["total_tracked_latency_s"] = offloaded_total.fillna(local_total)

    if "ts_sample_sent_to_edge_server" in derived.columns:
        batch_keys = derived["ts_sample_sent_to_edge_server"].fillna(-1)
        derived["edge_server_batch_id"] = pd.factorize(batch_keys)[0]
        derived.loc[batch_keys == -1, "edge_server_batch_id"] = pd.NA

    return derived


def median_seconds(series: pd.Series) -> str:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return "n/a"
    return f"~{clean.median():.4f}s"


def mean_seconds(series: pd.Series) -> str:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return "n/a"
    return f"~{clean.mean():.4f}s"


def build_timing_csv(derived: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "UUID",
        "Filename",
        "sml_inference_s",
        "offload_decision_s",
        "edge_buffer_wait_s",
        "edge_to_server_network_s",
        "server_queue_or_preprocess_s",
        "lml_inference_s",
        "server_postprocess_s",
        "server_to_edge_network_s",
        "offload_roundtrip_s",
        "edge_receive_to_saved_s",
        "total_tracked_latency_s",
        "edge_server_batch_id",
    ]
    available = [column for column in columns if column in derived.columns]
    timing = derived[available].copy()
    for column in timing.columns:
        if column.endswith("_s"):
            numeric = pd.to_numeric(timing[column], errors="coerce")
            timing[column] = numeric.map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
    if "edge_server_batch_id" in timing.columns:
        batch_ids = pd.to_numeric(timing["edge_server_batch_id"], errors="coerce")
        timing["edge_server_batch_id"] = batch_ids.map(
            lambda value: "" if pd.isna(value) else str(int(value))
        )
    return timing


def build_summary(derived: pd.DataFrame) -> str:
    lines = []
    lines.append(f"Rows: {len(derived)}")

    if "Offloaded" in derived.columns:
        offloaded = derived["Offloaded"].astype(str).str.lower().eq("true")
        lines.append(f"Offloaded: {offloaded.sum()} / {len(derived)}")

    if "Buffered" in derived.columns:
        buffered = derived["Buffered"].astype(str).str.lower().eq("true")
        lines.append(f"Still buffered: {buffered.sum()} / {len(derived)}")

    if "edge_server_batch_id" in derived.columns:
        batch_sizes = (
            derived.dropna(subset=["edge_server_batch_id"])
            .groupby("edge_server_batch_id")
            .size()
            .tolist()
        )
        lines.append(f"Edge-server batches observed: {len(batch_sizes)}")
        lines.append(f"Edge-server batch sizes: {batch_sizes}")

    lines.append(f"Total tracked latency median: {median_seconds(derived['total_tracked_latency_s'])}")
    lines.append(f"SML inference mean: {mean_seconds(derived['sml_inference_s'])}")
    lines.append(f"LML inference mean: {mean_seconds(derived['lml_inference_s'])}")
    lines.append(f"Offload roundtrip: {mean_seconds(derived['offload_roundtrip_s'])}")

    if "total_tracked_latency_s" in derived.columns and len(derived) > 0:
        start = pd.to_numeric(derived["ts_sml_inference_start"], errors="coerce").min()
        end = pd.to_numeric(
            derived["ts_results_received_from_offloading_module"].fillna(
                derived.get("ts_results_saved_not_offloaded")
            ),
            errors="coerce",
        ).max()
        if pd.notna(start) and pd.notna(end) and end > start:
            lines.append(f"Approx throughput: ~{len(derived) / (end - start):.2f} samples/s")

    return "\n".join(lines) + "\n"


def main() -> int:
    input_path = INPUT_RESULTS_CSV
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_results(input_path)
    derived = add_derived_columns(raw)
    summary = build_summary(derived)

    timing = build_timing_csv(derived)

    timing_path = output_dir / "timing_results.csv"
    summary_path = output_dir / "summary.md"
    timing.to_csv(timing_path, index=False)
    summary_path.write_text(summary)

    print(summary)
    print(f"Wrote timing CSV: {timing_path}")
    print(f"Wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
