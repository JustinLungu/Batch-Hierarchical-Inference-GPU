import argparse
from pathlib import Path

import pandas as pd

from offload_batch_analysis.analyzer import OffloadBatchAnalyzer, OffloadBatchContext
from offload_batch_analysis.plots import OffloadBatchPlotter


ANALYZED_CONFIGS = ("005", "006", "007")
DEFAULT_RESULTS_DIR = Path("results/thesis_reproduction_gpu")
THESIS_CONFIG_PATH = Path("config/thesis_configs.csv")


def load_contexts() -> dict[str, OffloadBatchContext]:
    configs = pd.read_csv(THESIS_CONFIG_PATH, dtype={"config_id": str})
    configs["config_id"] = configs["config_id"].str.zfill(3)
    selected = configs[configs["config_id"].isin(ANALYZED_CONFIGS)]
    found = set(selected["config_id"])
    missing = sorted(set(ANALYZED_CONFIGS).difference(found))
    if missing:
        raise RuntimeError(
            f"Missing thesis configuration(s) in {THESIS_CONFIG_PATH}: "
            + ", ".join(missing)
        )
    return {
        row.config_id: OffloadBatchContext(
            config_id=row.config_id,
            controller_batch_size=int(row.controller_batch_size),
        )
        for row in selected.itertuples(index=False)
    }


def analyze(results_dir: Path) -> Path:
    analyzer = OffloadBatchAnalyzer()
    contexts = load_contexts()
    measurements = []
    for config_id in ANALYZED_CONFIGS:
        raw_path = (
            results_dir / f"config_{config_id}" / "raw_edge_device_results.csv"
        )
        if not raw_path.is_file():
            raise FileNotFoundError(f"Missing raw results: {raw_path}")
        raw_results = pd.read_csv(raw_path)
        measurements.append(
            analyzer.extract_batch_measurements(contexts[config_id], raw_results)
        )

    all_measurements = pd.concat(measurements, ignore_index=True)
    grouped_summary = analyzer.summarize_by_batch_size(all_measurements)
    trends = analyzer.calculate_config_trends(all_measurements)

    output_dir = results_dir / "offload_batch_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    measurements_path = output_dir / "batch_measurements.csv"
    grouped_path = output_dir / "batch_size_summary.csv"
    trends_path = output_dir / "config_trends.csv"
    all_measurements.to_csv(measurements_path, index=False)
    grouped_summary.to_csv(grouped_path, index=False)
    trends.to_csv(trends_path, index=False)
    plots = OffloadBatchPlotter(output_dir / "plots").write_plots(
        all_measurements,
        grouped_summary,
    )

    print(f"Analyzed offload batches from: {results_dir}")
    print(f"Wrote request measurements: {measurements_path}")
    print(f"Wrote grouped statistics: {grouped_path}")
    print(f"Wrote trend calculations: {trends_path}")
    print(f"Wrote {len(plots)} plot(s): {output_dir / 'plots'}")
    for row in trends.itertuples(index=False):
        print(
            f"Config {row.config}: {row.request_count} server requests, "
            f"batch size {row.actual_batch_size_min}-{row.actual_batch_size_max}, "
            f"response Spearman r={row.response_time_spearman_r:.3f}"
        )
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze server response time against actual offloaded batch size "
            "for thesis configurations 005-007."
        )
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Thesis result folder (default: {DEFAULT_RESULTS_DIR}).",
    )
    args = parser.parse_args()
    analyze(args.results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
