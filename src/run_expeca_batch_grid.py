import json
import os
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import pandas as pd

from constants import CONFIG_FILE, REPO_ROOT, RUN_METADATA_FILENAME, SUMMARY_FILENAME
from run_expeca_public_ip_test import ExpecaPublicIpRunner
from utils import load_env_file, require_config


def main() -> int:
    runner = ExpecaBatchGridRunner()
    return runner.run()


class ExpecaBatchGridRunner:
    def __init__(self):
        os.chdir(REPO_ROOT)
        self.config = load_env_file(CONFIG_FILE)
        self.device = require_config(self.config, "DEVICE")
        self.results_dir = Path(require_config(self.config, "RESULTS_DIR"))
        self.started_at = datetime.now(timezone.utc)
        self.batch_sizes = self.parse_int_list("BATCH_SIZE_GRID")
        self.controller_batch_sizes = self.parse_controller_batch_sizes()
        self.pair_mode = self.config_value("BATCH_GRID_PAIR_MODE", "product").lower()
        self.analysis_dir = (
            self.results_dir / f"analysis_expeca_public_ip_{self.device}_grid"
        )
        self.summary_csv = self.analysis_dir / "summary.csv"
        self.summary_md = self.analysis_dir / SUMMARY_FILENAME
        self.metadata_json = self.analysis_dir / RUN_METADATA_FILENAME

    def run(self) -> int:
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        combinations = self.batch_combinations()
        print(
            f"Running ExPECA public-IP batch grid for {len(combinations)} combination(s)."
        )

        aggregate_rows = []
        for index, (batch_size, controller_batch_size) in enumerate(combinations, start=1):
            print(
                "\n"
                f"[{index}/{len(combinations)}] "
                f"BATCH_SIZE={batch_size}, "
                f"CONTROLLER_BATCH_SIZE={controller_batch_size}"
            )
            run = ExpecaPublicIpRunner(
                config_overrides={
                    "BATCH_SIZE": str(batch_size),
                    "CONTROLLER_BATCH_SIZE": str(controller_batch_size),
                },
                analysis_suffix=(
                    f"serverbatch{batch_size}_controllerbatch{controller_batch_size}"
                ),
            )
            try:
                run.start_services()
                run.send_config()
                run.send_samples()
                run.download_remote_results()
                timing = run.post_process_results()
            finally:
                run.stop_services()
            aggregate_rows.append(run.aggregate_metrics(timing))

        summary = pd.DataFrame(aggregate_rows)
        summary.to_csv(self.summary_csv, index=False)
        self.write_summary_md(summary)
        self.write_metadata(summary, combinations)

        print(f"\nWrote grid analysis folder: {self.analysis_dir}")
        print(f"Wrote aggregate CSV: {self.summary_csv}")
        print(f"Wrote aggregate summary: {self.summary_md}")
        print(f"Wrote aggregate metadata: {self.metadata_json}")
        return 0

    def parse_controller_batch_sizes(self) -> list[int]:
        raw_grid = self.config_value("CONTROLLER_BATCH_SIZE_GRID", "")
        if raw_grid:
            return self.parse_int_list("CONTROLLER_BATCH_SIZE_GRID")
        return [int(require_config(self.config, "CONTROLLER_BATCH_SIZE"))]

    def parse_int_list(self, key: str) -> list[int]:
        raw_value = self.config_value(key, "")
        if not raw_value:
            raise RuntimeError(f"Missing required grid config value: {key}")
        values = []
        for item in raw_value.split(","):
            item = item.strip()
            if not item:
                continue
            value = int(item)
            if value <= 0:
                raise ValueError(f"{key} must contain only positive integers.")
            values.append(value)
        if not values:
            raise RuntimeError(f"{key} must contain at least one positive integer.")
        return values

    def batch_combinations(self) -> list[tuple[int, int]]:
        if self.pair_mode == "product":
            return list(product(self.batch_sizes, self.controller_batch_sizes))
        if self.pair_mode == "zip":
            if len(self.batch_sizes) != len(self.controller_batch_sizes):
                raise ValueError(
                    "BATCH_GRID_PAIR_MODE=zip requires BATCH_SIZE_GRID and "
                    "CONTROLLER_BATCH_SIZE_GRID to have the same length."
                )
            return list(zip(self.batch_sizes, self.controller_batch_sizes))
        raise ValueError("BATCH_GRID_PAIR_MODE must be either 'product' or 'zip'.")

    def write_summary_md(self, summary: pd.DataFrame) -> None:
        lines = [
            f"Run: expeca_public_ip_{self.device}_grid",
            f"Combinations: {len(summary)}",
            f"Batch sizes: {self.batch_sizes}",
            f"Controller batch sizes: {self.controller_batch_sizes}",
            f"Pair mode: {self.pair_mode}",
            "",
            "| Batch | Controller Batch | Rows | Throughput | Latency Median | LML Mean | Offload Roundtrip |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]

        for _, row in summary.iterrows():
            lines.append(
                "| "
                f"{int(row['batch_size'])} | "
                f"{int(row['controller_batch_size'])} | "
                f"{int(row['rows'])} | "
                f"{self.format_float(row['throughput_samples_s'])} | "
                f"{self.format_seconds(row['total_latency_median_s'])} | "
                f"{self.format_seconds(row['lml_inference_mean_s'])} | "
                f"{self.format_seconds(row['offload_roundtrip_mean_s'])} |"
            )

        self.summary_md.write_text("\n".join(lines) + "\n")

    def write_metadata(
        self, summary: pd.DataFrame, combinations: list[tuple[int, int]]
    ) -> None:
        finished_at = datetime.now(timezone.utc)
        metadata = {
            "run_name": f"expeca_public_ip_{self.device}_grid",
            "mode": "expeca_public_ip_grid",
            "device": self.device,
            "started_at_utc": self.started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "duration_s": (finished_at - self.started_at).total_seconds(),
            "config_file": str(CONFIG_FILE),
            "batch_sizes": self.batch_sizes,
            "controller_batch_sizes": self.controller_batch_sizes,
            "pair_mode": self.pair_mode,
            "combinations": combinations,
            "outputs": {
                "analysis_folder": str(self.analysis_dir),
                "summary_csv": str(self.summary_csv),
                "summary_md": str(self.summary_md),
                "metadata_json": str(self.metadata_json),
            },
            "result_count": int(len(summary)),
        }
        self.metadata_json.write_text(json.dumps(metadata, indent=2) + "\n")

    def config_value(self, key: str, default: str) -> str:
        return os.environ.get(key, self.config.get(key, default)).strip()

    @staticmethod
    def format_seconds(value) -> str:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):.4f}s"

    @staticmethod
    def format_float(value) -> str:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
