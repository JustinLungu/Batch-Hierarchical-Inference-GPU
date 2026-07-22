import json
from datetime import datetime, timezone
import pandas as pd

from constants import CONFIG_FILE, DEFAULT_CONFIG_FILE
from thesis_models import THESIS_CONFIG_FILE, THESIS_REPRODUCTION_FILE


class ThesisReportWriter:
    def __init__(self, context):
        self.context = context

    def write_summary_md(self, summary: pd.DataFrame) -> None:
        lines = [
            f"Run: thesis_reproduction_{self.context.device}",
            f"Configurations: {len(summary)}",
            f"Configuration IDs: {self.context.config_id_label()}",
            f"Sample limit: {self.context.sample_limit}",
            f"Dataset: {self.context.thesis_base['SAMPLE_PATH']}",
            f"SML: {self.context.thesis_base['SML_ARCH']}",
            f"LML: {self.context.thesis_base['LML_ARCH']}",
            "",
            "| Config | Decision | Strategy | Controller Batch | Rows | Offloaded | Accuracy | Throughput | Latency Median | LML Mean |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]

        for _, row in summary.iterrows():
            offloaded = "n/a"
            if pd.notna(row["offloaded"]):
                offloaded = str(int(row["offloaded"]))
            lines.append(
                "| "
                f"{row['thesis_config']} | "
                f"{row['decision_method']} | "
                f"{row['offloading_strategy']} | "
                f"{int(row['controller_batch_size'])} | "
                f"{int(row['rows'])} | "
                f"{offloaded} | "
                f"{self.context.metrics.format_percent(row['accuracy'])} | "
                f"{self.context.metrics.format_float(row['throughput_samples_s'])} | "
                f"{self.context.metrics.format_seconds(row['total_latency_median_s'])} | "
                f"{self.context.metrics.format_seconds(row['lml_inference_mean_s'])} |"
            )

        lines.extend(
            [
                "",
                "Detailed CSVs and thesis-style plots are written beside this summary.",
            ]
        )

        self.context.summary_md.write_text("\n".join(lines) + "\n")

    def write_metadata(self, summary: pd.DataFrame) -> None:
        finished_at = datetime.now(timezone.utc)
        metadata = {
            "run_name": f"thesis_reproduction_{self.context.device}",
            "mode": "thesis_reproduction_public_ip",
            "device": self.context.device,
            "started_at_utc": self.context.started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "duration_s": (finished_at - self.context.started_at).total_seconds(),
            "default_config_file": str(DEFAULT_CONFIG_FILE),
            "experiment_config_file": str(CONFIG_FILE),
            "thesis_reproduction_file": str(THESIS_REPRODUCTION_FILE),
            "thesis_config_file": str(THESIS_CONFIG_FILE),
            "thesis_base": self.context.thesis_base,
            "sample_limit": self.context.sample_limit,
            "config_ids": [config.config_id for config in self.context.configurations],
            "configs": [
                {
                    "config_id": config.config_id,
                    "decision_method": config.decision_method,
                    "offloading_strategy": config.offloading_strategy,
                    "controller_batch_size": config.controller_batch_size,
                    "batch_size": config.batch_size,
                    "fixed_threshold_value": float(config.fixed_threshold_value),
                    "description": config.description,
                }
                for config in self.context.configurations
            ],
            "analysis_folder": str(self.context.output_dir),
            "result_count": int(len(summary)),
        }
        self.context.metadata_json.write_text(json.dumps(metadata, indent=2) + "\n")

