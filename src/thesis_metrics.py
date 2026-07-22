import pandas as pd

from thesis_models import ThesisConfiguration


class ThesisMetrics:
    def latency_breakdown_row(
        self, config: ThesisConfiguration, timing: pd.DataFrame
    ) -> dict:
        offloaded = self.offloaded_mask(timing)
        offload_path_mask = offloaded if offloaded.any() else None
        step_1 = self.duration_sum_mean(
            timing, ["sml_inference_s", "offload_decision_s"]
        )
        step_2 = self.duration_sum_mean(
            timing, ["edge_buffer_wait_s"], mask=offload_path_mask
        )
        step_3 = self.duration_sum_mean(
            timing, ["edge_to_server_network_s"], mask=offload_path_mask
        )
        step_4 = self.duration_sum_mean(
            timing,
            ["server_queue_or_preprocess_s", "lml_inference_s", "server_postprocess_s"],
            mask=offload_path_mask,
        )
        step_5 = self.duration_sum_mean(
            timing, ["server_to_edge_network_s"], mask=offload_path_mask
        )
        step_6 = self.duration_sum_mean(
            timing, ["edge_receive_to_saved_s"], mask=offload_path_mask
        )
        total = step_1 + step_2 + step_3 + step_4 + step_5 + step_6
        tracked = self.numeric_mean(timing, "total_tracked_latency_s")

        return {
            "config": config.config_id,
            "latency_breakdown_mode": "thesis_offloaded_path",
            "decision_method": config.decision_method,
            "offloading_strategy": config.offloading_strategy,
            "controller_batch_size": config.controller_batch_size,
            "step_1_ed_processing_s": step_1,
            "step_2_ed_offload_buffer_s": step_2,
            "step_3_ed_to_es_communication_s": step_3,
            "step_4_es_processing_s": step_4,
            "step_5_es_to_ed_communication_s": step_5,
            "step_6_ed_result_saving_s": step_6,
            "latency_breakdown_total_s": total,
            "tracked_latency_mean_s": tracked,
            "tracked_latency_median_s": self.numeric_median(
                timing, "total_tracked_latency_s"
            ),
        }

    def accuracy_metrics(self, timing: pd.DataFrame) -> dict:
        if timing.empty or "True Class" not in timing.columns:
            return {
                "accuracy": None,
                "sml_accuracy": None,
                "lml_accuracy_offloaded": None,
                "correct": None,
            }

        true_class = pd.to_numeric(timing["True Class"], errors="coerce")
        sml_prediction = pd.to_numeric(timing.get("SML Prediction"), errors="coerce")
        lml_prediction = pd.to_numeric(timing.get("LML Prediction"), errors="coerce")
        offloaded = self.offloaded_mask(timing)
        final_prediction = sml_prediction.copy()
        final_prediction.loc[offloaded] = lml_prediction.loc[offloaded]

        valid_final = true_class.notna() & final_prediction.notna()
        valid_sml = true_class.notna() & sml_prediction.notna()
        valid_lml = true_class.notna() & lml_prediction.notna() & offloaded

        correct = (final_prediction[valid_final] == true_class[valid_final]).sum()
        sml_correct = (sml_prediction[valid_sml] == true_class[valid_sml]).sum()
        lml_correct = (lml_prediction[valid_lml] == true_class[valid_lml]).sum()

        return {
            "accuracy": float(correct / valid_final.sum()) if valid_final.any() else None,
            "sml_accuracy": float(sml_correct / valid_sml.sum()) if valid_sml.any() else None,
            "lml_accuracy_offloaded": (
                float(lml_correct / valid_lml.sum()) if valid_lml.any() else None
            ),
            "sml_accuracy_not_offloaded": self.group_accuracy(
                true_class, sml_prediction, ~offloaded
            ),
            "correct": int(correct),
        }

    @staticmethod
    def group_accuracy(
        true_class: pd.Series, prediction: pd.Series, mask: pd.Series
    ) -> float | None:
        valid = true_class.notna() & prediction.notna() & mask
        if not valid.any():
            return None
        return float((prediction[valid] == true_class[valid]).mean())

    @staticmethod
    def offloaded_mask(timing: pd.DataFrame) -> pd.Series:
        if "Offloaded" in timing.columns:
            return timing["Offloaded"].astype(str).str.lower().eq("true")
        if "LML Prediction" in timing.columns:
            return pd.to_numeric(timing["LML Prediction"], errors="coerce").notna()
        if "lml_inference_s" in timing.columns:
            return pd.to_numeric(timing["lml_inference_s"], errors="coerce").notna()
        return pd.Series([False] * len(timing), index=timing.index)

    def communication_efficiency_row(
        self,
        config: ThesisConfiguration,
        timing: pd.DataFrame,
        summary_row: dict,
    ) -> dict:
        rows = len(timing)
        offloaded = int(self.offloaded_mask(timing).sum())
        transmissions = int(summary_row.get("edge_server_batches_observed") or 0)
        average_offload_batch = offloaded / transmissions if transmissions else 0.0
        offload_ratio = offloaded / rows if rows else 0.0

        return {
            "config": config.config_id,
            "decision_method": config.decision_method,
            "offloading_strategy": config.offloading_strategy,
            "controller_batch_size": config.controller_batch_size,
            "rows": rows,
            "offloaded_samples": offloaded,
            "offload_ratio": offload_ratio,
            "offload_transmissions": transmissions,
            "average_offload_batch_size": average_offload_batch,
            "transmission_reduction_vs_individual_percent": (
                100.0 * (1.0 - transmissions / offloaded) if offloaded else 0.0
            ),
        }

    def summary_communication_metrics(
        self, timing: pd.DataFrame, summary_row: dict
    ) -> dict:
        rows = len(timing)
        offloaded = int(self.offloaded_mask(timing).sum())
        transmissions = int(summary_row.get("edge_server_batches_observed") or 0)
        return {
            "offload_ratio": offloaded / rows if rows else 0.0,
            "offload_transmissions": transmissions,
            "average_offload_batch_size": (
                offloaded / transmissions if transmissions else 0.0
            ),
        }

    def threshold_trajectory_rows(
        self, config: ThesisConfiguration, timing: pd.DataFrame
    ) -> list[dict]:
        if config.decision_method != "adaptive_threshold":
            return []

        rows = []
        offloaded = self.offloaded_mask(timing)
        for sample_index, (_, row) in enumerate(timing.iterrows(), start=1):
            rows.append(
                {
                    "config": config.config_id,
                    "sample_index": sample_index,
                    "filename": row.get("Filename"),
                    "sml_confidence": self.optional_float(row.get("SML Confidence")),
                    "offloaded": bool(offloaded.loc[row.name]),
                    "decision_threshold": self.optional_float(
                        row.get("Decision Threshold")
                    ),
                    "adaptive_threshold_after_update": self.optional_float(
                        row.get("Adaptive Threshold After Update")
                    ),
                    "threshold_update_duration_s": self.optional_float(
                        row.get("ts_threshold_updated")
                    ),
                }
            )
        return rows

    def offloading_distribution_row(
        self, config: ThesisConfiguration, timing: pd.DataFrame
    ) -> dict:
        true_class = pd.to_numeric(timing.get("True Class"), errors="coerce")
        sml_prediction = pd.to_numeric(timing.get("SML Prediction"), errors="coerce")
        offloaded = self.offloaded_mask(timing)
        sml_correct = true_class.notna() & sml_prediction.notna() & (
            sml_prediction == true_class
        )
        sml_wrong = true_class.notna() & sml_prediction.notna() & (
            sml_prediction != true_class
        )
        total = max(len(timing), 1)

        true_positive = int((sml_wrong & offloaded).sum())
        true_negative = int((sml_correct & ~offloaded).sum())
        false_positive = int((sml_correct & offloaded).sum())
        false_negative = int((sml_wrong & ~offloaded).sum())

        return {
            "config": config.config_id,
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_positive_percent": 100.0 * true_positive / total,
            "true_negative_percent": 100.0 * true_negative / total,
            "false_positive_percent": 100.0 * false_positive / total,
            "false_negative_percent": 100.0 * false_negative / total,
        }

    def per_sample_latency_row(
        self, config: ThesisConfiguration, timing: pd.DataFrame
    ) -> dict:
        offloaded = self.offloaded_mask(timing)
        latency = pd.to_numeric(timing["total_tracked_latency_s"], errors="coerce")
        return {
            "config": config.config_id,
            "system_combined_s": self.series_mean(latency),
            "offloaded_samples_s": self.series_mean(latency[offloaded]),
            "not_offloaded_samples_s": self.series_mean(latency[~offloaded]),
        }

    def add_communication_baselines(self, communication: pd.DataFrame) -> pd.DataFrame:
        output = communication.copy()
        config_004 = output.loc[output["config"] == "004", "offload_transmissions"]
        baseline = int(config_004.iloc[0]) if not config_004.empty else None
        if baseline and baseline > 0:
            output["transmission_reduction_vs_config_004_percent"] = output[
                "offload_transmissions"
            ].map(lambda value: 100.0 * (1.0 - float(value) / baseline))
        else:
            output["transmission_reduction_vs_config_004_percent"] = pd.NA
        return output

    @staticmethod
    def duration_sum_mean(
        timing: pd.DataFrame, columns: list[str], mask: pd.Series | None = None
    ) -> float:
        if timing.empty:
            return 0.0
        selected = timing
        if mask is not None:
            selected = timing[mask]
        if selected.empty:
            return 0.0
        total = pd.Series([0.0] * len(selected), index=selected.index)
        for column in columns:
            if column in selected.columns:
                total += pd.to_numeric(selected[column], errors="coerce").fillna(0.0)
        return float(total.mean())

    @staticmethod
    def numeric_mean(data: pd.DataFrame, column: str) -> float | None:
        if column not in data.columns:
            return None
        values = pd.to_numeric(data[column], errors="coerce").dropna()
        if values.empty:
            return None
        return float(values.mean())

    @staticmethod
    def numeric_median(data: pd.DataFrame, column: str) -> float | None:
        if column not in data.columns:
            return None
        values = pd.to_numeric(data[column], errors="coerce").dropna()
        if values.empty:
            return None
        return float(values.median())

    @staticmethod
    def series_mean(values: pd.Series) -> float | None:
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        if numeric.empty:
            return None
        return float(numeric.mean())

    @staticmethod
    def count_true(data: pd.DataFrame, column: str) -> int | None:
        if column not in data.columns:
            return None
        return int(data[column].astype(str).str.lower().eq("true").sum())

    @staticmethod
    def optional_float(value) -> float | None:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            return None
        return float(numeric)

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

    @staticmethod
    def format_percent(value) -> str:
        if pd.isna(value):
            return "n/a"
        return f"{float(value) * 100.0:.2f}%"

