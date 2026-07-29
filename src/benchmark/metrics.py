import pandas as pd

from .config import BenchmarkConfiguration
from .utils import (
    duration_sum_mean,
    group_accuracy,
    numeric_mean,
    numeric_median,
    offloaded_mask,
    optional_float,
    series_mean,
)



def latency_breakdown_row(
    config: BenchmarkConfiguration, timing: pd.DataFrame
) -> dict:
    offloaded = offloaded_mask(timing)
    offload_path_mask = offloaded if offloaded.any() else None
    step_1 = duration_sum_mean(
        timing, ["sml_inference_s", "offload_decision_s"]
    )
    step_2 = duration_sum_mean(
        timing, ["edge_buffer_wait_s"], mask=offload_path_mask
    )
    step_3 = duration_sum_mean(
        timing, ["edge_to_server_network_s"], mask=offload_path_mask
    )
    step_4 = duration_sum_mean(
        timing,
        ["server_queue_or_preprocess_s", "lml_inference_s", "server_postprocess_s"],
        mask=offload_path_mask,
    )
    step_5 = duration_sum_mean(
        timing, ["server_to_edge_network_s"], mask=offload_path_mask
    )
    step_6 = duration_sum_mean(
        timing, ["edge_receive_to_saved_s"], mask=offload_path_mask
    )
    total = step_1 + step_2 + step_3 + step_4 + step_5 + step_6
    tracked = numeric_mean(timing, "total_tracked_latency_s")

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
        "tracked_latency_median_s": numeric_median(
            timing, "total_tracked_latency_s"
        ),
    }


def accuracy_metrics(timing: pd.DataFrame) -> dict:
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
    offloaded = offloaded_mask(timing)
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
        "sml_accuracy_not_offloaded": group_accuracy(
            true_class, sml_prediction, ~offloaded
        ),
        "correct": int(correct),
    }


def summary_communication_metrics(
    timing: pd.DataFrame, summary_row: dict
) -> dict:
    rows = len(timing)
    offloaded = int(offloaded_mask(timing).sum())
    transmissions = int(summary_row.get("edge_server_batches_observed") or 0)
    return {
        "offload_ratio": offloaded / rows if rows else 0.0,
        "offload_transmissions": transmissions,
        "average_offload_batch_size": (
            offloaded / transmissions if transmissions else 0.0
        ),
    }


def threshold_trajectory_rows(
    config: BenchmarkConfiguration, timing: pd.DataFrame
) -> list[dict]:
    if config.decision_method != "adaptive_threshold":
        return []

    rows = []
    offloaded = offloaded_mask(timing)
    for sample_index, (_, row) in enumerate(timing.iterrows(), start=1):
        rows.append(
            {
                "config": config.config_id,
                "sample_index": sample_index,
                "filename": row.get("Filename"),
                "sml_confidence": optional_float(row.get("SML Confidence")),
                "offloaded": bool(offloaded.loc[row.name]),
                "decision_threshold": optional_float(
                    row.get("Decision Threshold")
                ),
                "adaptive_threshold_after_update": optional_float(
                    row.get("Adaptive Threshold After Update")
                ),
                "threshold_update_duration_s": optional_float(
                    row.get("ts_threshold_updated")
                ),
            }
        )
    return rows


def offloading_distribution_row(
    config: BenchmarkConfiguration, timing: pd.DataFrame
) -> dict:
    true_class = pd.to_numeric(timing.get("True Class"), errors="coerce")
    sml_prediction = pd.to_numeric(timing.get("SML Prediction"), errors="coerce")
    offloaded = offloaded_mask(timing)
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
    config: BenchmarkConfiguration, timing: pd.DataFrame
) -> dict:
    offloaded = offloaded_mask(timing)
    latency = pd.to_numeric(timing["total_tracked_latency_s"], errors="coerce")
    return {
        "config": config.config_id,
        "system_combined_s": series_mean(latency),
        "offloaded_samples_s": series_mean(latency[offloaded]),
        "not_offloaded_samples_s": series_mean(latency[~offloaded]),
    }

