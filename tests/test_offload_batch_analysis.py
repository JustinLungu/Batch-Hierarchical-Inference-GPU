import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from offload_batch_analysis import OffloadBatchAnalyzer, OffloadBatchContext


class OffloadBatchAnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = OffloadBatchAnalyzer()
        self.context = OffloadBatchContext(config_id="005", controller_batch_size=5)

    def test_collapses_sample_rows_into_server_batches(self):
        raw_results = pd.DataFrame(
            {
                "ts_sample_sent_to_edge_server": [10.0, 10.0, 10.0, 20.0, None],
                "ts_sample_received_at_edge_server": [10.1, 10.1, 10.1, 20.2, None],
                "ts_lml_inference_start": [10.2, 10.2, 10.6, 20.4, None],
                "ts_lml_inference_end": [10.6, 10.6, 11.0, 20.9, None],
                "ts_results_sent_to_edge_device": [11.1, 11.1, 11.1, 21.0, None],
            }
        )

        batches = self.analyzer.extract_batch_measurements(self.context, raw_results)

        self.assertEqual(batches["actual_server_batch_size"].tolist(), [3, 1])
        self.assertAlmostEqual(batches.loc[0, "server_response_time_s"], 1.0)
        self.assertAlmostEqual(batches.loc[0, "per_image_server_time_s"], 1.0 / 3.0)
        self.assertEqual(batches.loc[0, "lml_micro_batches_observed"], 2)
        self.assertAlmostEqual(batches.loc[0, "lml_wall_time_s"], 0.8)
        self.assertAlmostEqual(
            batches.loc[0, "effective_server_throughput_samples_s"], 3.0
        )
        self.assertAlmostEqual(batches.loc[1, "server_response_time_s"], 0.8)

    def test_rejects_inconsistent_values_within_one_batch(self):
        raw_results = pd.DataFrame(
            {
                "ts_sample_sent_to_edge_server": [10.0, 10.0],
                "ts_sample_received_at_edge_server": [10.1, 10.2],
                "ts_lml_inference_start": [10.3, 10.3],
                "ts_lml_inference_end": [10.8, 10.8],
                "ts_results_sent_to_edge_device": [10.9, 10.9],
            }
        )

        with self.assertRaisesRegex(ValueError, "distinct values"):
            self.analyzer.extract_batch_measurements(self.context, raw_results)

    def test_rejects_missing_raw_columns(self):
        raw_results = pd.DataFrame({"ts_sample_sent_to_edge_server": [10.0]})

        with self.assertRaisesRegex(ValueError, "missing required"):
            self.analyzer.extract_batch_measurements(self.context, raw_results)

    def test_summarizes_measurements_by_actual_batch_size(self):
        measurements = self.measurements(
            batch_sizes=[1, 1, 2, 2],
            response_times=[1.0, 3.0, 4.0, 6.0],
        )

        summary = self.analyzer.summarize_by_batch_size(measurements)

        self.assertEqual(summary["actual_server_batch_size"].tolist(), [1, 2])
        self.assertEqual(summary["batch_count"].tolist(), [2, 2])
        self.assertEqual(summary["batch_share_percent"].tolist(), [50.0, 50.0])
        self.assertAlmostEqual(summary.loc[0, "response_time_mean_s"], 2.0)
        self.assertAlmostEqual(summary.loc[1, "response_time_median_s"], 5.0)
        self.assertAlmostEqual(summary.loc[1, "per_image_time_mean_s"], 2.5)
        self.assertAlmostEqual(
            summary.loc[1, "per_image_time_std_s"],
            np.std([2.0, 3.0], ddof=1),
        )
        self.assertAlmostEqual(
            summary.loc[1, "throughput_median_samples_s"],
            np.median([2.0 / 4.0, 2.0 / 6.0]),
        )

    def test_calculates_per_config_batch_size_trends(self):
        measurements = self.measurements(
            batch_sizes=[1, 2, 3, 4],
            response_times=[0.75, 1.25, 1.75, 2.25],
        )

        trends = self.analyzer.calculate_config_trends(measurements)

        self.assertEqual(trends.loc[0, "request_count"], 4)
        self.assertEqual(trends.loc[0, "actual_batch_size_min"], 1)
        self.assertEqual(trends.loc[0, "actual_batch_size_max"], 4)
        self.assertAlmostEqual(trends.loc[0, "response_time_pearson_r"], 1.0)
        self.assertAlmostEqual(trends.loc[0, "response_time_spearman_r"], 1.0)
        self.assertAlmostEqual(
            trends.loc[0, "response_time_slope_s_per_sample"], 0.5
        )
        self.assertAlmostEqual(
            trends.loc[0, "response_time_linear_r_squared"], 1.0
        )
        self.assertLess(trends.loc[0, "per_image_time_spearman_r"], 0.0)
        self.assertGreater(trends.loc[0, "throughput_spearman_r"], 0.0)

    def measurements(
        self,
        *,
        batch_sizes: list[int],
        response_times: list[float],
    ) -> pd.DataFrame:
        rows = []
        for batch_id, (batch_size, response_time) in enumerate(
            zip(batch_sizes, response_times)
        ):
            rows.append(
                {
                    "config": self.context.config_id,
                    "controller_batch_size": self.context.controller_batch_size,
                    "edge_server_batch_id": batch_id,
                    "actual_server_batch_size": batch_size,
                    "server_received_at": float(batch_id),
                    "server_results_sent_at": float(batch_id) + response_time,
                    "lml_micro_batches_observed": 1,
                    "server_queue_or_preprocess_s": 0.1,
                    "lml_wall_time_s": response_time - 0.2,
                    "server_postprocess_s": 0.1,
                    "server_response_time_s": response_time,
                    "per_image_server_time_s": response_time / batch_size,
                    "effective_server_throughput_samples_s": (
                        batch_size / response_time
                    ),
                }
            )
        return pd.DataFrame(rows, columns=self.analyzer.OUTPUT_COLUMNS)


if __name__ == "__main__":
    unittest.main()
