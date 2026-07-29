import sys
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
