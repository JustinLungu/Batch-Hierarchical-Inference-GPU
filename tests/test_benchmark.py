import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from benchmark.config import BenchmarkConfiguration
from benchmark.metrics import BenchmarkMetrics
from benchmark.run import BenchmarkRun
from benchmark.utils import load_env_file


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.metrics = BenchmarkMetrics()
        self.config = BenchmarkConfiguration(
            config_id="004",
            decision_method="adaptive_threshold",
            offloading_strategy="send_individually",
            controller_batch_size=1,
            batch_size=1,
            fixed_threshold_value="0.3888",
            description="Adaptive threshold",
        )

    def test_configuration_parsing_and_overrides(self):
        config = BenchmarkConfiguration.from_csv_row(
            {
                "config_id": " 005 ",
                "decision_method": " adaptive_threshold ",
                "offloading_strategy": " dynamic_batching ",
                "controller_batch_size": "5",
                "batch_size": "5",
                "fixed_threshold_value": " 0.3888 ",
                "description": " Dynamic batching ",
            }
        )

        overrides = config.overrides({"DEVICE": "cuda"}, sample_limit="100")

        self.assertEqual(config.config_id, "005")
        self.assertEqual(config.controller_batch_size, 5)
        self.assertEqual(overrides["DEVICE"], "cuda")
        self.assertEqual(overrides["DECISION_METHOD"], "adaptive_threshold")
        self.assertEqual(overrides["OFFLOADING_STRATEGY"], "dynamic_batching")
        self.assertEqual(overrides["CONTROLLER_BATCH_SIZE"], "5")
        self.assertEqual(overrides["BATCH_SIZE"], "5")
        self.assertEqual(overrides["CONTROLLER_MAX_SAMPLES"], "100")

    def test_load_env_file_ignores_comments_and_preserves_equals(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / "experiment.env"
            env_file.write_text(
                "\n"
                "# benchmark settings\n"
                "DEVICE='cuda'\n"
                "TOKEN=\"part=two\"\n"
                "INVALID_LINE\n"
            )

            values = load_env_file(env_file)

        self.assertEqual(values, {"DEVICE": "cuda", "TOKEN": "part=two"})

    def test_add_timing_durations_handles_offloaded_and_local_samples(self):
        raw = pd.DataFrame(
            {
                "ts_sml_inference_start": [1.0, 2.0],
                "ts_sml_inference_end": [1.2, 2.2],
                "ts_offload_decision_made": [1.21, 2.21],
                "ts_results_saved_not_offloaded": [float("nan"), 2.23],
                "ts_sample_sent_to_offloading": [1.22, float("nan")],
                "ts_sample_sent_to_edge_server": [1.30, float("nan")],
                "ts_sample_received_at_edge_server": [1.35, float("nan")],
                "ts_lml_inference_start": [1.40, float("nan")],
                "ts_lml_inference_end": [1.70, float("nan")],
                "ts_results_sent_to_edge_device": [1.72, float("nan")],
                "ts_results_received_from_edge_server": [1.77, float("nan")],
                "ts_results_received_from_offloading_module": [1.80, float("nan")],
            }
        )

        run = object.__new__(BenchmarkRun)
        timing = run.add_timing_durations(raw)

        self.assertAlmostEqual(timing.loc[0, "sml_inference_s"], 0.2)
        self.assertAlmostEqual(timing.loc[0, "edge_buffer_wait_s"], 0.08)
        self.assertAlmostEqual(timing.loc[0, "lml_inference_s"], 0.3)
        self.assertAlmostEqual(timing.loc[0, "total_tracked_latency_s"], 0.8)
        self.assertAlmostEqual(timing.loc[1, "total_tracked_latency_s"], 0.23)
        self.assertEqual(timing.loc[0, "edge_server_batch_id"], 0)
        self.assertTrue(pd.isna(timing.loc[1, "edge_server_batch_id"]))

    def test_accuracy_and_offloading_distribution_use_final_predictions(self):
        timing = pd.DataFrame(
            {
                "True Class": [1, 2, 3],
                "SML Prediction": [1, 1, 3],
                "LML Prediction": [9, 2, float("nan")],
                "Offloaded": [True, True, False],
            }
        )

        accuracy = self.metrics.accuracy_metrics(timing)
        distribution = self.metrics.offloading_distribution_row(self.config, timing)

        self.assertAlmostEqual(accuracy["accuracy"], 2.0 / 3.0)
        self.assertAlmostEqual(accuracy["sml_accuracy"], 2.0 / 3.0)
        self.assertAlmostEqual(accuracy["lml_accuracy_offloaded"], 0.5)
        self.assertAlmostEqual(accuracy["sml_accuracy_not_offloaded"], 1.0)
        self.assertEqual(accuracy["correct"], 2)
        self.assertEqual(distribution["true_positive"], 1)
        self.assertEqual(distribution["true_negative"], 1)
        self.assertEqual(distribution["false_positive"], 1)
        self.assertEqual(distribution["false_negative"], 0)

    def test_latency_breakdown_uses_only_offloaded_rows_for_server_path(self):
        timing = pd.DataFrame(
            {
                "Offloaded": [True, False],
                "sml_inference_s": [0.20, 0.40],
                "offload_decision_s": [0.01, 0.03],
                "edge_buffer_wait_s": [0.50, 50.0],
                "edge_to_server_network_s": [0.10, 50.0],
                "server_queue_or_preprocess_s": [0.20, 50.0],
                "lml_inference_s": [3.00, 50.0],
                "server_postprocess_s": [0.10, 50.0],
                "server_to_edge_network_s": [0.15, 50.0],
                "edge_receive_to_saved_s": [0.05, 50.0],
                "total_tracked_latency_s": [4.31, 0.43],
            }
        )

        breakdown = self.metrics.latency_breakdown_row(self.config, timing)

        self.assertAlmostEqual(breakdown["step_1_ed_processing_s"], 0.32)
        self.assertAlmostEqual(breakdown["step_2_ed_offload_buffer_s"], 0.50)
        self.assertAlmostEqual(breakdown["step_3_ed_to_es_communication_s"], 0.10)
        self.assertAlmostEqual(breakdown["step_4_es_processing_s"], 3.30)
        self.assertAlmostEqual(breakdown["step_5_es_to_ed_communication_s"], 0.15)
        self.assertAlmostEqual(breakdown["step_6_ed_result_saving_s"], 0.05)


if __name__ == "__main__":
    unittest.main()
