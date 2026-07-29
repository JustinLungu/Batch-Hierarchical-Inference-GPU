import csv
import os
from dataclasses import dataclass
from pathlib import Path


CONFIG_MATRIX_FILE = Path("config/thesis_configs.csv")
BENCHMARK_DEFAULTS_FILE = Path("config/thesis_reproduction.env")
EXPECTED_CONFIG_IDS = ("001", "002", "003", "004", "005", "006", "007")


@dataclass(frozen=True)
class BenchmarkConfiguration:
    config_id: str
    decision_method: str
    offloading_strategy: str
    controller_batch_size: int
    batch_size: int
    fixed_threshold_value: str
    description: str

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> "BenchmarkConfiguration":
        return cls(
            config_id=row["config_id"].strip(),
            decision_method=row["decision_method"].strip(),
            offloading_strategy=row["offloading_strategy"].strip(),
            controller_batch_size=int(row["controller_batch_size"]),
            batch_size=int(row["batch_size"]),
            fixed_threshold_value=row["fixed_threshold_value"].strip(),
            description=row["description"].strip(),
        )

    def overrides(
        self,
        benchmark_defaults: dict[str, str],
        sample_limit: str,
    ) -> dict[str, str]:
        return {
            **benchmark_defaults,
            "DECISION_METHOD": self.decision_method,
            "OFFLOADING_STRATEGY": self.offloading_strategy,
            "FIXED_THRESHOLD_VALUE": self.fixed_threshold_value,
            "CONTROLLER_BATCH_SIZE": str(self.controller_batch_size),
            "BATCH_SIZE": str(self.batch_size),
            "CONTROLLER_MAX_SAMPLES": sample_limit,
        }


def config_value(config: dict[str, str], key: str, default: str) -> str:
    return os.environ.get(key, config.get(key, default)).strip()


def parse_selected_config_ids(raw_value: str) -> set[str] | None:
    if raw_value.lower() in {"all", "001-007", "1-7"}:
        return None
    return {
        item.strip().zfill(3)
        for item in raw_value.replace(";", ",").split(",")
        if item.strip()
    }


def load_configurations(
    path: Path = CONFIG_MATRIX_FILE,
    selection: str = "all",
) -> list[BenchmarkConfiguration]:
    if not path.exists():
        raise RuntimeError(f"Missing benchmark config table: {path}")

    with path.open(newline="") as config_file:
        reader = csv.DictReader(config_file)
        required_columns = {
            "config_id",
            "decision_method",
            "offloading_strategy",
            "controller_batch_size",
            "batch_size",
            "fixed_threshold_value",
            "description",
        }
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise RuntimeError(f"{path} is missing column(s): {missing}")
        configurations = [
            BenchmarkConfiguration.from_csv_row(row) for row in reader
        ]

    if tuple(config.config_id for config in configurations) != EXPECTED_CONFIG_IDS:
        raise RuntimeError(f"{path} must define configs 001 through 007 in order.")

    selected_ids = parse_selected_config_ids(selection.strip())
    if selected_ids is None:
        return configurations

    selected = [
        config for config in configurations if config.config_id in selected_ids
    ]
    missing = selected_ids - {config.config_id for config in selected}
    if missing:
        raise ValueError(
            "THESIS_CONFIGS_TO_RUN contains unknown config id(s): "
            f"{', '.join(sorted(missing))}"
        )
    return selected


def configuration_label(configurations: list[BenchmarkConfiguration]) -> str:
    return ",".join(config.config_id for config in configurations)


def output_dir_name(config: dict[str, str], device: str) -> str:
    configured = config_value(config, "THESIS_OUTPUT_DIR", "")
    if configured:
        return configured
    if device == "cuda":
        return "thesis_reproduction_gpu"
    return "thesis_reproduction"


def validate_assets(benchmark_defaults: dict[str, str]) -> None:
    required_keys = ["SAMPLE_PATH", "SML_MODEL", "LML_MODEL"]
    missing = [
        benchmark_defaults[key]
        for key in required_keys
        if key not in benchmark_defaults or not Path(benchmark_defaults[key]).exists()
    ]
    if not missing:
        return

    formatted = "\n".join(f"  - {path}" for path in missing)
    raise RuntimeError(
        "Missing benchmark asset(s):\n"
        f"{formatted}\n"
        "Run `scripts/download_dataset.sh --imagenetv2` and "
        "`scripts/download_models.sh --all` first."
    )


def print_dry_run(
    *,
    device: str,
    sample_limit: str,
    benchmark_defaults: dict[str, str],
    configurations: list[BenchmarkConfiguration],
) -> None:
    print("Benchmark configuration:")
    print(f"  DEVICE={device}")
    print(f"  CONTROLLER_MAX_SAMPLES={sample_limit}")
    print(f"  SAMPLE_PATH={benchmark_defaults['SAMPLE_PATH']}")
    print(f"  SML_ARCH={benchmark_defaults['SML_ARCH']}")
    print(f"  LML_ARCH={benchmark_defaults['LML_ARCH']}")
    print()
    for config in configurations:
        print(
            f"{config.config_id}: "
            f"DECISION_METHOD={config.decision_method}, "
            f"OFFLOADING_STRATEGY={config.offloading_strategy}, "
            f"CONTROLLER_BATCH_SIZE={config.controller_batch_size}, "
            f"BATCH_SIZE={config.batch_size}, "
            f"FIXED_THRESHOLD_VALUE={config.fixed_threshold_value}"
        )
