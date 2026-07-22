from dataclasses import dataclass
from pathlib import Path


THESIS_CONFIG_FILE = Path("config/thesis_configs.csv")
THESIS_REPRODUCTION_FILE = Path("config/thesis_reproduction.env")


@dataclass(frozen=True)
class ThesisConfiguration:
    config_id: str
    decision_method: str
    offloading_strategy: str
    controller_batch_size: int
    batch_size: int
    fixed_threshold_value: str
    description: str

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> "ThesisConfiguration":
        return cls(
            config_id=row["config_id"].strip(),
            decision_method=row["decision_method"].strip(),
            offloading_strategy=row["offloading_strategy"].strip(),
            controller_batch_size=int(row["controller_batch_size"]),
            batch_size=int(row["batch_size"]),
            fixed_threshold_value=row["fixed_threshold_value"].strip(),
            description=row["description"].strip(),
        )

    def overrides(self, thesis_base: dict[str, str], sample_limit: str) -> dict[str, str]:
        return {
            **thesis_base,
            "DECISION_METHOD": self.decision_method,
            "OFFLOADING_STRATEGY": self.offloading_strategy,
            "FIXED_THRESHOLD_VALUE": self.fixed_threshold_value,
            "CONTROLLER_BATCH_SIZE": str(self.controller_batch_size),
            "BATCH_SIZE": str(self.batch_size),
            "CONTROLLER_MAX_SAMPLES": sample_limit,
        }
