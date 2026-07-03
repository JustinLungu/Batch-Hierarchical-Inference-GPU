import os
import subprocess
import sys

from constants import EDGE_DEVICE_SCRIPT, EDGE_SERVER_SCRIPT, SMOKE_LOG_DIR
from experiment_runner import ExperimentRunner
from utils import start_process, wait_for_server


class LocalPythonRunner(ExperimentRunner):
    MODE = "local_python_scripts"
    RUN_LABEL = "local_python"
    ANALYSIS_LABEL = "local"

    def __init__(self):
        super().__init__()
        self.processes: list[subprocess.Popen] = []

    def start_services(self) -> None:
        env = self.service_env()
        python = sys.executable

        print("Starting edge server...")
        self.processes.append(
            start_process(
                [python, str(EDGE_SERVER_SCRIPT)],
                SMOKE_LOG_DIR / "edge_server_stdout.log",
                env,
            )
        )
        wait_for_server(self.edge_server_url)

        print("Starting edge device...")
        self.processes.append(
            start_process(
                [python, str(EDGE_DEVICE_SCRIPT)],
                SMOKE_LOG_DIR / "edge_device_stdout.log",
                env,
            )
        )
        wait_for_server(self.edge_device_url)

    def service_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["DEVICE"] = self.device
        env["EDGE_SERVER_IP"] = self.edge_server_host
        env["EDGE_DEVICE_PORT"] = self.edge_device_port
        env["EDGE_SERVER_PORT"] = self.edge_server_port
        return env

    def stop_services(self) -> None:
        for process in reversed(self.processes):
            process.terminate()
        for process in reversed(self.processes):
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


def main() -> int:
    return LocalPythonRunner().run()


if __name__ == "__main__":
    raise SystemExit(main())
