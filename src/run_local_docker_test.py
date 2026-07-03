import subprocess

from constants import (
    DOCKER_EDGE_DEVICE_CONTAINER,
    DOCKER_EDGE_DEVICE_DOCKERFILE,
    DOCKER_EDGE_DEVICE_IMAGE,
    DOCKER_EDGE_SERVER_CONTAINER,
    DOCKER_EDGE_SERVER_DOCKERFILE,
    DOCKER_EDGE_SERVER_IMAGE,
    DOCKER_NETWORK,
    REPO_ROOT,
)
from experiment_runner import ExperimentRunner
from utils import require_config_bool, wait_for_server


class LocalDockerRunner(ExperimentRunner):
    MODE = "local_docker_containers"
    RUN_LABEL = "local_docker"
    ANALYSIS_LABEL = "local_docker"

    def __init__(self):
        super().__init__()
        self.force_docker_build = require_config_bool(
            {**self.config, "FORCE_DOCKER_BUILD": self.config.get("FORCE_DOCKER_BUILD", "false")},
            "FORCE_DOCKER_BUILD",
        )

    def start_services(self) -> None:
        self._build_images_if_needed()
        self._ensure_network()
        self._remove_container(DOCKER_EDGE_DEVICE_CONTAINER)
        self._remove_container(DOCKER_EDGE_SERVER_CONTAINER)

        print("Starting edge server container...")
        self._docker_run(
            [
                "--name",
                DOCKER_EDGE_SERVER_CONTAINER,
                "-p",
                f"{self.edge_server_port}:{self.edge_server_port}",
                "-e",
                f"DEVICE={self.device}",
                "-e",
                f"EDGE_SERVER_PORT={self.edge_server_port}",
                DOCKER_EDGE_SERVER_IMAGE,
            ]
        )
        wait_for_server(self.edge_server_url)

        print("Starting edge device container...")
        self._docker_run(
            [
                "--name",
                DOCKER_EDGE_DEVICE_CONTAINER,
                "-p",
                f"{self.edge_device_port}:{self.edge_device_port}",
                "-e",
                f"DEVICE={self.device}",
                "-e",
                f"EDGE_SERVER_IP={DOCKER_EDGE_SERVER_CONTAINER}",
                "-e",
                f"EDGE_DEVICE_PORT={self.edge_device_port}",
                "-e",
                f"EDGE_SERVER_PORT={self.edge_server_port}",
                DOCKER_EDGE_DEVICE_IMAGE,
            ]
        )
        wait_for_server(self.edge_device_url)

    def _docker_run(self, args: list[str]) -> None:
        command = [
            "docker",
            "run",
            "-d",
            "--network",
            DOCKER_NETWORK,
            "-v",
            f"{REPO_ROOT / 'data'}:/app/data:ro",
            "-v",
            f"{REPO_ROOT / 'results'}:/app/results",
            *args,
        ]
        self._run(command)

    def _build_images_if_needed(self) -> None:
        self._build_image_if_needed(
            DOCKER_EDGE_SERVER_IMAGE,
            DOCKER_EDGE_SERVER_DOCKERFILE,
        )
        self._build_image_if_needed(
            DOCKER_EDGE_DEVICE_IMAGE,
            DOCKER_EDGE_DEVICE_DOCKERFILE,
        )

    def _build_image_if_needed(self, image: str, dockerfile) -> None:
        if not self.force_docker_build and self._image_exists(image):
            print(f"Docker image already exists: {image}")
            return

        print(f"Building Docker image: {image}")
        self._run(
            [
                "docker",
                "build",
                "-f",
                str(dockerfile),
                "-t",
                image,
                ".",
            ]
        )

    def _image_exists(self, image: str) -> bool:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0

    def _ensure_network(self) -> None:
        result = subprocess.run(
            ["docker", "network", "inspect", DOCKER_NETWORK],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode == 0:
            return
        self._run(["docker", "network", "create", DOCKER_NETWORK])

    def _remove_container(self, container: str) -> None:
        subprocess.run(
            ["docker", "rm", "-f", container],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    def stop_services(self) -> None:
        self._remove_container(DOCKER_EDGE_DEVICE_CONTAINER)
        self._remove_container(DOCKER_EDGE_SERVER_CONTAINER)

    def _run(self, command: list[str]) -> None:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> int:
    return LocalDockerRunner().run()


if __name__ == "__main__":
    raise SystemExit(main())
