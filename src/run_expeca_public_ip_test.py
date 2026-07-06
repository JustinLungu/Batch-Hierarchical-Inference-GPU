import requests

from experiment_runner import ExperimentRunner


class ExpecaPublicIpRunner(ExperimentRunner):
    MODE = "expeca_public_ip"
    RUN_LABEL = "expeca_public_ip"
    ANALYSIS_LABEL = "expeca_public_ip"

    def start_services(self) -> None:
        print("Using already-running ExPECA public-IP containers.")
        self.check_remote_service(self.edge_server_url, "edge server")
        self.check_remote_service(self.edge_device_url, "edge device")

    def stop_services(self) -> None:
        print("Leaving ExPECA containers running.")

    def run(self) -> int:
        self.start_services()
        self.send_config()
        self.send_samples()
        self.download_remote_results()
        self.post_process_results()
        self.stop_services()
        return 0

    def download_remote_results(self) -> None:
        print("Downloading remote edge-device results...")
        response = requests.get(f"{self.edge_device_url}/results", timeout=60)
        response.raise_for_status()
        self.raw_results_csv.parent.mkdir(parents=True, exist_ok=True)
        temp_results_csv = self.raw_results_csv.with_suffix(".tmp")
        temp_results_csv.write_bytes(response.content)
        temp_results_csv.replace(self.raw_results_csv)
        print(f"Downloaded remote results to: {self.raw_results_csv}")

    @staticmethod
    def check_remote_service(url: str, label: str) -> None:
        response = requests.get(f"{url}/logs", timeout=30)
        response.raise_for_status()
        print(f"{label} reachable: {url}")


def main() -> int:
    return ExpecaPublicIpRunner().run()


if __name__ == "__main__":
    raise SystemExit(main())
