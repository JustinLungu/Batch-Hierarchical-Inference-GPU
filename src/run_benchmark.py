import argparse

from benchmark import BenchmarkRunner


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the CPU/GPU benchmark on ExPECA public-IP containers."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate assets and print configurations without sending requests.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate plots from existing CSV outputs without rerunning ExPECA.",
    )
    args = parser.parse_args()
    return BenchmarkRunner(dry_run=args.dry_run, plot_only=args.plot_only).run()


if __name__ == "__main__":
    raise SystemExit(main())
