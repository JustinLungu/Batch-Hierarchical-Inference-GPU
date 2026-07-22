import argparse

from thesis_reproduction_runner import ThesisReproductionRunner


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the thesis reproduction configurations on ExPECA public-IP containers."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate assets and print the thesis configurations without sending requests.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate thesis-style plots from existing CSV outputs without rerunning ExPECA.",
    )
    args = parser.parse_args()
    return ThesisReproductionRunner(
        dry_run=args.dry_run,
        plot_only=args.plot_only,
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
