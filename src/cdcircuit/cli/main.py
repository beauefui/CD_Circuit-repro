from __future__ import annotations

import argparse

from cdcircuit.pipelines.greater_than_pipeline import run_greater_than_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cdc")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run a pipeline")
    run_sub = run_parser.add_subparsers(dest="task", required=True)

    gt = run_sub.add_parser("greater-than", help="Run greater-than pipeline")
    gt.add_argument("--config", required=True, help="Path to YAML config")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run" and args.task == "greater-than":
        run_dir = run_greater_than_pipeline(args.config)
        print(run_dir)
        return

    parser.error("Unsupported command")


if __name__ == "__main__":
    main()
