"""Command-line entry point for the reproducible workflow."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from stem_analytics.config import load_config
from stem_analytics.data_ingestion import fetch_and_save


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stem-analytics",
        description="Build and evaluate the STEM subject-classification workflow.",
    )
    subparsers = parser.add_subparsers(dest="command")
    fetch = subparsers.add_parser("fetch", help="Download the pinned source dataset.")
    fetch.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "fetch":
        manifest = fetch_and_save(load_config(args.config))
        print(
            f"Fetched {manifest['selected_rows']} rows from "
            f"{manifest['dataset_name']}@{manifest['dataset_revision']}"
        )
    else:
        parser.print_help()
    return 0

