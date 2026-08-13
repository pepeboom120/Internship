"""Command-line entry point for the reproducible workflow."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from stem_analytics.analysis import build_eda_figures, write_figure_register
from stem_analytics.config import load_config
from stem_analytics.data_ingestion import fetch_and_save
from stem_analytics.data_validation import validate_and_save
from stem_analytics.database import build_database_and_export
from stem_analytics.evaluation import evaluate_and_save
from stem_analytics.modeling import train_and_save


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stem-analytics",
        description="Build and evaluate the STEM subject-classification workflow.",
    )
    subparsers = parser.add_subparsers(dest="command")
    fetch = subparsers.add_parser("fetch", help="Download the pinned source dataset.")
    fetch.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    validate = subparsers.add_parser("validate", help="Validate and split source records.")
    validate.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    database = subparsers.add_parser("build-db", help="Build SQLite and export SQL tables.")
    database.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    analyze = subparsers.add_parser("analyze", help="Generate EDA figures and provenance.")
    analyze.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    train = subparsers.add_parser("train", help="Tune candidates on grouped development folds.")
    train.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    evaluate = subparsers.add_parser(
        "evaluate", help="Evaluate the locked model on the final test set once."
    )
    evaluate.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    evaluate.add_argument("--run-id")
    evaluate.add_argument(
        "--force",
        action="store_true",
        help="Explicitly replace an existing final evaluation and record that action.",
    )
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
    elif args.command == "validate":
        config = load_config(args.config)
        source = pd.read_parquet(config.paths.raw_data)
        manifest = validate_and_save(source, config)
        print(
            f"Validated {manifest['clean_rows']} rows; excluded "
            f"{manifest['excluded_rows']}; crossing groups "
            f"{manifest['crossing_duplicate_groups']}"
        )
    elif args.command == "build-db":
        config = load_config(args.config)
        controlled = pd.read_parquet(config.paths.processed_data)
        result = build_database_and_export(controlled, config)
        print(
            f"Loaded {result['questions_loaded']} questions and exported "
            f"{result['queries_exported']} SQL tables"
        )
    elif args.command == "analyze":
        config = load_config(args.config)
        records = build_eda_figures(
            config.paths.database,
            config.paths.reports / "figures",
            Path("sql/analysis"),
        )
        write_figure_register(
            records, config.paths.reports / "qa" / "figure_source_register.csv"
        )
        print(f"Generated {len(records)} verified EDA figures")
    elif args.command == "train":
        config = load_config(args.config)
        controlled = pd.read_parquet(config.paths.processed_data)
        result = train_and_save(controlled, config)
        print(
            f"Selected {result.selected_model} with grouped-CV macro-F1 "
            f"{result.selection_decision['cv_macro_f1_mean']:.4f} "
            f"(run {result.run_id})"
        )
    elif args.command == "evaluate":
        config = load_config(args.config)
        controlled = pd.read_parquet(config.paths.processed_data)
        metrics, status = evaluate_and_save(
            controlled, config, run_id=args.run_id, force=args.force
        )
        print(
            f"Final evaluation {status}: {metrics['model_name']} macro-F1 "
            f"{metrics['macro_f1']:.4f} (run {metrics['run_id']})"
        )
    else:
        parser.print_help()
    return 0

