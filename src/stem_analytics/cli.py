"""Command-line entry point for the reproducible workflow."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from stem_analytics.analysis import build_eda_figures, write_figure_register
from stem_analytics.annotation import sample_intent_pilot
from stem_analytics.config import load_config
from stem_analytics.data_ingestion import fetch_and_save
from stem_analytics.data_validation import validate_and_save
from stem_analytics.database import build_database_and_export
from stem_analytics.error_analysis import diagnose_and_save
from stem_analytics.evaluation import evaluate_and_save, find_selection_decision
from stem_analytics.modeling import train_and_save
from stem_analytics.reporting import generate_report


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
    diagnose = subparsers.add_parser(
        "diagnose", help="Generate post-lock diagnostic evidence and figures."
    )
    diagnose.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    diagnose.add_argument("--run-id")
    report = subparsers.add_parser(
        "report", help="Generate evidence-grounded Markdown and README artifacts."
    )
    report.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    pilot = subparsers.add_parser(
        "sample-intent-pilot", help="Create an unlabeled future-work annotation template."
    )
    pilot.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    run_all = subparsers.add_parser(
        "run-all", help="Run every workflow stage, visibly reusing completed artifacts."
    )
    run_all.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    run_all.add_argument(
        "--force", action="store_true", help="Rerun all stages including final evaluation."
    )
    return parser


def _stage_outputs(stage: str, config: object) -> list[Path]:
    paths = config.paths
    mapping = {
        "fetch": [paths.raw_data, paths.artifacts / "manifests" / "source_manifest.json"],
        "validate": [paths.processed_data, paths.artifacts / "manifests" / "validation_manifest.json"],
        "build-db": [paths.database, paths.reports / "tables" / "05_split_integrity.csv"],
        "analyze": [paths.reports / "figures" / "05_split_distribution.png"],
        "evaluate": [paths.artifacts / "metrics" / "final_test_metrics.json", paths.artifacts / "predictions" / "final_test_predictions.csv"],
        "diagnose": [paths.reports / "figures" / "12_learning_curve.png", paths.reports / "tables" / "source_holdout_results.csv"],
        "report": [Path("docs/final_report.md"), Path("README.md"), paths.reports / "qa" / "numerical_consistency_register.csv"],
    }
    if stage == "train":
        return list((paths.artifacts / "runs").glob("*/selection_decision.json"))
    return mapping[stage]


def _run_all(config_path: Path, force: bool) -> int:
    config = load_config(config_path)
    for stage in [
        "fetch", "validate", "build-db", "analyze", "train", "evaluate", "diagnose", "report"
    ]:
        outputs = _stage_outputs(stage, config)
        if not force and outputs and all(path.exists() for path in outputs):
            print(f"CACHED [{stage}]")
            continue
        arguments = [stage, "--config", str(config_path)]
        if stage == "evaluate" and force:
            arguments.append("--force")
        result = main(arguments)
        if result != 0:
            return result
    return 0


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
    elif args.command == "diagnose":
        config = load_config(args.config)
        controlled = pd.read_parquet(config.paths.processed_data)
        selection_path = find_selection_decision(config.paths.artifacts, args.run_id)
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        metrics = json.loads(
            (config.paths.artifacts / "metrics" / "final_test_metrics.json").read_text(
                encoding="utf-8"
            )
        )
        predictions = pd.read_csv(
            config.paths.artifacts / "predictions" / "final_test_predictions.csv"
        )
        result = diagnose_and_save(
            controlled, selection, predictions, metrics, config
        )
        print(
            f"Generated {result['diagnostic_tables']} diagnostic tables and "
            f"{result['model_figures']} figures; source holdout "
            f"{result['source_holdout_status']}"
        )
    elif args.command == "report":
        result = generate_report(load_config(args.config))
        print(
            f"Generated {result['capability']} report with "
            f"{result['evidence_rows']} verified claims and "
            f"{result['consistency_failures']} consistency failures"
        )
    elif args.command == "sample-intent-pilot":
        config = load_config(args.config)
        controlled = pd.read_parquet(config.paths.processed_data)
        pilot_frame = sample_intent_pilot(
            controlled.loc[controlled["split_name"] == "development"],
            n_per_subject=10,
            seed=config.random_seed,
        )
        output = config.paths.reports / "tables" / "intent_pilot_sample.csv"
        pilot_frame.to_csv(output, index=False, encoding="utf-8")
        print(f"Generated {len(pilot_frame)} unlabeled pilot rows at {output}")
    elif args.command == "run-all":
        return _run_all(args.config, args.force)
    else:
        parser.print_help()
    return 0

