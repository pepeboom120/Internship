"""One-time evaluation of the model selected without final-test access."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline

from stem_analytics.config import ProjectConfig
from stem_analytics.database import (
    export_named_queries,
    load_model_run_and_predictions,
)
from stem_analytics.modeling import build_pipeline
from stem_analytics.provenance import canonical_sha256, config_hash, write_json_atomic


@dataclass
class EvaluationResult:
    """In-memory evidence produced by the locked final evaluation."""

    metrics: dict[str, Any]
    predictions: pd.DataFrame
    estimator: Pipeline


def classification_metrics(
    y_true: list[str] | pd.Series,
    y_pred: list[str] | pd.Series,
    labels: list[str],
) -> dict[str, Any]:
    """Calculate fixed-label aggregate, per-class, and confusion metrics."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    per_class = {
        label: {
            "precision": float(class_precision),
            "recall": float(class_recall),
            "f1": float(class_f1),
            "support": int(class_support),
        }
        for label, class_precision, class_recall, class_f1, class_support in zip(
            labels, precision, recall, f1, support, strict=True
        )
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "label_order": labels,
    }


def stratified_bootstrap_macro_f1(
    y_true: list[str] | pd.Series,
    y_pred: list[str] | pd.Series,
    labels: list[str],
    iterations: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    """Return a percentile interval from bootstrap samples within each class."""
    if iterations < 1:
        raise ValueError("bootstrap iterations must be positive")
    true_values = np.asarray(y_true)
    predicted_values = np.asarray(y_pred)
    if len(true_values) != len(predicted_values) or len(true_values) == 0:
        raise ValueError("bootstrap inputs must be non-empty and have equal length")

    class_indices = [np.flatnonzero(true_values == label) for label in labels]
    if any(len(indices) == 0 for indices in class_indices):
        raise ValueError("every configured label must appear in bootstrap data")

    generator = np.random.default_rng(seed)
    scores = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sampled = np.concatenate(
            [generator.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        scores[iteration] = f1_score(
            true_values[sampled],
            predicted_values[sampled],
            labels=labels,
            average="macro",
            zero_division=0,
        )
    lower, upper = np.percentile(scores, [2.5, 97.5])
    return float(lower), float(upper)


def _development_hash(frame: pd.DataFrame) -> str:
    development = frame.loc[frame["split_name"] == "development"]
    return canonical_sha256(
        development[["question_text", "category", "duplicate_group_id"]].to_dict(
            "records"
        )
    )


def _restored_parameters(selection: dict[str, Any]) -> dict[str, Any]:
    parameters = dict(selection["best_params"])
    ngram_range = parameters.get("tfidf__ngram_range")
    if isinstance(ngram_range, list):
        parameters["tfidf__ngram_range"] = tuple(ngram_range)
    return parameters


def evaluate_locked_model(
    selection: dict[str, Any], frame: pd.DataFrame, config: ProjectConfig
) -> EvaluationResult:
    """Refit the exact selected pipeline and inspect the final test set once."""
    if selection.get("config_hash") != config_hash(config):
        raise ValueError("selection decision does not match the current config")
    development_hash = _development_hash(frame)
    if selection.get("development_data_hash") != development_hash:
        raise ValueError("selection decision does not match the development data")

    selected_model = selection.get("selected_model")
    if selected_model not in {"logistic_regression", "linear_svc"}:
        raise ValueError(f"unsupported locked model: {selected_model}")
    development = frame.loc[frame["split_name"] == "development"].reset_index(drop=True)
    final_test = frame.loc[frame["split_name"] == "test"].reset_index(drop=True)
    if development.empty or final_test.empty:
        raise ValueError("both development and final-test rows are required")

    estimator = build_pipeline(selected_model, config.random_seed)
    estimator.set_params(**_restored_parameters(selection))
    started = time.perf_counter()
    estimator.fit(development["question_text"], development["category"])
    predicted = estimator.predict(final_test["question_text"])
    runtime_seconds = time.perf_counter() - started

    confidence: np.ndarray | list[None]
    decision_score: np.ndarray | list[None]
    if hasattr(estimator, "predict_proba"):
        probability = estimator.predict_proba(final_test["question_text"])
        confidence = np.max(probability, axis=1)
        decision_score = [None] * len(final_test)
    else:
        # LinearSVC margins are useful diagnostically but are not probabilities.
        margins = estimator.decision_function(final_test["question_text"])
        confidence = [None] * len(final_test)
        decision_score = np.max(margins, axis=1)

    metrics = classification_metrics(
        final_test["category"], predicted, config.subjects
    )
    lower, upper = stratified_bootstrap_macro_f1(
        final_test["category"],
        predicted,
        config.subjects,
        config.evaluation.bootstrap_iterations,
        config.random_seed,
    )
    metrics.update(
        {
            "run_id": selection["run_id"],
            "model_name": selected_model,
            "best_params": selection["best_params"],
            "cv_macro_f1_mean": selection["cv_macro_f1_mean"],
            "cv_macro_f1_std": selection["cv_macro_f1_std"],
            "bootstrap_macro_f1_95_ci": {
                "lower": lower,
                "upper": upper,
                "iterations": config.evaluation.bootstrap_iterations,
                "method": "stratified percentile bootstrap",
            },
            "development_rows": len(development),
            "final_test_rows": len(final_test),
            "runtime_seconds": runtime_seconds,
            "config_hash": selection["config_hash"],
            "development_data_hash": development_hash,
            "evaluated_at_utc": datetime.now(UTC).isoformat(),
            "test_usage": "one-time confirmation after model selection was locked",
        }
    )

    predictions = final_test[
        [
            "question_id",
            "category",
            "src",
            "text_length",
            "duplicate_group_id",
        ]
    ].copy()
    predictions = predictions.rename(
        columns={
            "question_id": "source_question_id",
            "category": "true_label",
            "src": "source_name",
        }
    )
    predictions["predicted_label"] = predicted
    predictions["confidence"] = confidence
    predictions["decision_score"] = decision_score
    predictions["is_correct"] = (
        predictions["true_label"] == predictions["predicted_label"]
    ).astype(int)
    predictions["run_id"] = selection["run_id"]
    predictions["model_name"] = selected_model
    return EvaluationResult(metrics=metrics, predictions=predictions, estimator=estimator)


def _write_public_tables(
    result: EvaluationResult,
    selection: dict[str, Any],
    run_dir: Path,
    table_dir: Path,
) -> None:
    cv_results = pd.read_csv(run_dir / "cv_results.csv")
    rows: list[dict[str, Any]] = []
    for model_name in ("majority_baseline", "logistic_regression", "linear_svc"):
        candidates = cv_results.loc[cv_results["model_name"] == model_name]
        best = candidates.sort_values("mean_test_score", ascending=False).iloc[0]
        selected = model_name == selection["selected_model"]
        rows.append(
            {
                "model_name": model_name,
                "grouped_cv_macro_f1_mean": best["mean_test_score"],
                "grouped_cv_macro_f1_std": best["std_test_score"],
                "locked_final_model": selected,
                "final_test_macro_f1": result.metrics["macro_f1"]
                if selected
                else "not_evaluated",
                "test_evaluation_policy": "one-time locked confirmation"
                if selected
                else "not evaluated to avoid test-set model selection",
            }
        )
    table_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        table_dir / "model_comparison.csv", index=False, encoding="utf-8"
    )
    per_class = pd.DataFrame.from_dict(result.metrics["per_class"], orient="index")
    per_class.index.name = "subject"
    per_class.reset_index().to_csv(
        table_dir / "per_class_metrics.csv", index=False, encoding="utf-8"
    )


def find_selection_decision(artifacts_dir: Path, run_id: str | None = None) -> Path:
    """Locate an explicit run or the sole locked selection decision."""
    if run_id:
        path = artifacts_dir / "runs" / run_id / "selection_decision.json"
        if not path.exists():
            raise FileNotFoundError(f"selection decision not found: {path}")
        return path
    candidates = sorted((artifacts_dir / "runs").glob("*/selection_decision.json"))
    if len(candidates) != 1:
        raise ValueError("specify --run-id when there is not exactly one locked run")
    return candidates[0]


def evaluate_and_save(
    frame: pd.DataFrame,
    config: ProjectConfig,
    root: Path = Path("."),
    run_id: str | None = None,
    force: bool = False,
) -> tuple[dict[str, Any], str]:
    """Run or reuse locked evaluation and persist reproducible evidence."""
    artifacts_dir = root / config.paths.artifacts
    selection_path = find_selection_decision(artifacts_dir, run_id)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    metrics_path = artifacts_dir / "metrics" / "final_test_metrics.json"
    predictions_path = artifacts_dir / "predictions" / "final_test_predictions.csv"
    if metrics_path.exists() and predictions_path.exists() and not force:
        cached = json.loads(metrics_path.read_text(encoding="utf-8"))
        expected = {
            "run_id": selection["run_id"],
            "config_hash": selection["config_hash"],
            "development_data_hash": selection["development_data_hash"],
        }
        if all(cached.get(key) == value for key, value in expected.items()):
            return cached, "cached"
        raise ValueError("existing final metrics do not match the locked run; use --force")

    result = evaluate_locked_model(selection, frame, config)
    result.metrics["force_requested"] = force
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(metrics_path, result.metrics)
    result.predictions.to_csv(predictions_path, index=False, encoding="utf-8")
    table_dir = root / config.paths.reports / "tables"
    _write_public_tables(result, selection, selection_path.parent, table_dir)
    load_model_run_and_predictions(
        root / config.paths.database,
        selection,
        result.metrics,
        result.predictions,
        config.random_seed,
    )
    export_named_queries(
        root / config.paths.database,
        root / "sql" / "analysis",
        table_dir,
        names={f"{number:02d}" for number in range(6, 11)},
    )
    return result.metrics, "evaluated"
