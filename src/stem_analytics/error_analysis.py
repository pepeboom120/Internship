"""Post-lock diagnostics for errors, features, data sufficiency, and robustness."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str((Path("artifacts") / "matplotlib-cache").resolve())
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import f1_score, recall_score

from stem_analytics.analysis import FigureRecord
from stem_analytics.config import ProjectConfig
from stem_analytics.modeling import build_cv, build_pipeline
from stem_analytics.provenance import sha256_file

LENGTH_BINS = [-np.inf, 100, 200, 400, np.inf]
LENGTH_LABELS = ["<100", "100-199", "200-399", "400+"]


def _slice_rows(frame: pd.DataFrame, column: str, slice_type: str) -> pd.DataFrame:
    grouped = frame.groupby(column, observed=False, dropna=False)["is_correct"]
    result = grouped.agg(n="size", correct="sum").reset_index()
    result["errors"] = result["n"] - result["correct"]
    result["accuracy"] = result["correct"] / result["n"]
    result["error_rate"] = result["errors"] / result["n"]
    result.insert(0, "slice_type", slice_type)
    result = result.rename(columns={column: "slice_value"})
    return result[
        ["slice_type", "slice_value", "n", "correct", "errors", "accuracy", "error_rate"]
    ]


def build_error_slices(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize correctness by true subject, source, and fixed length band."""
    frame = predictions.copy()
    frame["length_band"] = pd.cut(
        frame["text_length"], bins=LENGTH_BINS, labels=LENGTH_LABELS, right=False
    )
    return pd.concat(
        [
            _slice_rows(frame, "true_label", "subject"),
            _slice_rows(frame, "source_name", "source"),
            _slice_rows(frame, "length_band", "length_band"),
        ],
        ignore_index=True,
    )


def confidence_coverage_curve(
    predictions: pd.DataFrame, thresholds: Sequence[float]
) -> pd.DataFrame:
    """Calculate automated coverage and accuracy above probability thresholds."""
    valid = predictions.loc[predictions["confidence"].notna()].copy()
    if valid.empty:
        return pd.DataFrame(
            [
                {
                    "threshold": np.nan,
                    "automated_count": 0,
                    "automated_coverage": 0.0,
                    "manual_review_volume": len(predictions),
                    "automated_accuracy": np.nan,
                    "status": "not_available_locked_model_has_no_calibrated_probability",
                }
            ]
        )
    rows = []
    for threshold in thresholds:
        automated = valid.loc[valid["confidence"] >= threshold]
        rows.append(
            {
                "threshold": float(threshold),
                "automated_count": len(automated),
                "automated_coverage": len(automated) / len(predictions),
                "manual_review_volume": len(predictions) - len(automated),
                "automated_accuracy": automated["is_correct"].mean()
                if len(automated)
                else np.nan,
                "status": "available",
            }
        )
    return pd.DataFrame(rows)


def _round_robin_cases(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    selected: list[int] = []
    grouped = {
        label: group.index.tolist()
        for label, group in frame.groupby("true_label", sort=True)
    }
    while len(selected) < count and any(grouped.values()):
        for label in sorted(grouped):
            if grouped[label] and len(selected) < count:
                selected.append(grouped[label].pop(0))
    return frame.loc[selected]


def representative_cases(
    predictions: pd.DataFrame, per_outcome: int = 5
) -> pd.DataFrame:
    """Select deterministic, class-diverse correct and incorrect examples."""
    rows = []
    for is_correct, outcome in ((0, "incorrect"), (1, "correct")):
        subset = predictions.loc[predictions["is_correct"] == is_correct].copy()
        score_column = (
            "confidence" if subset["confidence"].notna().any() else "decision_score"
        )
        subset = subset.sort_values(
            ["true_label", score_column, "source_question_id"],
            ascending=[True, is_correct == 0, True],
            na_position="last",
        )
        selected = _round_robin_cases(subset, min(per_outcome, len(subset))).copy()
        selected["outcome"] = outcome
        rows.append(selected)
    result = pd.concat(rows, ignore_index=True)
    result["question_excerpt"] = result["question_text"].astype(str).str.slice(0, 240)
    columns = [
        "outcome",
        "source_question_id",
        "true_label",
        "predicted_label",
        "source_name",
        "text_length",
        "confidence",
        "decision_score",
        "question_excerpt",
    ]
    return result[columns]


def extract_linear_features(
    pipeline: Any,
    labels: Sequence[str],
    top_n: int = 20,
    source_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Map linear coefficients back to TF-IDF features in both directions."""
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    features = np.asarray(vectorizer.get_feature_names_out())
    coefficients = np.asarray(classifier.coef_)
    classes = list(classifier.classes_)
    if coefficients.shape[0] == 1 and len(classes) == 2:
        coefficients = np.vstack([-coefficients[0], coefficients[0]])

    source_dominance: dict[str, tuple[str, float]] = {}
    normalized_sources: set[str] = set()
    if source_frame is not None and not source_frame.empty:
        matrix = vectorizer.transform(source_frame["question_text"])
        source_means = []
        source_names = []
        for source, indices in source_frame.groupby("src").groups.items():
            source_names.append(str(source))
            normalized_sources.add(str(source).lower().replace("_", " ").replace("-", " "))
            source_means.append(np.asarray(matrix[list(indices)].mean(axis=0)).ravel())
        means = np.vstack(source_means)
        denominators = means.sum(axis=0)
        ratios = np.divide(
            means.max(axis=0), denominators, out=np.zeros_like(denominators), where=denominators > 0
        )
        maxima = means.argmax(axis=0)
        source_dominance = {
            feature: (source_names[int(maxima[index])], float(ratios[index]))
            for index, feature in enumerate(features)
        }

    rows = []
    for label in labels:
        class_index = classes.index(label)
        weights = coefficients[class_index]
        for direction, indices in (
            ("positive", np.argsort(weights)[-top_n:][::-1]),
            ("negative", np.argsort(weights)[:top_n]),
        ):
            for rank, feature_index in enumerate(indices, start=1):
                feature = str(features[feature_index])
                source, dominance = source_dominance.get(feature, ("", np.nan))
                rows.append(
                    {
                        "subject": label,
                        "direction": direction,
                        "rank": rank,
                        "feature": feature,
                        "coefficient": float(weights[feature_index]),
                        "exact_source_name_match": feature in normalized_sources,
                        "dominant_source": source,
                        "source_dominance_ratio": dominance,
                        "possible_source_shortcut": bool(dominance >= 0.8)
                        if not np.isnan(dominance)
                        else False,
                    }
                )
    return pd.DataFrame(rows)


def _locked_estimator(selection: dict[str, Any], config: ProjectConfig) -> Any:
    estimator = build_pipeline(selection["selected_model"], config.random_seed)
    parameters = dict(selection["best_params"])
    if isinstance(parameters.get("tfidf__ngram_range"), list):
        parameters["tfidf__ngram_range"] = tuple(parameters["tfidf__ngram_range"])
    return estimator.set_params(**parameters)


def _stratified_group_subset(
    frame: pd.DataFrame, fraction: float, seed: int
) -> pd.DataFrame:
    if fraction >= 1:
        return frame
    groups = frame[["duplicate_group_id", "category"]].drop_duplicates()
    selected = []
    generator = np.random.default_rng(seed)
    for _, subject_groups in groups.groupby("category"):
        count = max(1, round(len(subject_groups) * fraction))
        selected.extend(
            generator.choice(
                subject_groups["duplicate_group_id"].to_numpy(), count, replace=False
            ).tolist()
        )
    return frame.loc[frame["duplicate_group_id"].isin(selected)]


def compute_learning_curve(
    frame: pd.DataFrame, selection: dict[str, Any], config: ProjectConfig
) -> pd.DataFrame:
    """Estimate data sufficiency using grouped folds on development data only."""
    development = frame.loc[frame["split_name"] == "development"].reset_index(drop=True)
    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    scores: dict[float, list[tuple[int, float, float]]] = {value: [] for value in fractions}
    cv = build_cv(config)
    for fold, (train_indices, valid_indices) in enumerate(
        cv.split(
            development["question_text"],
            development["category"],
            development["duplicate_group_id"],
        )
    ):
        fold_train = development.iloc[train_indices]
        fold_valid = development.iloc[valid_indices]
        for fraction in fractions:
            sampled = _stratified_group_subset(
                fold_train, fraction, config.random_seed + fold
            )
            estimator = clone(_locked_estimator(selection, config))
            estimator.fit(sampled["question_text"], sampled["category"])
            train_prediction = estimator.predict(sampled["question_text"])
            valid_prediction = estimator.predict(fold_valid["question_text"])
            scores[fraction].append(
                (
                    len(sampled),
                    f1_score(
                        sampled["category"], train_prediction, average="macro", zero_division=0
                    ),
                    f1_score(
                        fold_valid["category"], valid_prediction, average="macro", zero_division=0
                    ),
                )
            )
    rows = []
    for fraction, values in scores.items():
        array = np.asarray(values)
        rows.append(
            {
                "development_fraction": fraction,
                "mean_training_rows": float(array[:, 0].mean()),
                "train_macro_f1_mean": float(array[:, 1].mean()),
                "train_macro_f1_std": float(array[:, 1].std()),
                "validation_macro_f1_mean": float(array[:, 2].mean()),
                "validation_macro_f1_std": float(array[:, 2].std()),
                "cv_folds": len(values),
            }
        )
    return pd.DataFrame(rows)


def run_source_holdout(
    frame: pd.DataFrame, selection: dict[str, Any], config: ProjectConfig
) -> pd.DataFrame:
    """Test eligible multi-class sources, or record why the test is infeasible."""
    development = frame.loc[frame["split_name"] == "development"].reset_index(drop=True)
    eligible = [
        source
        for source, group in development.groupby("src")
        if len(group) >= 80 and set(config.subjects).issubset(set(group["category"]))
    ]
    if not eligible:
        return pd.DataFrame(
            [
                {
                    "status": "not_feasible",
                    "held_out_source": "",
                    "n": 0,
                    "macro_f1": np.nan,
                    "per_class_recall": "{}",
                    "reason": (
                        "No development source has at least 80 rows and all four target "
                        "subjects; source is strongly confounded with subject."
                    ),
                }
            ]
        )
    rows = []
    for source in eligible:
        holdout = development.loc[development["src"] == source]
        training = development.loc[development["src"] != source]
        estimator = clone(_locked_estimator(selection, config))
        estimator.fit(training["question_text"], training["category"])
        predicted = estimator.predict(holdout["question_text"])
        recalls = recall_score(
            holdout["category"],
            predicted,
            labels=config.subjects,
            average=None,
            zero_division=0,
        )
        rows.append(
            {
                "status": "evaluated",
                "held_out_source": source,
                "n": len(holdout),
                "macro_f1": f1_score(
                    holdout["category"],
                    predicted,
                    labels=config.subjects,
                    average="macro",
                    zero_division=0,
                ),
                "per_class_recall": json.dumps(
                    dict(zip(config.subjects, recalls.tolist(), strict=True)),
                    sort_keys=True,
                ),
                "reason": "",
            }
        )
    return pd.DataFrame(rows)


def _finish_figure(ax: plt.Axes, title: str, sample_size: int, path: Path) -> None:
    ax.set_title(f"{title} (n = {sample_size:,})", loc="left", weight="bold")
    ax.figure.text(
        0.10,
        0.02,
        "MMLU-Pro benchmark questions; final-test results are from the locked model only.",
        fontsize=8,
        color="#555555",
    )
    ax.figure.tight_layout(rect=(0, 0.08, 1, 1))
    ax.figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(ax.figure)


def build_model_figures(
    tables: dict[str, pd.DataFrame],
    metrics: dict[str, Any],
    output_dir: Path,
) -> list[FigureRecord]:
    """Render result figures 06 through 12 from persisted diagnostic tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    n = int(metrics["final_test_rows"])
    records: list[FigureRecord] = []

    comparison = tables["model_comparison"]
    path = output_dir / "06_model_comparison.png"
    _, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=comparison,
        x="model_name",
        y="grouped_cv_macro_f1_mean",
        color="#4477AA",
        ax=ax,
    )
    ax.errorbar(
        range(len(comparison)),
        comparison["grouped_cv_macro_f1_mean"],
        yerr=comparison["grouped_cv_macro_f1_std"],
        fmt="none",
        color="black",
        capsize=4,
    )
    ax.set(xlabel="Development-only candidate", ylabel="Grouped-CV macro-F1", ylim=(0, 1))
    ax.tick_params(axis="x", rotation=15)
    title = "LinearSVC led development-only model selection"
    _finish_figure(ax, title, int(metrics["development_rows"]), path)
    records.append(FigureRecord(6, title, path, "model_comparison.csv", int(metrics["development_rows"])))

    path = output_dir / "07_confusion_matrix.png"
    matrix = np.asarray(metrics["confusion_matrix"])
    _, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=metrics["label_order"], yticklabels=metrics["label_order"], ax=ax)
    ax.set(xlabel="Predicted subject", ylabel="True subject")
    title = "Most final-test questions lie on the confusion-matrix diagonal"
    _finish_figure(ax, title, n, path)
    records.append(FigureRecord(7, title, path, "final_test_metrics.json", n))

    per_class = tables["per_class_metrics"].melt(
        id_vars=["subject", "support"],
        value_vars=["precision", "recall", "f1"],
        var_name="metric",
        value_name="score",
    )
    path = output_dir / "08_per_class_metrics.png"
    _, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=per_class, x="subject", y="score", hue="metric", ax=ax)
    ax.set(xlabel="Subject", ylabel="Final-test score", ylim=(0, 1))
    title = "Chemistry is the weakest final-test class"
    _finish_figure(ax, title, n, path)
    records.append(FigureRecord(8, title, path, "per_class_metrics.csv", n))

    folds = tables["cv_fold_variation"]
    path = output_dir / "09_cv_fold_variation.png"
    _, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=folds, x="model_name", y="macro_f1", color="#66CCEE", ax=ax)
    sns.stripplot(data=folds, x="model_name", y="macro_f1", color="#222222", ax=ax)
    ax.set(xlabel="Tuned candidate", ylabel="Grouped-fold macro-F1", ylim=(0.82, 0.92))
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.2f}")
    title = "Candidate performance is stable across grouped folds"
    _finish_figure(ax, title, int(metrics["development_rows"]), path)
    records.append(FigureRecord(9, title, path, "cv_fold_variation.csv", int(metrics["development_rows"])))

    slices = tables["error_slices"]
    display_slices = slices.loc[slices["slice_type"].isin(["subject", "length_band"])]
    path = output_dir / "10_error_slices.png"
    _, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=display_slices, x="slice_value", y="error_rate", hue="slice_type", ax=ax)
    ax.set(xlabel="Final-test slice", ylabel="Error rate", ylim=(0, max(0.25, display_slices["error_rate"].max() * 1.15)))
    ax.tick_params(axis="x", rotation=20)
    title = "Errors vary by subject and question length"
    _finish_figure(ax, title, n, path)
    records.append(FigureRecord(10, title, path, "error_slices.csv", n))

    curve = tables["confidence_coverage"]
    path = output_dir / "11_confidence_coverage.png"
    _, ax = plt.subplots(figsize=(9, 5))
    available = curve.loc[curve["status"] == "available"]
    if available.empty:
        ax.axis("off")
        ax.text(0.5, 0.58, "Not available", ha="center", va="center", fontsize=20, weight="bold")
        ax.text(0.5, 0.42, "Locked LinearSVC margins are not calibrated probabilities.\nA calibration protocol is required before threshold automation.", ha="center", va="center", fontsize=11)
        title = "Confidence/coverage is intentionally not claimed"
    else:
        ax.plot(available["automated_coverage"], available["automated_accuracy"], marker="o")
        ax.set(xlabel="Automated coverage", ylabel="Accuracy above threshold", xlim=(0, 1), ylim=(0, 1))
        title = "Higher confidence trades coverage for accuracy"
    _finish_figure(ax, title, n, path)
    records.append(FigureRecord(11, title, path, "confidence_coverage.csv", n))

    learning = tables["learning_curve"]
    path = output_dir / "12_learning_curve.png"
    _, ax = plt.subplots(figsize=(9, 5))
    ax.plot(learning["mean_training_rows"], learning["train_macro_f1_mean"], marker="o", label="Training")
    ax.plot(learning["mean_training_rows"], learning["validation_macro_f1_mean"], marker="o", label="Validation")
    ax.fill_between(learning["mean_training_rows"], learning["validation_macro_f1_mean"] - learning["validation_macro_f1_std"], learning["validation_macro_f1_mean"] + learning["validation_macro_f1_std"], alpha=0.2)
    ax.set(xlabel="Mean training rows per grouped fold", ylabel="Macro-F1", ylim=(0, 1))
    ax.legend()
    title = "Validation performance improves as development data grows"
    _finish_figure(ax, title, int(metrics["development_rows"]), path)
    records.append(FigureRecord(12, title, path, "learning_curve.csv", int(metrics["development_rows"])))
    return records


def _cv_fold_table(cv_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    split_columns = [column for column in cv_results if column.startswith("split") and column.endswith("_test_score")]
    for model_name in ("logistic_regression", "linear_svc"):
        best = cv_results.loc[cv_results["model_name"] == model_name].sort_values("mean_test_score", ascending=False).iloc[0]
        for fold, column in enumerate(split_columns):
            rows.append({"model_name": model_name, "fold": fold, "macro_f1": best[column]})
    return pd.DataFrame(rows)


def update_figure_register(records: list[FigureRecord], register_path: Path) -> None:
    """Replace model-figure records while retaining verified EDA records."""
    existing = pd.read_csv(register_path) if register_path.exists() else pd.DataFrame()
    if not existing.empty:
        existing = existing.loc[existing["number"] < 6]
    rows = []
    for record in records:
        row = asdict(record)
        row["path"] = record.path.as_posix()
        row["figure_sha256"] = sha256_file(record.path)
        rows.append(row)
    combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    combined.sort_values("number").to_csv(register_path, index=False, encoding="utf-8")


def diagnose_and_save(
    frame: pd.DataFrame,
    selection: dict[str, Any],
    predictions: pd.DataFrame,
    metrics: dict[str, Any],
    config: ProjectConfig,
    root: Path = Path("."),
) -> dict[str, Any]:
    """Generate all public-safe diagnostic tables and figures."""
    development = frame.loc[frame["split_name"] == "development"].reset_index(drop=True)
    enriched = predictions.merge(
        frame[["question_id", "question_text"]],
        left_on="source_question_id",
        right_on="question_id",
        how="left",
        validate="one_to_one",
    ).drop(columns="question_id")
    estimator = _locked_estimator(selection, config)
    estimator.fit(development["question_text"], development["category"])

    run_dir = root / config.paths.artifacts / "runs" / selection["run_id"]
    cv_results = pd.read_csv(run_dir / "cv_results.csv")
    table_dir = root / config.paths.reports / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "error_slices": build_error_slices(enriched),
        "representative_cases": representative_cases(enriched),
        "top_features": extract_linear_features(
            estimator, config.subjects, top_n=20, source_frame=development
        ),
        "confidence_coverage": confidence_coverage_curve(
            enriched, [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        ),
        "learning_curve": compute_learning_curve(frame, selection, config),
        "source_holdout_results": run_source_holdout(frame, selection, config),
        "cv_fold_variation": _cv_fold_table(cv_results),
        "model_comparison": pd.read_csv(table_dir / "model_comparison.csv"),
        "per_class_metrics": pd.read_csv(table_dir / "per_class_metrics.csv"),
    }
    for name, table in tables.items():
        table.to_csv(table_dir / f"{name}.csv", index=False, encoding="utf-8")
    records = build_model_figures(
        tables, metrics, root / config.paths.reports / "figures"
    )
    update_figure_register(
        records, root / config.paths.reports / "qa" / "figure_source_register.csv"
    )
    return {
        "diagnostic_tables": len(tables),
        "model_figures": len(records),
        "source_holdout_status": tables["source_holdout_results"].iloc[0]["status"],
    }
