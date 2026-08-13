"""Leakage-safe baselines, text pipelines, and grouped model selection."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from stem_analytics.config import ProjectConfig
from stem_analytics.provenance import canonical_sha256, config_hash, sha256_file, write_json_atomic

ModelName = Literal["logistic_regression", "linear_svc"]


@dataclass
class ModelSelectionResult:
    run_id: str
    selected_model: str
    best_estimator: Pipeline
    selection_decision: dict[str, Any]
    cv_results: pd.DataFrame


def prepare_xyg(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    return frame["question_text"], frame["category"], frame["duplicate_group_id"]


def build_pipeline(model_name: ModelName, seed: int = 42) -> Pipeline:
    if model_name == "logistic_regression":
        classifier = LogisticRegression(max_iter=5000, random_state=seed)
    elif model_name == "linear_svc":
        classifier = LinearSVC(random_state=seed)
    else:
        raise ValueError(f"unknown model: {model_name}")
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(sublinear_tf=True, strip_accents=None, lowercase=True)),
            ("classifier", classifier),
        ]
    )


def parameter_grid(model_name: ModelName, config: ProjectConfig) -> dict[str, list[Any]]:
    if model_name not in {"logistic_regression", "linear_svc"}:
        raise ValueError(f"unknown model: {model_name}")
    return {
        "tfidf__ngram_range": list(config.models.tfidf_ngram_ranges),
        "tfidf__min_df": config.models.tfidf_min_df,
        "classifier__C": config.models.c_values,
        "classifier__class_weight": config.models.class_weights,
    }


def build_cv(config: ProjectConfig) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(
        n_splits=config.split.cv_folds,
        shuffle=True,
        random_state=config.random_seed,
    )


def _baseline_scores(
    x: pd.Series, y: pd.Series, groups: pd.Series, config: ProjectConfig
) -> list[float]:
    scores = []
    for train_index, valid_index in build_cv(config).split(x, y, groups):
        model = DummyClassifier(strategy="most_frequent")
        model.fit(x.iloc[train_index].to_numpy().reshape(-1, 1), y.iloc[train_index])
        predicted = model.predict(x.iloc[valid_index].to_numpy().reshape(-1, 1))
        scores.append(f1_score(y.iloc[valid_index], predicted, average="macro"))
    return scores


def _oof_recall(
    estimator: Pipeline,
    x: pd.Series,
    y: pd.Series,
    groups: pd.Series,
    config: ProjectConfig,
) -> dict[str, float]:
    predicted = pd.Series(index=y.index, dtype="string")
    for train_index, valid_index in build_cv(config).split(x, y, groups):
        fold_model = clone(estimator)
        fold_model.fit(x.iloc[train_index], y.iloc[train_index])
        predicted.iloc[valid_index] = fold_model.predict(x.iloc[valid_index])
    recalls = recall_score(
        y,
        predicted,
        labels=config.subjects,
        average=None,
        zero_division=0,
    )
    return {subject: float(value) for subject, value in zip(config.subjects, recalls, strict=True)}


def fit_candidates(frame: pd.DataFrame, config: ProjectConfig) -> ModelSelectionResult:
    development = frame.loc[frame["split_name"] == "development"].reset_index(drop=True)
    x, y, groups = prepare_xyg(development)
    baseline = _baseline_scores(x, y, groups, config)
    result_frames = [
        pd.DataFrame(
            {
                "model_name": ["majority_baseline"],
                "mean_test_score": [float(np.mean(baseline))],
                "std_test_score": [float(np.std(baseline))],
                "mean_fit_time": [0.0],
                "params": ["{}"],
                "rank_test_score": [1],
            }
        )
    ]
    searches: dict[str, GridSearchCV] = {}
    elapsed: dict[str, float] = {}
    for model_name in ("logistic_regression", "linear_svc"):
        search = GridSearchCV(
            build_pipeline(model_name, config.random_seed),
            parameter_grid(model_name, config),
            scoring="f1_macro",
            cv=build_cv(config),
            n_jobs=-1,
            return_train_score=True,
            refit=True,
        )
        started = time.perf_counter()
        search.fit(x, y, groups=groups)
        elapsed[model_name] = time.perf_counter() - started
        searches[model_name] = search
        table = pd.DataFrame(search.cv_results_)
        table.insert(0, "model_name", model_name)
        result_frames.append(table)

    selected_model = max(searches, key=lambda name: searches[name].best_score_)
    selected_search = searches[selected_model]
    per_class_recall = _oof_recall(
        selected_search.best_estimator_, x, y, groups, config
    )
    data_hash = canonical_sha256(
        development[["question_text", "category", "duplicate_group_id"]].to_dict("records")
    )
    run_id = canonical_sha256(
        {"config": config_hash(config), "development_data": data_hash}
    )[:12]
    decision = {
        "run_id": run_id,
        "selected_model": selected_model,
        "best_params": selected_search.best_params_,
        "cv_macro_f1_mean": float(selected_search.best_score_),
        "cv_macro_f1_std": float(
            pd.DataFrame(selected_search.cv_results_)
            .loc[selected_search.best_index_, "std_test_score"]
        ),
        "cv_per_class_recall": per_class_recall,
        "search_runtime_seconds": elapsed[selected_model],
        "config_hash": config_hash(config),
        "development_data_hash": data_hash,
        "selection_rationale": (
            "Selected by the highest grouped five-fold cross-validated macro-F1; "
            "the final test set was not inspected."
        ),
    }
    return ModelSelectionResult(
        run_id=run_id,
        selected_model=selected_model,
        best_estimator=selected_search.best_estimator_,
        selection_decision=decision,
        cv_results=pd.concat(result_frames, ignore_index=True),
    )


def train_and_save(
    frame: pd.DataFrame, config: ProjectConfig, root: Path = Path(".")
) -> ModelSelectionResult:
    result = fit_candidates(frame, config)
    run_dir = root / config.paths.artifacts / "runs" / result.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cv_path = run_dir / "cv_results.csv"
    decision_path = run_dir / "selection_decision.json"
    model_path = run_dir / "selected_model.joblib"
    result.cv_results.to_csv(cv_path, index=False, encoding="utf-8")
    result.selection_decision["cv_results_sha256"] = sha256_file(cv_path)
    write_json_atomic(decision_path, result.selection_decision)
    joblib.dump(result.best_estimator, model_path)
    return result
