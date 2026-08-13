from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from stem_analytics.config import load_config
from stem_analytics.modeling import (
    build_cv,
    build_pipeline,
    parameter_grid,
    prepare_xyg,
)


def modeling_frame() -> pd.DataFrame:
    rows = []
    for subject in ["math", "physics", "chemistry", "biology"]:
        for index in range(10):
            rows.append(
                {
                    "question_text": f"Unique {subject} vocabulary question {index}",
                    "category": subject,
                    "duplicate_group_id": len(rows),
                }
            )
    return pd.DataFrame(rows)


def test_pipeline_fits_tfidf_inside_pipeline() -> None:
    pipeline = build_pipeline("logistic_regression", seed=42)
    assert list(pipeline.named_steps) == ["tfidf", "classifier"]
    assert isinstance(pipeline.named_steps["tfidf"], TfidfVectorizer)


def test_training_uses_question_text_only() -> None:
    frame = modeling_frame()
    x, y, groups = prepare_xyg(frame)
    assert x.name == "question_text"
    assert x.tolist() == frame["question_text"].tolist()
    assert y.tolist() == frame["category"].tolist()
    assert groups.tolist() == frame["duplicate_group_id"].tolist()


def test_parameter_grid_matches_predeclared_space() -> None:
    config = load_config(Path("configs/project.yaml"))
    grid = parameter_grid("linear_svc", config)
    assert grid["tfidf__ngram_range"] == [(1, 1), (1, 2)]
    assert grid["tfidf__min_df"] == [1, 2, 5]
    assert grid["classifier__C"] == [0.1, 1.0, 10.0]
    assert grid["classifier__class_weight"] == [None, "balanced"]


def test_grouped_cv_never_splits_duplicate_group() -> None:
    config = load_config(Path("configs/project.yaml"))
    frame = modeling_frame()
    frame.loc[1, "duplicate_group_id"] = frame.loc[0, "duplicate_group_id"]
    x, y, groups = prepare_xyg(frame)
    for train_index, valid_index in build_cv(config).split(x, y, groups):
        assert set(groups.iloc[train_index]).isdisjoint(set(groups.iloc[valid_index]))
