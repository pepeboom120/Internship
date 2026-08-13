from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from stem_analytics.error_analysis import (
    build_error_slices,
    confidence_coverage_curve,
    extract_linear_features,
    representative_cases,
)


def prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_question_id": [1, 2, 3, 4],
            "true_label": ["math", "math", "physics", "physics"],
            "predicted_label": ["math", "physics", "physics", "physics"],
            "confidence": [0.95, 0.55, 0.85, 0.65],
            "decision_score": [2.0, 0.2, 1.5, 0.5],
            "is_correct": [1, 0, 1, 1],
            "source_name": ["a", "a", "b", "b"],
            "text_length": [50, 150, 250, 450],
            "question_text": ["one", "two", "three", "four"],
        }
    )


def test_error_slices_include_subject_source_and_length() -> None:
    frame = prediction_frame()
    slices = build_error_slices(frame)
    assert set(slices["slice_type"]) >= {"subject", "source", "length_band"}
    assert slices["n"].sum() >= len(frame)


def test_confidence_threshold_reduces_coverage() -> None:
    frame = prediction_frame()
    curve = confidence_coverage_curve(frame, [0.0, 0.7, 0.9])
    assert curve["automated_coverage"].is_monotonic_decreasing
    assert (
        curve["manual_review_volume"] == len(frame) - curve["automated_count"]
    ).all()


def test_feature_table_has_both_directions() -> None:
    text = [
        "algebra number equation",
        "geometry number proof",
        "force energy motion",
        "velocity energy force",
    ]
    target = ["math", "math", "physics", "physics"]
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    ).fit(text, target)
    table = extract_linear_features(pipeline, ["math", "physics"], top_n=3)
    assert set(table["direction"]) == {"positive", "negative"}
    assert table.groupby(["subject", "direction"]).size().eq(3).all()


def test_representative_cases_caps_question_excerpt() -> None:
    frame = prediction_frame()
    frame.loc[0, "question_text"] = "x" * 300
    cases = representative_cases(frame, per_outcome=2)
    assert cases["question_excerpt"].str.len().max() <= 240
    assert set(cases["outcome"]) == {"correct", "incorrect"}


def test_module_does_not_write_to_current_directory() -> None:
    before = set(Path(".").iterdir())
    build_error_slices(prediction_frame())
    assert set(Path(".").iterdir()) == before
