from pathlib import Path

import pandas as pd
import pytest

from stem_analytics.config import load_config
from stem_analytics.evaluation import (
    classification_metrics,
    evaluate_locked_model,
    stratified_bootstrap_macro_f1,
)


def test_classification_metrics_reports_macro_and_per_class() -> None:
    result = classification_metrics(
        ["math", "math", "physics", "physics"],
        ["math", "physics", "physics", "physics"],
        ["math", "physics"],
    )
    assert result["accuracy"] == pytest.approx(0.75)
    assert result["macro_f1"] == pytest.approx(((2 / 3) + 0.8) / 2)
    assert set(result["per_class"]) == {"math", "physics"}


def test_bootstrap_interval_is_deterministic() -> None:
    true = ["math", "math", "physics", "physics", "biology", "biology"]
    predicted = ["math", "physics", "physics", "physics", "biology", "math"]
    first = stratified_bootstrap_macro_f1(true, predicted, ["math", "physics", "biology"], 200, 42)
    second = stratified_bootstrap_macro_f1(true, predicted, ["math", "physics", "biology"], 200, 42)
    assert first == second


def test_evaluation_rejects_changed_config() -> None:
    config = load_config(Path("configs/project.yaml"))
    changed = config.model_copy(update={"random_seed": 7})
    frame = pd.DataFrame(
        {"question_text": ["a"], "category": ["math"], "duplicate_group_id": [1], "split_name": ["development"], "question_id": [1], "src": ["fixture"], "text_length": [1]}
    )
    selection = {"config_hash": "does-not-match", "development_data_hash": "x"}
    with pytest.raises(ValueError, match="selection decision does not match"):
        evaluate_locked_model(selection, frame, changed)
