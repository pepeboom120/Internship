"""Leakage-safe deterministic project splitting."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from stem_analytics.config import ProjectConfig


def assign_project_split(frame: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    splitter = StratifiedGroupKFold(
        n_splits=config.split.final_test_folds,
        shuffle=True,
        random_state=config.random_seed,
    )
    _, test_indices = next(
        splitter.split(result["question_text"], result["category"], result["duplicate_group_id"])
    )
    result["split_name"] = "development"
    result.loc[test_indices, "split_name"] = "test"
    assert_split_integrity(result)
    return result


def assert_split_integrity(frame: pd.DataFrame) -> None:
    counts = frame.groupby("duplicate_group_id")["split_name"].nunique()
    if (counts > 1).any():
        raise ValueError("duplicate group crosses splits")
