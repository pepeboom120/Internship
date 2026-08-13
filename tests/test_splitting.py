from pathlib import Path

import pandas as pd
import pytest

from stem_analytics.config import load_config
from stem_analytics.data_validation import assign_duplicate_groups
from stem_analytics.splitting import assert_split_integrity, assign_project_split


def test_near_duplicates_share_group() -> None:
    frame = pd.DataFrame(
        {
            "question_id": [101, 102, 103],
            "question_text": [
                "Calculate the acceleration of this object under constant force",
                "Calculate acceleration of this object under a constant force",
                "Which molecule carries genetic information?",
            ],
            "category": ["physics", "physics", "biology"],
        }
    )
    grouped = assign_duplicate_groups(frame, load_config(Path("configs/project.yaml")))
    assert grouped.loc[grouped["question_id"].isin([101, 102]), "duplicate_group_id"].nunique() == 1
    assert grouped.loc[grouped["question_id"].isin([101, 102]), "duplicate_group_max_similarity"].min() >= 0.92


def test_grouped_split_has_no_contamination() -> None:
    rows = []
    for subject in ["math", "physics", "chemistry", "biology"]:
        for index in range(8):
            rows.append({"question_id": len(rows), "question_text": f"Unique {subject} question {index}", "category": subject, "duplicate_group_id": len(rows)})
    split = assign_project_split(pd.DataFrame(rows), load_config(Path("configs/project.yaml")))
    assert set(split["split_name"]) == {"development", "test"}
    assert_split_integrity(split)


def test_integrity_rejects_group_contamination() -> None:
    frame = pd.DataFrame({"duplicate_group_id": [1, 1], "split_name": ["development", "test"]})
    with pytest.raises(ValueError, match="duplicate group crosses splits"):
        assert_split_integrity(frame)
