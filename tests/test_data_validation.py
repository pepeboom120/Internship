from pathlib import Path

import pandas as pd

from stem_analytics.config import load_config
from stem_analytics.data_validation import normalize_question, validate_and_clean


def test_normalization_preserves_formula_punctuation() -> None:
    assert normalize_question("  If  E = mc²,\n why?  ") == "If E = mc², why?"


def test_validation_records_exclusion_reasons() -> None:
    frame = pd.DataFrame(
        [
            {"question_id": 1, "question": "Valid?", "category": "math", "src": "fixture"},
            {"question_id": 2, "question": " ", "category": "math", "src": "fixture"},
            {"question_id": 3, "question": "Invalid", "category": "history", "src": "fixture"},
            {"question_id": 1, "question": "Repeated id", "category": "math", "src": "fixture"},
            {"question_id": 4, "question": "Missing source", "category": "math", "src": " "},
        ]
    )
    clean, excluded = validate_and_clean(frame, load_config(Path("configs/project.yaml")))
    assert len(clean) == 1
    assert set(excluded["exclusion_reason"]) == {
        "empty_question",
        "invalid_subject",
        "duplicate_question_id",
        "missing_source",
    }
