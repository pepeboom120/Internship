import pandas as pd

from stem_analytics.annotation import sample_intent_pilot, validate_annotation_sheet


def clean_frame() -> pd.DataFrame:
    rows = []
    for subject in ["math", "physics", "chemistry", "biology"]:
        for index in range(4):
            rows.append(
                {
                    "question_id": len(rows) + 1,
                    "category": subject,
                    "question_text": f"{subject} question {index}",
                }
            )
    return pd.DataFrame(rows)


def test_intent_pilot_is_balanced_and_deterministic() -> None:
    first = sample_intent_pilot(clean_frame(), n_per_subject=2, seed=42)
    second = sample_intent_pilot(clean_frame(), n_per_subject=2, seed=42)
    assert first["question_id"].tolist() == second["question_id"].tolist()
    assert first.groupby("subject").size().eq(2).all()


def test_annotation_validation_rejects_unknown_intent() -> None:
    frame = sample_intent_pilot(clean_frame(), n_per_subject=1, seed=42)
    frame["annotator_id"] = "reviewer-a"
    frame["intent_label"] = "conceptual_explanation"
    frame["confidence"] = 3
    frame.loc[0, "intent_label"] = "unknown"
    result = validate_annotation_sheet(frame)
    assert result["valid"] is False
