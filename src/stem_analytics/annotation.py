"""Sampling and validation utilities for future human intent annotation."""

from __future__ import annotations

from typing import Any

import pandas as pd

ALLOWED_INTENTS = {
    "conceptual_explanation",
    "calculation_problem_solving",
    "definition_factual_recall",
    "application_interpretation",
    "ambiguous_exclude",
}


def sample_intent_pilot(
    frame: pd.DataFrame, n_per_subject: int = 10, seed: int = 42
) -> pd.DataFrame:
    """Create a deterministic balanced, unlabeled annotation template."""
    rows = []
    for subject, group in frame.groupby("category", sort=True):
        if len(group) < n_per_subject:
            raise ValueError(f"not enough {subject} rows for the requested pilot")
        sampled = group.sample(n=n_per_subject, random_state=seed).sort_values("question_id")
        selected = sampled[["question_id", "question_text"]].copy()
        selected.insert(1, "subject", subject)
        rows.append(selected)
    result = pd.concat(rows, ignore_index=True)
    result["annotator_id"] = ""
    result["intent_label"] = ""
    result["confidence"] = pd.NA
    result["notes"] = ""
    return result


def validate_annotation_sheet(frame: pd.DataFrame) -> dict[str, Any]:
    """Validate labels without implying that annotation has occurred."""
    required = {
        "question_id", "subject", "question_text", "annotator_id",
        "intent_label", "confidence", "notes",
    }
    missing = sorted(required - set(frame.columns))
    errors = []
    if missing:
        errors.append(f"missing columns: {missing}")
    else:
        unknown = sorted(set(frame["intent_label"].dropna()) - ALLOWED_INTENTS)
        if unknown:
            errors.append(f"unknown intent labels: {unknown}")
        if frame["annotator_id"].astype(str).str.strip().eq("").any():
            errors.append("annotator_id must be non-empty")
        numeric_confidence = pd.to_numeric(frame["confidence"], errors="coerce")
        if numeric_confidence.isna().any() or not numeric_confidence.between(1, 3).all():
            errors.append("confidence must be an integer from 1 to 3")
        if frame.duplicated(["question_id", "annotator_id"]).any():
            errors.append("question_id and annotator_id pairs must be unique")
    counts = frame.groupby("question_id").size() if "question_id" in frame else pd.Series(dtype=int)
    return {
        "valid": not errors,
        "errors": errors,
        "rows": len(frame),
        "agreement_ready_questions": int(counts.ge(2).sum()),
        "human_annotation_completed": False,
    }
