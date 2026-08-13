from pathlib import Path

import pytest

from stem_analytics.config import load_config
from stem_analytics.reporting import (
    build_evidence_matrix,
    build_report_context,
    presentation_claims,
    verify_numerical_consistency,
)

PUBLIC_PRESENTATION_PATHS = [
    Path("README.md"),
    Path("docs/project_brief.md"),
    Path("docs/interview_guide.md"),
    Path("docs/resume_assets.md"),
]


def test_major_claims_have_authoritative_sources() -> None:
    context = build_report_context(load_config(Path("configs/project.yaml")))
    matrix = build_evidence_matrix(context)
    major = matrix.loc[matrix["claim_priority"] == "major"]
    assert major["authoritative_source"].notna().all()
    assert major["verification_status"].eq("verified").all()


def test_report_rejects_metric_disagreement() -> None:
    context = build_report_context(load_config(Path("configs/project.yaml")))
    context["readme_metrics"]["macro_f1"] += 0.01
    with pytest.raises(ValueError, match="numerical consistency"):
        verify_numerical_consistency(context)


def test_presentation_claims_match_locked_metrics() -> None:
    context = build_report_context(load_config(Path("configs/project.yaml")))
    claims = presentation_claims(context)
    assert claims["macro_f1"] == f"{context['metrics']['macro_f1']:.4f}"
    assert claims["accuracy"] == f"{context['metrics']['accuracy']:.4f}"
    assert claims["test_rows"] == context["metrics"]["final_test_rows"]
    assert claims["crossing_groups"] == 0


def test_presentation_assets_are_utf8_and_have_expected_chinese() -> None:
    for path in PUBLIC_PRESENTATION_PATHS:
        text = path.read_text(encoding="utf-8")
        assert "\ufffd" not in text
    assert "這是一個可重現的端到端機器學習作品集" in Path("README.md").read_text(
        encoding="utf-8"
    )
