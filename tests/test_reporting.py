from pathlib import Path

import pytest

from stem_analytics.config import load_config
from stem_analytics.reporting import (
    build_evidence_matrix,
    build_report_context,
    verify_numerical_consistency,
)


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
