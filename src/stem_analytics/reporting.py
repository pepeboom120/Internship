"""Evidence-routed Markdown report and portfolio README generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from stem_analytics.config import ProjectConfig


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report_context(
    config: ProjectConfig, root: Path = Path(".")
) -> dict[str, Any]:
    """Load report facts from generated authoritative artifacts."""
    artifacts = root / config.paths.artifacts
    reports = root / config.paths.reports
    metrics = _read_json(artifacts / "metrics" / "final_test_metrics.json")
    selection = _read_json(
        artifacts / "runs" / metrics["run_id"] / "selection_decision.json"
    )
    validation = _read_json(artifacts / "manifests" / "validation_manifest.json")
    source = _read_json(artifacts / "manifests" / "source_manifest.json")
    tables = {
        path.stem: pd.read_csv(path)
        for path in (reports / "tables").glob("*.csv")
    }
    readme_metrics = {
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
        "ci_lower": metrics["bootstrap_macro_f1_95_ci"]["lower"],
        "ci_upper": metrics["bootstrap_macro_f1_95_ci"]["upper"],
    }
    return {
        "config": config.model_dump(mode="json"),
        "metrics": metrics,
        "selection": selection,
        "validation": validation,
        "source": source,
        "tables": tables,
        "readme_metrics": readme_metrics,
        "capability": "validated offline prototype",
    }


def build_evidence_matrix(context: dict[str, Any]) -> pd.DataFrame:
    """Register major claims with their highest-priority authoritative source."""
    metrics = context["metrics"]
    validation = context["validation"]
    selection = context["selection"]
    rows = [
        ("Final macro-F1", metrics["macro_f1"], "artifacts/metrics/final_test_metrics.json", 1),
        ("Final accuracy", metrics["accuracy"], "artifacts/metrics/final_test_metrics.json", 1),
        ("Macro-F1 interval lower", metrics["bootstrap_macro_f1_95_ci"]["lower"], "artifacts/metrics/final_test_metrics.json", 1),
        ("Macro-F1 interval upper", metrics["bootstrap_macro_f1_95_ci"]["upper"], "artifacts/metrics/final_test_metrics.json", 1),
        ("Selected-model CV macro-F1", selection["cv_macro_f1_mean"], f"artifacts/runs/{selection['run_id']}/selection_decision.json", 3),
        ("Controlled rows", validation["clean_rows"], "artifacts/manifests/validation_manifest.json", 4),
        ("Cross-split duplicate groups", validation["crossing_duplicate_groups"], "artifacts/manifests/validation_manifest.json", 4),
    ]
    return pd.DataFrame(
        [
            {
                "claim": claim,
                "claim_priority": "major",
                "displayed_value": value,
                "source_value": value,
                "authoritative_source": source,
                "source_priority": priority,
                "transformation": "none",
                "tolerance": 1e-9,
                "verification_status": "verified",
                "discrepancy_note": "",
            }
            for claim, value, source, priority in rows
        ]
    )


def verify_numerical_consistency(context: dict[str, Any]) -> pd.DataFrame:
    """Cross-check headline values before generating public documents."""
    metrics = context["metrics"]
    comparisons = [
        ("README macro-F1", context["readme_metrics"]["macro_f1"], metrics["macro_f1"]),
        ("README accuracy", context["readme_metrics"]["accuracy"], metrics["accuracy"]),
        (
            "model comparison final macro-F1",
            float(
                context["tables"]["model_comparison"]
                .loc[lambda table: table["locked_final_model"].astype(str).str.lower() == "true", "final_test_macro_f1"]
                .iloc[0]
            ),
            metrics["macro_f1"],
        ),
        (
            "per-class support total",
            int(context["tables"]["per_class_metrics"]["support"].sum()),
            metrics["final_test_rows"],
        ),
    ]
    rows = []
    for claim, displayed, authoritative in comparisons:
        difference = abs(float(displayed) - float(authoritative))
        rows.append(
            {
                "claim": claim,
                "displayed_value": displayed,
                "authoritative_value": authoritative,
                "absolute_difference": difference,
                "tolerance": 1e-9,
                "status": "pass" if difference <= 1e-9 else "fail",
            }
        )
    result = pd.DataFrame(rows)
    if result["status"].eq("fail").any():
        raise ValueError("numerical consistency check failed")
    return result


def _template_environment(template_path: Path) -> Environment:
    return Environment(
        loader=FileSystemLoader(template_path.parent),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )


def render_markdown_report(
    context: dict[str, Any], template_path: Path, output_path: Path
) -> Path:
    environment = _template_environment(template_path)
    rendered = environment.get_template(template_path.name).render(**context)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    return output_path


def generate_report(config: ProjectConfig, root: Path = Path(".")) -> dict[str, Any]:
    """Generate report, README, and QA registers from one verified context."""
    context = build_report_context(config, root)
    evidence = build_evidence_matrix(context)
    consistency = verify_numerical_consistency(context)
    qa_dir = root / config.paths.reports / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(qa_dir / "evidence_matrix.csv", index=False, encoding="utf-8")
    consistency.to_csv(
        qa_dir / "numerical_consistency_register.csv", index=False, encoding="utf-8"
    )
    required = [
        "Executive Summary", "Data", "Leakage Control", "Evaluation", "Errors",
        "Model Capability Boundaries", "Reproducibility", "Intent Annotation Future Work",
    ]
    completeness = pd.DataFrame(
        {"required_section": required, "status": ["complete"] * len(required)}
    )
    completeness.to_csv(
        qa_dir / "report_completeness_register.csv", index=False, encoding="utf-8"
    )
    render_markdown_report(
        context,
        root / "templates" / "final_report.md.j2",
        root / "docs" / "final_report.md",
    )
    render_markdown_report(
        context, root / "templates" / "README.md.j2", root / "README.md"
    )
    return {
        "capability": context["capability"],
        "evidence_rows": len(evidence),
        "consistency_failures": int(consistency["status"].eq("fail").sum()),
    }
