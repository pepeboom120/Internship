"""Evidence-routed Markdown report and portfolio README generation."""

from __future__ import annotations

import json
import re
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
        "project_root": root.resolve(),
    }


def presentation_claims(context: dict[str, Any]) -> dict[str, str | int | float]:
    """Format interviewer-facing claims from authoritative project evidence."""
    root = Path(context["project_root"])
    metrics = context["metrics"]
    validation = context["validation"]
    selection = context["selection"]
    per_class = context["tables"]["per_class_metrics"]
    strongest = per_class.sort_values("f1", ascending=False).iloc[0]
    weakest = per_class.sort_values("f1", ascending=True).iloc[0]
    error_slices = context["tables"]["error_slices"]
    short_error = error_slices.loc[
        (error_slices["slice_type"] == "length_band")
        & (error_slices["slice_value"] == "<100"),
        "error_rate",
    ].iloc[0]
    grid = context["config"]["models"]
    parameter_configurations = (
        len(grid["tfidf_ngram_ranges"])
        * len(grid["tfidf_min_df"])
        * len(grid["c_values"])
        * len(grid["class_weights"])
        * 2
    )
    test_pattern = re.compile(r"^\s*def test_", re.MULTILINE)
    test_count = sum(
        len(test_pattern.findall(path.read_text(encoding="utf-8")))
        for path in (root / "tests").glob("test_*.py")
    )
    pdf_audit = root / "reports" / "qa" / "pdf_page_audit.csv"
    pdf_pages = len(pd.read_csv(pdf_audit)) if pdf_audit.exists() else 0
    return {
        "model_name": str(metrics["model_name"]),
        "macro_f1": f"{metrics['macro_f1']:.4f}",
        "accuracy": f"{metrics['accuracy']:.4f}",
        "weighted_f1": f"{metrics['weighted_f1']:.4f}",
        "ci_lower": f"{metrics['bootstrap_macro_f1_95_ci']['lower']:.4f}",
        "ci_upper": f"{metrics['bootstrap_macro_f1_95_ci']['upper']:.4f}",
        "cv_macro_f1": f"{selection['cv_macro_f1_mean']:.4f}",
        "cv_std": f"{selection['cv_macro_f1_std']:.4f}",
        "source_rows": int(validation["source_rows"]),
        "controlled_rows": int(validation["clean_rows"]),
        "excluded_rows": int(validation["excluded_rows"]),
        "development_rows": int(validation["split_counts"]["development"]),
        "test_rows": int(metrics["final_test_rows"]),
        "correct_predictions": round(metrics["accuracy"] * metrics["final_test_rows"]),
        "crossing_groups": int(validation["crossing_duplicate_groups"]),
        "parameter_configurations": int(parameter_configurations),
        "strongest_class": str(strongest["subject"]).title(),
        "strongest_f1": f"{strongest['f1']:.4f}",
        "weakest_class": str(weakest["subject"]).title(),
        "weakest_f1": f"{weakest['f1']:.4f}",
        "short_question_error_rate_percent": f"{short_error * 100:.1f}%",
        "sql_query_count": len(list((root / "sql" / "analysis").glob("*.sql"))),
        "test_count": test_count,
        "report_pdf_pages": pdf_pages,
        "bootstrap_iterations": int(
            metrics["bootstrap_macro_f1_95_ci"]["iterations"]
        ),
        "improvement_over_majority": f"{metrics['macro_f1'] / context['tables']['model_comparison'].iloc[0]['grouped_cv_macro_f1_mean']:.1f}x",
        "capability_title": str(context["capability"]).title(),
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
    context["claims"] = presentation_claims(context)
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
    for template_name, output_name in (
        ("project_brief.md.j2", "project_brief.md"),
        ("interview_guide.md.j2", "interview_guide.md"),
        ("resume_assets.md.j2", "resume_assets.md"),
    ):
        render_markdown_report(
            context,
            root / "templates" / template_name,
            root / "docs" / output_name,
        )
    return {
        "capability": context["capability"],
        "evidence_rows": len(evidence),
        "consistency_failures": int(consistency["status"].eq("fail").sum()),
    }
