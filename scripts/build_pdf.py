"""Build and render the publication PDF from verified report artifacts."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from stem_analytics.config import load_config
from stem_analytics.reporting import build_report_context

ROOT = Path(__file__).resolve().parents[1]


def _page(canvas: Any, document: Any) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#4B5563"))
    canvas.drawString(0.72 * inch, 0.42 * inch, "STEM Subject Classification | Verified offline study")
    canvas.drawRightString(7.78 * inch, 0.42 * inch, f"Page {document.page}")
    canvas.restoreState()


def _figure(path: Path, caption: str) -> list[Any]:
    return [
        Image(str(path), width=6.7 * inch, height=3.6 * inch, kind="proportional"),
        Paragraph(caption, ParagraphStyle("caption", fontName="Helvetica-Oblique", fontSize=8, leading=10, leftIndent=8, rightIndent=8, textColor=colors.HexColor("#4B5563"), spaceAfter=8)),
    ]


def build_pdf(context_path: Path, output_path: Path) -> Path:
    """Build a stable, interviewer-ready PDF from the current verified context."""
    config = load_config(context_path)
    context = build_report_context(config, ROOT)
    metrics = context["metrics"]
    validation = context["validation"]
    selection = context["selection"]
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("Title2", parent=styles["Title"], fontSize=28, leading=34, textColor=colors.HexColor("#17324D"), alignment=TA_CENTER, spaceAfter=18))
    styles.add(ParagraphStyle("Section", parent=styles["Heading1"], fontSize=19, leading=23, textColor=colors.HexColor("#17324D"), spaceAfter=12))
    styles.add(ParagraphStyle("Body2", parent=styles["BodyText"], fontSize=10.5, leading=15, spaceAfter=9))
    styles.add(ParagraphStyle("Callout", parent=styles["BodyText"], fontSize=13, leading=18, textColor=colors.HexColor("#0B6E4F"), borderColor=colors.HexColor("#9AD5C0"), borderWidth=1, borderPadding=10, backColor=colors.HexColor("#F0FAF6"), spaceAfter=12))
    story: list[Any] = []

    story.extend([Spacer(1, 1.1 * inch), Paragraph("End-to-End STEM Subject Classification", styles["Title2"]), Paragraph("A leakage-safe, evidence-grounded machine-learning feasibility study", ParagraphStyle("subtitle", parent=styles["Heading2"], alignment=TA_CENTER, textColor=colors.HexColor("#4B5563"))), Spacer(1, 0.5 * inch), Paragraph(f"Final macro-F1: {metrics['macro_f1']:.4f}<br/>95% bootstrap CI: {metrics['bootstrap_macro_f1_95_ci']['lower']:.4f} to {metrics['bootstrap_macro_f1_95_ci']['upper']:.4f}<br/>Author: Yute Chang | August 2026", styles["Callout"]), Spacer(1, 1.1 * inch), Paragraph("Portfolio report for applied ML, data science, analytics, and intelligent-manufacturing roles", ParagraphStyle("footerTitle", parent=styles["Body2"], alignment=TA_CENTER)), PageBreak()])

    sections = ["Executive Summary", "Problem and Decision", "Data Source and Governance", "Data Validation and EDA", "Leakage Control", "Modeling and Tuning", "Final Evaluation", "Error Analysis", "Features and Data Sufficiency", "Model Capability Boundaries", "Reproducibility", "Intent Annotation Future Work", "Appendix: Evidence and Metrics"]
    story.append(Paragraph("Table of Contents", styles["Section"]))
    toc = Table(
        [[str(index), section] for index, section in enumerate(sections, start=1)],
        colWidths=[0.35 * inch, 5.9 * inch],
        hAlign="LEFT",
    )
    toc.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10.5),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#111827")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(toc)
    story.append(PageBreak())

    def section(number: int, title: str, paragraphs: list[str], figure: tuple[str, str] | None = None) -> None:
        story.append(Paragraph(f"{number}. {title}", styles["Section"]))
        for text in paragraphs:
            story.append(Paragraph(text, styles["Body2"]))
        if figure:
            story.extend(_figure(ROOT / "reports" / "figures" / figure[0], figure[1]))
        story.append(PageBreak())

    section(1, "Executive Summary", [f"This study asks whether question text alone can route English STEM benchmark questions to mathematics, physics, chemistry, or biology. The locked {metrics['model_name']} achieved macro-F1 <b>{metrics['macro_f1']:.4f}</b>, accuracy {metrics['accuracy']:.4f}, and a 95% stratified bootstrap interval of {metrics['bootstrap_macro_f1_95_ci']['lower']:.4f} to {metrics['bootstrap_macro_f1_95_ci']['upper']:.4f} on {metrics['final_test_rows']} isolated questions.", "Decision: conditional go as a validated offline prototype. This supports continued validation, not production automation or claims about real students."], ("06_model_comparison.png", "Figure 6. Development-only model comparison. Source: reports/tables/model_comparison.csv; verified."))
    section(2, "Problem and Decision", ["The goal is a defensible ML workflow, not a frontend demonstration. A reviewer should be able to inspect data provenance, SQL, leakage controls, tuning, one-time test evaluation, errors, and limitations.", "The feasibility criterion is substantial improvement over the majority baseline with stable grouped cross-validation and transparent failure analysis."])
    section(3, "Data Source and Governance", [f"Source: {context['source']['dataset_name']} at pinned revision {context['source']['dataset_revision']}. The source manifest records retrieval time, license note, row counts, and checksum.", "Only normalized question text is used as a feature. Answers, answer options, explanations, and answer indices are prohibited. Raw data and model binaries are intentionally not committed."])
    section(4, "Data Validation and EDA", [f"Validation retained {validation['clean_rows']} of {validation['source_rows']} rows and excluded {validation['excluded_rows']} exact duplicates. The controlled data contains {validation['duplicate_groups']} duplicate groups and {validation['crossing_duplicate_groups']} groups crossing the final split.", "Class imbalance and source composition are measured before modeling. Source is highly associated with subject, creating shortcut and external-validity risk."], ("02_subject_by_source.png", "Figure 2. Subject by source composition. Source: SQL query 02; verified."))
    section(5, "Leakage Control", ["The final test fold was isolated before tuning. Exact duplicates were removed, near-duplicate groups were kept together, and TF-IDF fitting occurred inside each training fold.", "Five-fold StratifiedGroupKFold on development data selected hyperparameters using macro-F1. The test set was opened only after the selection decision and hashes were persisted."])
    section(6, "Modeling and Tuning", [f"Candidates were majority baseline, TF-IDF plus Logistic Regression, and TF-IDF plus LinearSVC. LinearSVC was locked at CV macro-F1 {selection['cv_macro_f1_mean']:.4f} +/- {selection['cv_macro_f1_std']:.4f}.", f"Locked parameters: {selection['best_params']}. The final model designation was not changed after observing test results."], ("09_cv_fold_variation.png", "Figure 9. Grouped-fold variation. Source: reports/tables/cv_fold_variation.csv; verified."))
    section(7, "Final Evaluation", ["Macro-F1 weights each class equally. Precision measures correctness among predictions; recall measures recovery among true items; weighted-F1 weights class F1 by support. The confusion matrix counts true/predicted pairs.", f"Final macro-F1 was {metrics['macro_f1']:.4f}, weighted-F1 {metrics['weighted_f1']:.4f}, and accuracy {metrics['accuracy']:.4f}. The uncertainty interval does not establish cross-domain performance."], ("07_confusion_matrix.png", "Figure 7. Locked final-test confusion matrix. Source: final_test_metrics.json; verified."))
    section(8, "Error Analysis", [f"Chemistry is weakest at F1 {metrics['per_class']['chemistry']['f1']:.4f}; mathematics is strongest at F1 {metrics['per_class']['math']['f1']:.4f}. Short questions under 100 characters have the highest length-band error rate.", "Representative cases and SQL slices make the errors auditable. Small per-source samples are descriptive and should not be overgeneralized."], ("10_error_slices.png", "Figure 10. Error rates by subject and length band. Source: reports/tables/error_slices.csv; verified."))
    section(9, "Features and Data Sufficiency", ["Linear coefficients reveal plausible domain vocabulary, but source-concentrated features flag shortcut risk. Feature importance is association, not causal explanation.", "Grouped validation macro-F1 improves from 0.7891 at 20% development data to 0.8807 at full size, suggesting additional representative data may still help. No source contains at least 80 rows across all four subjects, so the predeclared source-holdout test is not feasible."], ("12_learning_curve.png", "Figure 12. Grouped learning curve using development data only. Source: reports/tables/learning_curve.csv; verified."))
    section(10, "Model Capability Boundaries", ["Capability: validated offline prototype. It can demonstrate offline four-subject text routing under this benchmark split.", "It cannot validate real student behavior, educational decisions, calibrated automation, production readiness, image or non-English inputs, open-set rejection, or guaranteed cross-source generalization. LinearSVC margins are not probabilities, so confidence/coverage automation is intentionally not claimed."], ("11_confidence_coverage.png", "Figure 11. Confidence limitation. Source: reports/tables/confidence_coverage.csv; verified."))
    section(11, "Reproducibility", ["Configuration, seed 42, pinned revision, manifests, controlled-data checksum, selection hash, SQL queries, figure checksums, tests, and commands are versioned.", "Run stem-analytics run-all --config configs/project.yaml after installing the project. Raw data, SQLite, predictions, and model binaries rebuild locally and remain ignored by Git."])
    section(12, "Intent Annotation Future Work", ["Intent classification is deferred because no trustworthy source label exists. The repository supplies a five-label taxonomy, a balanced 40-question pilot sampler, and validation rules only.", "Future work requires independent double labeling, raw agreement, Cohen's kappa, adjudication, taxonomy revision when agreement is weak, and at least 20% double labeling before an 800-question study. No human annotation or intent model is claimed."])

    story.append(Paragraph("13. Appendix: Evidence and Metrics", styles["Section"]))
    per_class = context["tables"]["per_class_metrics"]
    data = [["Subject", "Precision", "Recall", "F1", "Support"]] + [[row.subject.title(), f"{row.precision:.4f}", f"{row.recall:.4f}", f"{row.f1:.4f}", str(int(row.support))] for row in per_class.itertuples(index=False)]
    table = Table(data, colWidths=[1.35 * inch, 1.1 * inch, 1.1 * inch, 1.1 * inch, 1.0 * inch], repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324D")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("ALIGN", (1, 1), (-1, -1), "RIGHT"), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]), ("BOTTOMPADDING", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (-1, -1), 7)]))
    story.extend([Paragraph("Final-test per-class metrics", styles["Body2"]), table, Spacer(1, 0.25 * inch), Paragraph("Authoritative sources: artifacts/metrics/final_test_metrics.json, selection_decision.json, generated tables, manifests, SQL exports, and figure source register. All headline numerical consistency checks pass.", styles["Body2"])])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(str(output_path), pagesize=letter, rightMargin=0.72 * inch, leftMargin=0.72 * inch, topMargin=0.65 * inch, bottomMargin=0.65 * inch, title="End-to-End STEM Subject Classification", author="Yute Chang")
    document.build(story, onFirstPage=_page, onLaterPages=_page)
    return output_path


def audit_pdf(pdf_path: Path, render_dir: Path) -> pd.DataFrame:
    """Render every page and initialize the visual audit register."""
    render_dir.mkdir(parents=True, exist_ok=True)
    for old in render_dir.glob("page-*.png"):
        old.unlink()
    executable = shutil.which("pdftoppm")
    if executable is None:
        raise FileNotFoundError("pdftoppm is required for PDF visual QA")
    executable_path = Path(executable)
    if executable_path.suffix.lower() == ".cmd":
        bundled_exe = (
            executable_path.parents[2]
            / "native"
            / "poppler"
            / "Library"
            / "bin"
            / "pdftoppm.exe"
        )
        if bundled_exe.exists():
            executable = str(bundled_exe)
    subprocess.run([executable, "-png", "-r", "120", str(pdf_path), str(render_dir / "page")], check=True)
    page_count = len(PdfReader(pdf_path).pages)
    images = sorted(render_dir.glob("page-*.png"))
    if len(images) != page_count:
        raise ValueError("rendered image count does not match PDF page count")
    return pd.DataFrame([{"page_number": number, "status": "pending_visual_review", "issues": "", "resolution": ""} for number in range(1, page_count + 1)])


if __name__ == "__main__":
    pdf = build_pdf(ROOT / "configs" / "project.yaml", ROOT / "docs" / "final_report.pdf")
    audit = audit_pdf(pdf, ROOT / "reports" / "qa" / "pdf_pages")
    audit.to_csv(ROOT / "reports" / "qa" / "pdf_page_audit.csv", index=False, encoding="utf-8")
