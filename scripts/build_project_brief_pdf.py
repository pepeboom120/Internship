"""Build and render the one-page interviewer project brief."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

from stem_analytics.config import load_config
from stem_analytics.reporting import build_report_context, presentation_claims

ROOT = Path(__file__).resolve().parents[1]
PAGE_WIDTH, PAGE_HEIGHT = letter
NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#4477AA")
GREEN = colors.HexColor("#0B6E4F")
LIGHT_BLUE = colors.HexColor("#EEF5FA")
LIGHT_GREEN = colors.HexColor("#ECF8F3")
GRAY = colors.HexColor("#4B5563")
LINE = colors.HexColor("#CBD5E1")


def _wrapped_lines(text: str, font: str, size: float, width: float) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if stringWidth(candidate, font, size) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _paragraph(
    pdf: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    width: float,
    size: float = 8.4,
    leading: float = 11.2,
    color: colors.Color = colors.black,
    font: str = "Helvetica",
) -> float:
    pdf.setFont(font, size)
    pdf.setFillColor(color)
    for line in _wrapped_lines(text, font, size, width):
        pdf.drawString(x, y, line)
        y -= leading
    return y


def _section_title(pdf: canvas.Canvas, title: str, x: float, y: float) -> float:
    pdf.setFont("Helvetica-Bold", 10.5)
    pdf.setFillColor(NAVY)
    pdf.drawString(x, y, title)
    pdf.setStrokeColor(LINE)
    pdf.line(x, y - 4, x + 238, y - 4)
    return y - 17


def _metric_card(
    pdf: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    value: str,
    label: str,
    note: str,
) -> None:
    pdf.setFillColor(LIGHT_BLUE)
    pdf.roundRect(x, y, width, 58, 6, fill=1, stroke=0)
    pdf.setFillColor(NAVY)
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(x + 12, y + 33, value)
    pdf.setFont("Helvetica-Bold", 8)
    pdf.drawString(x + 12, y + 20, label.upper())
    pdf.setFont("Helvetica", 6.8)
    pdf.setFillColor(GRAY)
    pdf.drawString(x + 12, y + 9, note)


def build_project_brief(config_path: Path, output_path: Path) -> Path:
    """Create a fixed one-page PDF from verified reporting context."""
    config = load_config(config_path)
    context = build_report_context(config, ROOT)
    claims = presentation_claims(context)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(output_path), pagesize=letter)
    pdf.setTitle("STEM Subject Classification - One-Page Project Brief")
    pdf.setAuthor("Yute Chang")

    margin = 42
    content_width = PAGE_WIDTH - 2 * margin
    pdf.setFillColor(NAVY)
    pdf.rect(0, PAGE_HEIGHT - 106, PAGE_WIDTH, 106, fill=1, stroke=0)
    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 22)
    pdf.drawString(margin, PAGE_HEIGHT - 47, "End-to-End STEM Subject Classification")
    pdf.setFont("Helvetica", 10)
    pdf.drawString(
        margin,
        PAGE_HEIGHT - 67,
        "Leakage-safe ML workflow | Python | SQL | scikit-learn | CI/CD",
    )
    pdf.setFont("Helvetica-Bold", 9)
    pdf.setFillColor(colors.HexColor("#A7F3D0"))
    pdf.drawString(
        margin,
        PAGE_HEIGHT - 88,
        f"Conditional Go - {claims['capability_title']}",
    )

    card_y = PAGE_HEIGHT - 181
    gap = 9
    card_width = (content_width - 2 * gap) / 3
    _metric_card(
        pdf,
        margin,
        card_y,
        card_width,
        str(claims["macro_f1"]),
        "Final macro-F1",
        f"95% CI {claims['ci_lower']}-{claims['ci_upper']}",
    )
    _metric_card(
        pdf,
        margin + card_width + gap,
        card_y,
        card_width,
        str(claims["accuracy"]),
        "Final accuracy",
        f"{claims['correct_predictions']}/{claims['test_rows']} correct",
    )
    _metric_card(
        pdf,
        margin + 2 * (card_width + gap),
        card_y,
        card_width,
        str(claims["cv_macro_f1"]),
        "Grouped-CV macro-F1",
        f"+/- {claims['cv_std']} before test access",
    )

    flow_y = card_y - 27
    stages = ["PIN DATA", "VALIDATE", "GROUP SPLIT", "TUNE", "LOCK TEST", "DIAGNOSE"]
    stage_gap = 5
    stage_width = (content_width - stage_gap * (len(stages) - 1)) / len(stages)
    for index, stage in enumerate(stages):
        x = margin + index * (stage_width + stage_gap)
        pdf.setFillColor(BLUE if index < 4 else GREEN)
        pdf.roundRect(x, flow_y, stage_width, 19, 4, fill=1, stroke=0)
        pdf.setFillColor(colors.white)
        pdf.setFont("Helvetica-Bold", 6.4)
        pdf.drawCentredString(x + stage_width / 2, flow_y + 6.5, stage)

    left_x = margin
    right_x = 318
    column_width = 252
    y = flow_y - 28
    left_y = _section_title(pdf, "Problem and Data", left_x, y)
    left_y = _paragraph(
        pdf,
        (
            "Can question text alone route English STEM benchmark questions into "
            "mathematics, physics, chemistry, or biology? The pipeline pinned "
            f"{claims['source_rows']} records, removed {claims['excluded_rows']} exact "
            f"duplicates, and retained {claims['controlled_rows']} controlled rows."
        ),
        left_x,
        left_y,
        column_width,
    )
    left_y -= 7
    left_y = _section_title(pdf, "Leakage Controls", left_x, left_y)
    for bullet in (
        f"{claims['crossing_groups']} duplicate groups cross development and final test.",
        "TF-IDF is fitted inside every training fold.",
        f"{claims['parameter_configurations']} parameter settings use grouped CV only.",
        "Selection hashes are persisted before one-time test evaluation.",
    ):
        left_y = _paragraph(pdf, f"- {bullet}", left_x, left_y, column_width)
        left_y -= 2
    left_y -= 5
    left_y = _section_title(pdf, "Engineering Evidence", left_x, left_y)
    _paragraph(
        pdf,
        (
            f"Importable Python package and CLI; {claims['sql_query_count']} named SQL "
            "analyses; artifact hashes; automated tests; clean-room install; GitHub "
            f"Actions; numerical gates; and a visually audited {claims['report_pdf_pages']}-page report."
        ),
        left_x,
        left_y,
        column_width,
    )

    right_y = _section_title(pdf, "Findings", right_x, y)
    for bullet in (
        f"{claims['strongest_class']} is strongest at F1 {claims['strongest_f1']}.",
        f"{claims['weakest_class']} is weakest at F1 {claims['weakest_f1']}.",
        f"Questions below 100 characters have {claims['short_question_error_rate_percent']} error.",
        "The learning curve still improves with more development data.",
    ):
        right_y = _paragraph(pdf, f"- {bullet}", right_x, right_y, column_width)
        right_y -= 2
    right_y -= 5
    right_y = _section_title(pdf, "Capability Boundaries", right_x, right_y)
    for bullet in (
        "Benchmark evidence, not student-behavior or manufacturing validation.",
        "LinearSVC margins are not calibrated probabilities.",
        "Source-subject confounding leaves cross-source transfer unverified.",
        "Production readiness and automated decisions are not claimed.",
    ):
        right_y = _paragraph(pdf, f"- {bullet}", right_x, right_y, column_width)
        right_y -= 2
    right_y -= 5
    right_y = _section_title(pdf, "Decision", right_x, right_y)
    pdf.setFillColor(LIGHT_GREEN)
    pdf.roundRect(right_x, right_y - 55, column_width, 57, 5, fill=1, stroke=0)
    _paragraph(
        pdf,
        (
            "Proceed as an offline four-subject routing prototype. Next valid evidence "
            "requires new multi-source data or a separately frozen manufacturing case, "
            "not reuse of the current final test."
        ),
        right_x + 10,
        right_y - 13,
        column_width - 20,
        size=8.1,
        leading=10.5,
        color=GREEN,
        font="Helvetica-Bold",
    )

    lower_title_y = 290
    _section_title(pdf, "Final-Test Behavior", left_x, lower_title_y)
    pdf.drawImage(
        str(ROOT / "reports" / "figures" / "07_confusion_matrix.png"),
        left_x + 25,
        78,
        width=188,
        height=188,
        preserveAspectRatio=True,
        anchor="c",
        mask="auto",
    )
    pdf.setFont("Helvetica-Oblique", 6.5)
    pdf.setFillColor(GRAY)
    pdf.drawString(left_x + 22, 70, "Locked model only | n = 630 | verified metric artifact")

    review_y = _section_title(pdf, "Five-Minute Review Path", right_x, lower_title_y)
    for number, item in enumerate(
        (
            "Read this one-page decision summary.",
            "Open selection_decision.json for the lock evidence.",
            "Inspect confusion and error-slice artifacts.",
            "Review limits and the not-feasible holdout.",
            "Run stem-analytics run-all to show reproducibility.",
        ),
        start=1,
    ):
        pdf.setFillColor(BLUE)
        pdf.circle(right_x + 8, review_y + 3, 7, fill=1, stroke=0)
        pdf.setFillColor(colors.white)
        pdf.setFont("Helvetica-Bold", 7)
        pdf.drawCentredString(right_x + 8, review_y + 0.5, str(number))
        review_y = _paragraph(
            pdf,
            item,
            right_x + 22,
            review_y + 6,
            column_width - 22,
            size=8.2,
            leading=10.2,
        )
        review_y -= 7

    pdf.setFillColor(LIGHT_BLUE)
    pdf.roundRect(right_x, 78, column_width, 67, 5, fill=1, stroke=0)
    pdf.setFillColor(NAVY)
    pdf.setFont("Helvetica-Bold", 9)
    pdf.drawString(right_x + 11, 127, "Core portfolio signal")
    _paragraph(
        pdf,
        (
            "The result matters, but the stronger evidence is disciplined data control, "
            "honest evaluation, reproducible engineering, and decisions tied to failure analysis."
        ),
        right_x + 11,
        112,
        column_width - 22,
        size=7.7,
        leading=9.4,
        color=GRAY,
    )

    footer_y = 39
    pdf.setStrokeColor(LINE)
    pdf.line(margin, footer_y + 18, PAGE_WIDTH - margin, footer_y + 18)
    pdf.setFillColor(NAVY)
    pdf.setFont("Helvetica-Bold", 8.2)
    pdf.drawString(margin, footer_y, "Yute Chang")
    pdf.setFillColor(GRAY)
    pdf.setFont("Helvetica", 7.6)
    repository = "github.com/pepeboom120/Internship"
    pdf.drawRightString(PAGE_WIDTH - margin, footer_y, repository)
    pdf.linkURL(
        "https://github.com/pepeboom120/Internship",
        (PAGE_WIDTH - margin - 160, footer_y - 3, PAGE_WIDTH - margin, footer_y + 9),
        relative=0,
    )
    pdf.showPage()
    pdf.save()
    return output_path


def _poppler_executable() -> str:
    executable = shutil.which("pdftoppm")
    if executable is None:
        raise FileNotFoundError("pdftoppm is required for project brief visual QA")
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
            return str(bundled_exe)
    return executable


def render_project_brief(pdf_path: Path, render_dir: Path) -> Path:
    """Render and structurally confirm the single-page brief."""
    render_dir.mkdir(parents=True, exist_ok=True)
    for old in render_dir.glob("page-*.png"):
        old.unlink()
    subprocess.run(
        [
            _poppler_executable(),
            "-png",
            "-r",
            "150",
            str(pdf_path),
            str(render_dir / "page"),
        ],
        check=True,
    )
    if len(PdfReader(pdf_path).pages) != 1:
        raise ValueError("project brief must contain exactly one page")
    images = sorted(render_dir.glob("page-*.png"))
    if len(images) != 1:
        raise ValueError("project brief render must produce exactly one image")
    return images[0]


if __name__ == "__main__":
    brief = build_project_brief(
        ROOT / "configs" / "project.yaml", ROOT / "docs" / "project_brief.pdf"
    )
    rendered = render_project_brief(
        brief, ROOT / "reports" / "qa" / "project_brief_pages"
    )
    print(f"Built {brief} and rendered {rendered}")
