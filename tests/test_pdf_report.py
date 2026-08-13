from pathlib import Path

import pytest
from pypdf import PdfReader


def test_pdf_has_expected_structure() -> None:
    path = Path("docs/final_report.pdf")
    if not path.exists():
        pytest.fail("final report PDF has not been generated")
    reader = PdfReader(path)
    assert len(reader.pages) >= 12
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert "Executive Summary" in text
    assert "Model Capability Boundaries" in text
    assert "Reproducibility" in text
    assert "Intent Annotation Future Work" in text
