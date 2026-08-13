from pathlib import Path

import pytest
from pypdf import PdfReader


def test_project_brief_is_exactly_one_page() -> None:
    path = Path("docs/project_brief.pdf")
    if not path.exists():
        pytest.fail("project brief PDF has not been generated")
    reader = PdfReader(path)
    assert len(reader.pages) == 1
    text = reader.pages[0].extract_text() or ""
    assert "0.8755" in text
    assert "Validated Offline Prototype" in text
    assert "Capability Boundaries" in text
