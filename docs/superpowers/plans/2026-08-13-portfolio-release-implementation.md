# Portfolio Release and Resume Presentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the completed ML workflow as an interviewer-ready GitHub `v1.0.0` release with a polished README, one-page brief, resume/interview assets, verified CI, and stable artifacts on `main`.

**Architecture:** Reuse `build_report_context()` as the only source for public numerical claims, render all Markdown presentation assets through Jinja templates, and build the one-page PDF from the same context. Complete local evidence gates before GitHub mutation, then use a PR-based merge and release flow without rewriting history.

**Tech Stack:** Python 3.11/3.12, Jinja2, ReportLab, pypdf, Poppler, pytest, Ruff, Git, GitHub Actions, GitHub CLI or authenticated in-app browser.

## Global Constraints

- Do not retrain a model or reevaluate the final test set.
- Every displayed metric must trace to `artifacts/metrics/final_test_metrics.json` or a verified generated table.
- Do not claim manufacturing validation, student-behavior validation, calibrated confidence, or production readiness.
- Public text files must decode as UTF-8 and contain no Unicode replacement character.
- The project brief PDF must be exactly one visually verified page.
- Do not commit raw data, processed data, SQLite databases, predictions, model binaries, credentials, or files over 10 MB.
- Do not force-push or rewrite published branch history.

---

### Task 1: Generate Interviewer-Facing Markdown Assets from Verified Evidence

**Files:**
- Modify: `src/stem_analytics/reporting.py`
- Modify: `templates/README.md.j2`
- Create: `templates/project_brief.md.j2`
- Create: `templates/interview_guide.md.j2`
- Create: `templates/resume_assets.md.j2`
- Generate: `docs/project_brief.md`
- Generate: `docs/interview_guide.md`
- Generate: `docs/resume_assets.md`
- Modify: `tests/test_reporting.py`

**Interfaces:**
- Consumes: `build_report_context(config: ProjectConfig, root: Path = Path('.')) -> dict[str, Any]`.
- Produces: `presentation_claims(context: dict[str, Any]) -> dict[str, str | int | float]`.
- Produces: `generate_report(...)` outputs the existing report plus README, project brief, interview guide, and resume assets.

- [ ] **Step 1: Write failing evidence and UTF-8 tests**

Add tests that require every public document and verified value:

```python
def test_presentation_assets_are_utf8_and_have_no_replacement_character() -> None:
    for path in PUBLIC_PRESENTATION_PATHS:
        text = path.read_text(encoding="utf-8")
        assert "\ufffd" not in text


def test_resume_metrics_match_locked_metrics() -> None:
    context = build_report_context(load_config(Path("configs/project.yaml")))
    claims = presentation_claims(context)
    assert claims["macro_f1"] == f"{context['metrics']['macro_f1']:.4f}"
    assert claims["test_rows"] == context["metrics"]["final_test_rows"]
```

- [ ] **Step 2: Run focused tests and confirm RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_reporting.py -q`

Expected: FAIL because `presentation_claims` and the generated assets do not exist.

- [ ] **Step 3: Implement the presentation claim contract**

Add a formatting-only dictionary that contains the locked metrics, counts, model name, strongest/weakest class, CV result, test-set policy, test count, SQL query count, test count, and PDF page count. Values must be computed from the report context or verified registers; do not type headline metrics inside templates.

- [ ] **Step 4: Create the three presentation templates**

The project brief must contain Problem, Data, Workflow, Controls, Results, Findings, Decision, Limits, and Links. The interview guide must include a 30-second pitch, five-minute review path, six likely technical questions, and concise evidence-backed answers. The resume asset must include three English bullets, three Chinese bullets, a one-line project title, and two honest variants for ML/Data Science and IMC/MFG applications.

- [ ] **Step 5: Restructure the README for five-minute review**

Add main-branch CI/release badges; maintain the English and Traditional Chinese summary; place the reviewer path before repository details; add direct links to `docs/project_brief.pdf`, `docs/final_report.pdf`, `docs/interview_guide.md`, and `docs/resume_assets.md`. Use ASCII hyphens for ranges and separators where practical.

- [ ] **Step 6: Generate assets and verify GREEN**

Run:

```powershell
.\.venv\Scripts\stem-analytics.exe report --config configs/project.yaml
.\.venv\Scripts\python.exe -m pytest tests/test_reporting.py -q
.\.venv\Scripts\python.exe -m ruff check src tests
```

Expected: all presentation files exist, decode as UTF-8, contain no replacement characters, and metrics agree.

- [ ] **Step 7: Commit the Markdown presentation layer**

```powershell
git add README.md src/stem_analytics/reporting.py templates tests/test_reporting.py docs/project_brief.md docs/interview_guide.md docs/resume_assets.md
git commit -m "docs: add interviewer-ready project presentation"
```

---

### Task 2: Build and Visually Verify the One-Page Project Brief PDF

**Files:**
- Create: `scripts/build_project_brief_pdf.py`
- Create: `tests/test_project_brief_pdf.py`
- Generate: `docs/project_brief.pdf`
- Generate: `reports/qa/project_brief_page_audit.csv`
- Generate: `reports/qa/project_brief_qa_report.md`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `build_report_context()` and `presentation_claims()`.
- Produces: `build_project_brief(config_path: Path, output_path: Path) -> Path`.
- Produces: `render_project_brief(pdf_path: Path, render_dir: Path) -> Path`.

- [ ] **Step 1: Write the failing structural PDF test**

```python
def test_project_brief_is_exactly_one_page() -> None:
    reader = PdfReader("docs/project_brief.pdf")
    assert len(reader.pages) == 1
    text = reader.pages[0].extract_text() or ""
    assert "0.8755" in text
    assert "Validated Offline Prototype" in text
    assert "Capability Boundaries" in text
```

- [ ] **Step 2: Run focused test and confirm RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_project_brief_pdf.py -q`

Expected: FAIL because the one-page PDF does not exist.

- [ ] **Step 3: Implement the one-page ReportLab layout**

Use letter size, a restrained blue/green palette, one small workflow strip, three metric cards, two result findings, three capability limits, and stable links. Use only evidence supplied through `presentation_claims()`.

- [ ] **Step 4: Render with bundled Poppler**

Run: `.\.venv\Scripts\python.exe scripts/build_project_brief_pdf.py`

The script must locate the bundled `pdftoppm.exe` when the Windows `.cmd` wrapper is present and render to ignored `reports/qa/project_brief_pages/`.

- [ ] **Step 5: Inspect the rendered page and record QA**

Inspect at original detail for clipping, overlap, broken characters, unreadable captions, or excess whitespace. Correct the source and regenerate until it passes. Record exactly one `pass` row with any corrections in `project_brief_page_audit.csv` and summarize the review in `project_brief_qa_report.md`.

- [ ] **Step 6: Run PDF and full tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_project_brief_pdf.py tests/test_pdf_report.py -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests
```

Expected: project brief is exactly one page and all tests pass.

- [ ] **Step 7: Commit the one-page brief**

```powershell
git add .gitignore scripts/build_project_brief_pdf.py tests/test_project_brief_pdf.py docs/project_brief.pdf reports/qa/project_brief_page_audit.csv reports/qa/project_brief_qa_report.md
git commit -m "docs: add verified one-page project brief"
```

---

### Task 3: Prepare Version 1.0 Release Evidence

**Files:**
- Modify: `pyproject.toml`
- Create: `docs/release_notes_v1.0.0.md`
- Modify: `reports/qa/release_checklist.md`
- Modify: `tests/test_pipeline_integration.py`

**Interfaces:**
- Produces package version `1.0.0`.
- Produces release notes whose metrics are checked against the authoritative context.

- [ ] **Step 1: Write failing version and release-note tests**

```python
def test_release_version_and_notes_are_consistent() -> None:
    assert tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]["version"] == "1.0.0"
    notes = Path("docs/release_notes_v1.0.0.md").read_text(encoding="utf-8")
    assert "0.8755" in notes
    assert "validated offline prototype" in notes.lower()
```

- [ ] **Step 2: Run focused test and confirm RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_pipeline_integration.py -q`

Expected: FAIL because the version is `0.1.0` and release notes do not exist.

- [ ] **Step 3: Update version and generate release notes**

Set `project.version = "1.0.0"`. Release notes must list verified metrics, pipeline stages, included PDFs, reproducibility status, and current capability boundaries without introducing new conclusions.

- [ ] **Step 4: Run final local gates**

Run the complete test suite, Ruff, `stem-analytics run-all`, clean-room verification, UTF-8 scan, numerical register, PDF audits, secret patterns, forbidden tracked files, and the 10 MB limit. Update the checklist with exact evidence and date.

- [ ] **Step 5: Commit release preparation**

```powershell
git add pyproject.toml docs/release_notes_v1.0.0.md reports/qa/release_checklist.md tests/test_pipeline_integration.py
git commit -m "chore: prepare version 1.0.0 release"
```

---

### Task 4: Publish Through Pull Request and Verify Main

**Files:**
- Remote mutation only after local gates pass.

**Interfaces:**
- Produces a GitHub PR from `feat/end-to-end-ml-workflow` to `main`.
- Produces a merged remote `main` with successful GitHub Actions.

- [ ] **Step 1: Push the verified feature branch**

Run: `git push origin feat/end-to-end-ml-workflow`

Expected: remote feature SHA equals local HEAD.

- [ ] **Step 2: Create the pull request**

Use GitHub CLI when authenticated; otherwise use the authenticated in-app browser. Title: `Build reproducible end-to-end ML workflow`. The body must summarize data governance, modeling, locked evaluation, diagnostics, reporting, verification, result metrics, and known limitations.

- [ ] **Step 3: Wait for GitHub Actions**

Require the Python 3.11 and 3.12 jobs to succeed. On failure, inspect logs, reproduce locally, fix on the feature branch, rerun all relevant tests, push, and wait again.

- [ ] **Step 4: Merge the PR**

Merge only after required checks pass. Do not force-push. Preserve the feature branch until release verification is finished.

- [ ] **Step 5: Verify remote main**

Require remote `main` to contain the merged final commit. Open the repository homepage and verify README headings, Chinese summary, badges, figures, and document links render correctly.

---

### Task 5: Publish GitHub Release and Repository Metadata

**Files:**
- Remote GitHub release and repository metadata.

**Interfaces:**
- Produces tag/release `v1.0.0` on merged `main`.
- Attaches `docs/final_report.pdf` and `docs/project_brief.pdf`.

- [ ] **Step 1: Create the release**

Create annotated tag and release `v1.0.0` using `docs/release_notes_v1.0.0.md`. Attach the full technical report and one-page project brief PDFs.

- [ ] **Step 2: Add repository metadata**

Set description to `Leakage-safe, reproducible ML workflow with SQL analytics, grouped CV, locked evaluation, error analysis, CI, and verified reports.` Add topics: `machine-learning`, `data-science`, `scikit-learn`, `mlops`, `sql`, `nlp`, `reproducibility`, `portfolio`.

- [ ] **Step 3: Verify public release**

Open the release page and verify tag, commit, notes, and both downloadable PDF assets. Confirm the README release badge resolves.

- [ ] **Step 4: Pin or document the final profile action**

If the authenticated UI permits profile customization, pin the repository. Otherwise provide the exact GitHub profile path: `Your profile -> Customize your pins -> Internship -> Save pins`.

- [ ] **Step 5: Record final handoff**

Report the repository URL, PR URL, release URL, main SHA, test count, clean-room result, PDF page counts, and the recommended resume bullets. Preserve the feature worktree until the user confirms no further release edits are needed.
