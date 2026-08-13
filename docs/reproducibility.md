# Reproducibility

Use Python 3.11 or 3.12 on Windows, macOS, or Linux. Create a virtual environment, install `.[dev]`, then run `stem-analytics run-all --config configs/project.yaml`. The full-data path downloads the pinned MMLU-Pro revision; raw Parquet, SQLite, predictions, and model binaries are intentionally ignored by Git. Public metrics, tables, figures, manifests, tests, and report artifacts are versioned.

The pipeline seed is 42. The dataset revision, source checksum, controlled-data checksum, configuration hash, selection decision, and figure hashes are stored under `artifacts/` and `reports/qa/`.

## PDF build and QA

The PDF uses ReportLab from the project environment and Poppler from the Codex bundled runtime (`26.812.11052`). Run `.\.venv\Scripts\python.exe scripts\build_pdf.py`; the script builds `docs/final_report.pdf`, renders every page with `pdftoppm -png -r 120`, and writes the initial audit register. The final visual decisions are recorded in `reports/qa/pdf_page_audit.csv` and `pdf_qa_report.md`. Structural verification uses `pypdf` through `pytest tests/test_pdf_report.py -q`.
