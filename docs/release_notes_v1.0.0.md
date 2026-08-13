# v1.0.0 — Reproducible End-to-End ML Workflow

This release packages the STEM subject-classification feasibility study as a public,
auditable ML portfolio project. The locked LinearSVC model reaches **0.8755 final-test
macro-F1** and **0.8778 accuracy** on 630 untouched questions. Its 95% bootstrap interval
for macro-F1 is 0.8471–0.8996.

The evidence supports a **validated offline prototype** for routing English benchmark
questions among mathematics, physics, chemistry, and biology. It does not establish
student-behavior prediction, manufacturing-domain performance, calibrated probabilities,
or production readiness.

## What is included

- Reproducible Python package and `stem-analytics` CLI covering validation, deduplication,
  grouped splitting, SQL analysis, model tuning, locked evaluation, diagnostics, and reports.
- Leakage controls: exact duplicates removed before splitting, duplicate groups kept together,
  TF-IDF fitted inside each fold, and the final test opened only after selection was persisted.
- A 72-configuration comparison of logistic regression and LinearSVC using grouped CV.
- Persisted selection hashes, model metrics, predictions, confusion matrix, error slices,
  learning curve, numerical checks, and figure-source evidence.
- GitHub Actions checks for Python 3.11 and 3.12.

## Verified result

| Measure | Value |
|---|---:|
| Final-test macro-F1 | 0.8755 |
| Final-test accuracy | 0.8778 (553/630) |
| Grouped-CV macro-F1 | 0.8807 ± 0.0171 |
| Bootstrap 95% CI | 0.8471–0.8996 |
| Duplicate groups crossing development/final test | 0 |

Mathematics is the strongest class (F1 0.9361), chemistry is the weakest (F1 0.8391),
and questions shorter than 100 characters have a 19.7% error rate. The learning curve
still improves with more development data, so broader multi-source validation is the next
appropriate experiment.

## Review assets

- `docs/project_brief.pdf` — one-page interview summary
- `docs/final_report.pdf` — complete 15-page technical report
- `docs/interview_guide.md` — five-minute review path and likely interview questions
- `docs/resume_assets.md` — resume bullets and project pitch

The release intentionally excludes raw data, processed records, SQLite databases,
predictions containing question text, and model binaries. All reported claims are traceable
to committed aggregate artifacts and QA registers.
