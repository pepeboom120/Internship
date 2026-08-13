# Portfolio Release and Resume Presentation Design

**Status:** Proposed for final user review  
**Date:** 2026-08-13  
**Scope:** First-priority packaging and public release of the completed ML workflow

## 1. Objective

Turn the completed technical repository into an interviewer-ready public artifact that can be understood in under five minutes and cited directly from a resume. This phase does not change the model, reuse the final test set, add a web application, or claim new scientific results.

## 2. Deliverables

1. A polished UTF-8 GitHub README on `main` with a concise bilingual summary, verified metrics, representative figures, CI/release badges, a five-minute review path, and direct links to the full report and one-page brief.
2. A one-page English project brief in Markdown and PDF containing the problem, dataset, workflow, leakage controls, results, limitations, and feasibility decision.
3. An interview guide with a five-minute walkthrough, a 30-second pitch, common technical questions, and evidence-backed answers.
4. A resume asset file with concise English and Chinese bullet variants. Every metric must trace to the existing authoritative artifacts.
5. A GitHub pull request from `feat/end-to-end-ml-workflow` to `main`, successful CI checks, merge to `main`, repository description/topics, and a `v1.0.0` release containing the full report PDF and one-page brief PDF.
6. Pinning guidance or profile pinning when the authenticated GitHub interface permits it.

## 3. Information Architecture

The README will present information in this order:

1. project purpose and feasibility decision;
2. three headline metrics;
3. workflow and representative figures;
4. five-minute reviewer path;
5. technical decisions and leakage controls;
6. error findings and capability boundaries;
7. reproducibility commands;
8. links to the brief, full report, interview guide, and resume bullets.

The one-page brief will favor scanability over methodological completeness. The existing 15-page report remains the authoritative long-form source.

## 4. Evidence and Numerical Controls

- Headline values must be generated from `artifacts/metrics/final_test_metrics.json` and existing verified tables.
- The project brief and resume bullets must not introduce unsupported business impact, student-behavior claims, production-readiness claims, or manufacturing-domain claims.
- A consistency test will check the README, brief source, and resume asset values against the locked metric JSON.
- UTF-8 files will be decoded in automated tests, scanned for the Unicode replacement character, and visually checked on GitHub after publishing.

## 5. PDF Design

The one-page PDF will use the same restrained blue/green visual language as the main report. It will contain no dense appendix material. It must remain exactly one page, render through Poppler, and pass visual checks for clipping, overlap, broken glyphs, and legibility.

## 6. GitHub Release Flow

1. Implement and verify all packaging artifacts on the feature branch.
2. Push the branch and open a pull request to remote `main`.
3. Wait for GitHub Actions and investigate any failure before merging.
4. Merge without rewriting published feature history.
5. Verify `main`, README rendering, links, and CI.
6. Create annotated release/tag `v1.0.0` and attach both PDFs.
7. Add a concise repository description and relevant topics.

## 7. Error Handling and Safety

- Do not merge when tests, numerical checks, PDF QA, or CI are failing.
- Do not force-push or rewrite remote history.
- If GitHub authentication or profile pinning is unavailable, preserve the completed local and remote branch work and report the exact manual step.
- Do not expose private email addresses, credentials, raw data, predictions, databases, or model binaries.

## 8. Acceptance Criteria

- Full local suite and Ruff pass.
- Clean-room verification passes.
- README and presentation files decode as UTF-8 with no replacement characters.
- All public numerical claims match authoritative artifacts.
- Project brief PDF is exactly one visually verified page.
- PR CI succeeds and the final commit is present on remote `main`.
- `v1.0.0` release is public with stable PDF assets.
- Resume and interview materials describe current capability honestly.

## 9. Explicit Non-goals

- No new model training or final-test evaluation.
- No Web UI, API, Docker, Kubernetes, MLflow, or monitoring implementation in this phase.
- No semiconductor-manufacturing companion model in this phase.
