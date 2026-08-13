# End-to-End STEM Subject Classification ML Workflow Design

**Status:** Approved design  
**Date:** 2026-08-13  
**Primary audience:** Hiring managers and technical interviewers for applied machine-learning, data-science, data-analysis, and intelligent-manufacturing roles  
**Target presentation modes:** Public GitHub repository and locally reproducible demonstration

## 1. Purpose

Build a public, evidence-grounded portfolio project that demonstrates a complete machine-learning workflow for classifying English STEM questions into Mathematics, Physics, Chemistry, or Biology. The project must show disciplined data processing, SQL analysis, model development, hyperparameter tuning, leakage control, evaluation, error diagnosis, reproducibility, and technical communication.

The work is a feasibility study using a public benchmark dataset. It is not a production deployment and must not be described as evidence about real student behavior.

## 2. Success Criteria

The finished project must provide:

1. A public GitHub repository that lets a reviewer understand the problem, workflow, results, limitations, and technical decisions without running the code.
2. A locally reproducible pipeline that can rebuild the core data, analysis, model, evaluation, figures, and report from documented commands.
3. A complete English technical report with traceable numerical results, representative visualizations, model limitations, and a defensible feasibility decision.
4. A PDF version suitable for sending directly to an interviewer.
5. Evidence of Python, SQL, scikit-learn pipelines, hyperparameter tuning, evaluation design, error analysis, automated testing, and basic CI.

No Web UI, REST API, Kubernetes deployment, microservice architecture, or interactive question-to-prediction demo is required.

## 3. Scope

### 3.1 Core scope

- Pin and document a licensed MMLU-Pro dataset revision.
- Select the Mathematics, Physics, Chemistry, and Biology records.
- Validate schema and content, normalize text conservatively, remove unusable rows, and detect duplicates.
- Create leakage-safe, deterministic data splits.
- Store controlled analytical data and model outputs in SQLite.
- Produce stakeholder-relevant SQL analyses and exploratory visualizations.
- Compare a majority-class baseline, TF-IDF plus Logistic Regression, and TF-IDF plus LinearSVC.
- Conduct cross-validated hyperparameter tuning using macro-F1.
- Evaluate the locked model once on an isolated final test set.
- Analyze per-class performance, uncertainty, confidence thresholds, error slices, learned features, and learning curves.
- Generate GitHub-facing documentation, a complete technical report, and a visually verified PDF.
- Provide tests, a command-line workflow, experiment metadata, and GitHub Actions checks.

### 3.2 Deferred scope

Intent classification is deferred because the source dataset does not provide trustworthy intent labels. The repository will include:

- an intent taxonomy;
- an annotation guide;
- deterministic tooling to sample a 40-question annotation pilot;
- a proposed agreement and adjudication protocol;
- a documented future-work path.

It will not claim that human annotation, agreement measurement, an 800-question reviewed dataset, or an intent classifier has been completed.

### 3.3 Non-goals

- Production deployment or real-time model serving
- Frontend or Streamlit development
- An unrestricted generative-AI learning assistant
- Use of private or personal student data
- Exact prediction of learner needs or educational outcomes
- Image-based questions or OCR
- Non-English inputs or subjects beyond the four declared classes
- Claims of real-world manufacturing or student-platform validation

## 4. Audience and Portfolio Positioning

The project is optimized for roles similar to intelligent-manufacturing AI/ML, data-science, and data-analysis positions. It should demonstrate that the author can:

- formulate an ML feasibility question;
- establish trustworthy data inputs;
- choose suitable baselines and candidate models;
- tune models without contaminating the final test set;
- diagnose failures rather than reporting one headline score;
- connect technical evidence to a go, conditional-go, or no-go recommendation;
- organize analytical code as a reproducible engineering workflow;
- communicate limitations honestly to technical and non-technical readers.

The central portfolio story is the quality of the workflow and judgment, not a promise of a high model score.

## 5. System Architecture and Data Flow

```text
Pinned MMLU-Pro revision
        |
        v
Dataset retrieval + source/license manifest + checksum
        |
        v
Schema/content validation + normalization + exclusion register
        |
        v
Exact and near-duplicate grouping + leakage-safe deterministic split
        |
        v
SQLite analytical database
        |
        +--> Named SQL analyses
        |
        +--> EDA and data-quality figures
        |
        v
scikit-learn Pipeline
        +--> Majority baseline
        +--> TF-IDF + Logistic Regression
        +--> TF-IDF + LinearSVC
        |
        v
Five-fold stratified CV + predeclared hyperparameter search
        |
        v
Locked model selection + one-time final-test evaluation
        |
        v
Predictions + metrics + configuration + runtime + model artifact
        |
        v
Error analysis + robustness checks + confidence/coverage analysis
        |
        v
Verified tables + figures + README + Markdown report + PDF report
```

The SQL analyses, figures, README, and report must derive from the same controlled artifacts. A report must not contain manually copied values that cannot be traced to a generated table, metric file, or database query.

## 6. Repository Design

```text
stem-learning-analytics/
├─ README.md
├─ LICENSE
├─ pyproject.toml
├─ .gitignore
├─ configs/
│  └─ project.yaml
├─ data/
│  ├─ README.md
│  ├─ raw/.gitkeep
│  ├─ interim/.gitkeep
│  └─ processed/.gitkeep
├─ docs/
│  ├─ data_dictionary.md
│  ├─ annotation_guide.md
│  ├─ final_report.md
│  ├─ final_report.pdf
│  ├─ limitation_register.md
│  └─ reproducibility.md
├─ sql/
│  ├─ schema.sql
│  └─ analysis/
├─ src/stem_analytics/
│  ├─ __init__.py
│  ├─ __main__.py
│  ├─ cli.py
│  ├─ config.py
│  ├─ data_ingestion.py
│  ├─ data_validation.py
│  ├─ splitting.py
│  ├─ database.py
│  ├─ analysis.py
│  ├─ modeling.py
│  ├─ evaluation.py
│  ├─ error_analysis.py
│  ├─ reporting.py
│  └─ provenance.py
├─ reports/
│  ├─ figures/
│  ├─ tables/
│  └─ qa/
├─ artifacts/
│  ├─ manifests/
│  ├─ metrics/
│  ├─ predictions/
│  ├─ runs/
│  └─ models/
├─ tests/
│  ├─ fixtures/
│  ├─ test_data_validation.py
│  ├─ test_splitting.py
│  ├─ test_database.py
│  ├─ test_modeling.py
│  ├─ test_evaluation.py
│  └─ test_pipeline_integration.py
└─ .github/workflows/ci.yml
```

Core logic belongs in importable Python modules. Notebooks may be added for exploration, but no result should require undocumented manual notebook execution.

## 7. Data Governance and Processing

### 7.1 Source control

The project will use an explicitly pinned MMLU-Pro revision. The data manifest will record:

- source URL and dataset identifier;
- dataset revision or commit identifier;
- retrieval timestamp;
- license and redistribution note;
- original and selected row counts;
- checksums for controlled local inputs;
- all exclusions with reason counts.

Raw data, rebuilt SQLite databases, large models, credentials, and other regenerable large artifacts will not be committed. The repository will retain download and reconstruction instructions.

### 7.2 Allowed inputs

The subject classifier may use normalized `question_text` only. Correct answers, explanations, answer-derived information, manually inserted subject tokens, and post-outcome information are prohibited model features.

### 7.3 Validation and cleaning

Validation must fail clearly for missing required columns, unknown subjects, empty question text, duplicate stable identifiers, a revision mismatch, or an unexpected checksum change.

Cleaning is conservative:

- normalize Unicode and whitespace;
- preserve scientifically meaningful punctuation, units, and formulas;
- reject empty or malformed records with recorded reasons;
- remove exact duplicate questions before splitting;
- identify near-duplicate or repeated-stem groups using a documented text fingerprint or similarity procedure.

### 7.4 Splitting

Use seed `42`. Reserve approximately 15% as a final test set. The remaining records are used for model development and stratified five-fold cross-validation. Duplicate groups must remain within one split so closely related records cannot contaminate evaluation.

The test set is not used for feature decisions, hyperparameter selection, model selection, or threshold tuning. The final model is locked from cross-validation evidence before test evaluation.

## 8. SQLite and SQL Analysis

SQLite will store controlled question records, dataset metadata, model runs, predictions, and evaluation results where relational analysis is useful. Foreign keys and allowed-label constraints must be enforced.

Named SQL files must cover at least:

1. record count and percentage by subject;
2. subject-by-source distribution;
3. missing, malformed, duplicate, and excluded record counts;
4. question-length summary by subject;
5. split distribution and integrity checks;
6. model error rate by true subject;
7. model error rate by source;
8. confusion pairs with the largest counts;
9. error rate by question-length band;
10. Logistic Regression accuracy and coverage by confidence band.

Each query must state which analytical or deployment decision it informs. Generated CSV tables become authoritative report sources.

## 9. Exploratory Analysis and Visualizations

The core figure set will include:

- class distribution;
- subject-by-source composition;
- question-length distributions;
- data-quality and exclusion summary;
- model comparison with the majority baseline;
- final confusion matrix;
- per-class precision, recall, and F1;
- cross-validation fold variation;
- error rate by source and text-length band;
- confidence, automated coverage, and manual-review trade-off;
- top positive class features;
- learning curves.

If source metadata is sufficiently populated, a source-aware holdout result will be shown as a robustness figure or table.

Each stakeholder-facing figure must have a finding-led title, readable labels and units, sample size, source-data note, and a short decision implication. Figures must not imply that benchmark frequencies represent real platform demand.

## 10. Modeling Design

### 10.1 Candidates

1. Majority-class baseline
2. TF-IDF plus multinomial Logistic Regression
3. TF-IDF plus LinearSVC

TF-IDF and preprocessing must be fitted inside a scikit-learn `Pipeline` on training folds only.

### 10.2 Predeclared search space

- TF-IDF word n-grams: `(1, 1)` and `(1, 2)`
- TF-IDF `min_df`: `1`, `2`, and `5`
- Logistic Regression `C`: `0.1`, `1.0`, and `10.0`
- Logistic Regression class weight: `None` and `balanced`
- LinearSVC `C`: `0.1`, `1.0`, and `10.0`
- LinearSVC class weight: `None` and `balanced`

Search uses stratified five-fold cross-validation on development data and selects by mean macro-F1. Runtime is recorded. The search must remain within this declared space unless a deviation is documented before viewing the test result.

### 10.3 Final model selection

The selected model is the candidate with the strongest cross-validated macro-F1, subject to fold stability, minority-class recall, runtime, interpretability, and evidence of source shortcuts. The selection rationale is stored before the test evaluation is run.

## 11. Evaluation and Diagnostic Analysis

### 11.1 Required metrics

- accuracy;
- macro-F1 as the primary comparison metric;
- weighted-F1;
- per-class precision, recall, and F1;
- confusion matrix;
- cross-validation mean and fold variation;
- training and inference runtime;
- bootstrap 95% confidence interval for final-test macro-F1.

The report must compare models with the majority baseline and explain both statistical uncertainty and practical effect size.

### 11.2 Error analysis

Diagnostics include:

- errors by true subject;
- errors by source;
- errors by text-length band;
- major confusion pairs;
- representative correct and incorrect cases;
- Logistic Regression confidence versus accuracy and review volume;
- top weighted features to detect sensible subject cues or source shortcuts;
- learning curves to distinguish likely data limitations from model limitations.

LinearSVC decision-function scores must not be called probabilities. Logistic Regression outputs are described as model confidence unless calibration quality has been explicitly evaluated.

### 11.3 Robustness

When feasible, a source-aware holdout will test whether performance degrades when source patterns change. This result supplements but does not replace the predeclared final-test result.

### 11.4 Feasibility decision

- **Green:** test macro-F1 at least `0.80` and every subject recall at least `0.70`, with no severe robustness warning.
- **Yellow:** macro-F1 from `0.65` to below `0.80`, any subject recall below `0.70`, confidence intervals showing material uncertainty, or notable source dependence.
- **Red:** macro-F1 below `0.65`, material leakage, or evidence that the evaluation is not trustworthy.

The final conclusion must consider uncertainty, class breadth, source robustness, and failure modes rather than applying the overall score mechanically.

## 12. Command-Line Workflow

The primary local entry point is:

```powershell
python -m stem_analytics run-all --config configs/project.yaml
```

The orchestration sequence is:

```text
fetch -> validate -> split -> build-db -> analyze -> train -> evaluate -> report
```

Each stage must also be independently callable so a failure does not require repeating all completed work. Every generated stage records dataset revision, configuration hash, seed, input hashes, creation time, and relevant program version.

Previously generated outputs may be reused only when their provenance matches current inputs and configuration. A mismatch invalidates the dependent stage and requires regeneration.

## 13. Presentation Modes

### 13.1 GitHub presentation

The public repository provides a static, immediately readable portfolio presentation:

- README project summary;
- workflow overview;
- key verified metrics and figures;
- major technical decisions;
- model capability and limitation statement;
- installation and reproduction commands;
- links to the full Markdown and PDF reports;
- CI status.

The README opening section must answer:

1. What problem was investigated?
2. How were the data and models controlled?
3. What was the evidence-backed result?
4. What can and cannot be concluded?

### 13.2 Local presentation

The local demonstration shows that the work is executable, not only documented. A reviewer can:

- install the project;
- run the test suite;
- execute individual pipeline stages or `run-all`;
- inspect generated SQLite tables and named SQL outputs;
- review model rankings and selected hyperparameters;
- regenerate figures and the final report;
- inspect saved predictions and representative errors.

The local demonstration remains terminal/report based. It does not require a Web interface.

## 14. Reporting Design

The English technical report will include:

1. Executive summary
2. Background and feasibility question
3. Scope and non-goals
4. Data source, license, revision, and representativeness
5. Data validation, cleaning, duplicates, and exclusions
6. SQLite design and SQL findings
7. EDA and data-quality risks
8. Validation design and leakage controls
9. Baselines, model candidates, and hyperparameter tuning
10. Cross-validation and final-test results
11. Error, confidence, feature, learning-curve, and robustness analyses
12. Feasibility decision and recommended use
13. Limitations and capability boundaries
14. Reproducibility and run order
15. Intent-annotation future work
16. Appendices for queries, parameters, metrics, and evidence registers

The README, code documentation, figures, and report are primarily English. The README will contain a concise Chinese summary for Taiwan-based interviews.

## 15. Testing and Quality Control

### 15.1 Automated tests

- Unit tests for normalization, validation, exclusion rules, duplicate grouping, splitting, metrics, and SQL execution
- Leakage tests for prohibited features, train-only TF-IDF fitting, and duplicate-group split isolation
- Database integrity tests for keys, label constraints, row counts, and referential integrity
- Determinism tests for stable splits and key outputs under identical inputs, configuration, and seed
- A fixture-based end-to-end integration test
- GitHub Actions checks for tests and linting without downloading the complete dataset or running full hyperparameter tuning

### 15.2 Evidence and numerical consistency

Every major report value records its authoritative source, source value, displayed transformation, tolerance, and verification status. README, report, tables, and figures must agree.

The project will maintain:

- an evidence matrix;
- a numerical consistency register;
- a figure source register;
- a limitation register;
- a report completeness checklist.

### 15.3 PDF QA

After PDF generation, every page is rendered to images and checked for clipping, overlap, unreadable tables, broken glyphs, missing figures, raw markup, and inconsistent page structure. Failed pages are corrected and regenerated before delivery.

### 15.4 Clean-room verification

Before completion, installation, fast tests, and the documented core reconstruction path are verified in a clean environment. Completion claims must cite the verification commands and observed results.

## 16. Error Handling

The pipeline must fail early with actionable messages for:

- unavailable or mismatched dataset revision;
- checksum changes;
- schema changes or missing required fields;
- invalid labels or empty question text;
- split contamination by duplicates;
- forbidden model inputs;
- incompatible cached artifacts;
- missing metrics required by the report;
- disagreement between authoritative result files and displayed numbers.

A failure should preserve already verified upstream artifacts and identify the first stage requiring correction or regeneration.

## 17. Implementation Order

Work follows technical dependencies rather than weeks or people:

1. Repository, environment, CLI, configuration, and test foundation
2. Dataset pinning, manifest, validation, cleaning, and split controls
3. SQLite schema, SQL analysis, and EDA
4. Baseline and candidate pipelines with cross-validation and tuning
5. Locked final-test evaluation
6. Error, confidence, feature, learning-curve, and source-robustness analysis
7. Artifact-driven tables, figures, README, and technical report
8. PDF generation, numerical consistency, and clean-room verification
9. Intent annotation guide and future-work handoff

## 18. Definition of Done

The project is complete only when:

- a new environment can follow the README to install the project and run tests;
- one CLI command can rebuild the core workflow;
- source, revision, license, checksums, and exclusions are documented;
- deterministic splitting and feature processing pass leakage checks;
- the majority baseline, Logistic Regression, and LinearSVC receive a fair comparison;
- tuning uses development data only;
- the final test set does not influence model selection;
- all major conclusions trace to controlled artifacts;
- SQL tables, figures, README, Markdown report, and PDF agree numerically;
- the final report explains failures, uncertainty, limits, and deployment readiness;
- the PDF passes visual QA;
- the public repository excludes credentials, raw data, large regenerable artifacts, and restricted material;
- GitHub and local presentation instructions have been verified;
- no claim is made for a completed intent model, production deployment, or real-student validation.

## 19. Explicit Design Decisions

- Optimize for an end-to-end ML workflow rather than a frontend demo.
- Treat subject classification as the only completed modeling task.
- Defer intent classification until credible human labels exist.
- Prefer interpretable sparse linear models over unnecessary model complexity.
- Add only engineering practices that are supported by the project: modular code, CLI orchestration, provenance, tests, and CI.
- Present finished artifacts on GitHub and provide executable reconstruction locally.
- Use evidence-backed language even when results are disappointing.

