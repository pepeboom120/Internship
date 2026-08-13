# STEM Learning Analytics and Machine Learning Feasibility Study

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task by task. Steps use checkbox syntax for tracking.

**Document type:** Project charter, research plan, and implementation roadmap  
**Status:** Draft for team review  
**Version:** 1.0  
**Date:** 2026-08-12  

**Goal:** Build and evaluate a small, reproducible proof of concept that uses SQL analytics, visualization, and machine learning to classify STEM learning questions and turn the results into clear recommendations for a mixed-background team and non-technical stakeholders.

**Architecture:** The core system ingests a licensed public STEM question dataset, stores normalized records and model outputs in SQLite, produces SQL-based analytics and stakeholder-ready visualizations, and trains two text-classification pipelines. Subject classification uses existing Mathematics, Physics, Chemistry, and Biology labels; intent classification uses a smaller, human-annotated subset. A retrieval-grounded AI response demo is optional and begins only after the core analysis and model report pass review.

**Recommended tech stack:** Python 3.11+, SQLite, pandas, scikit-learn, matplotlib/seaborn or Plotly, Jupyter, pytest, and Markdown. Streamlit and an external or local LLM are optional extensions, not core dependencies.

## Global Constraints

- Treat this as a greenfield implementation. No file or model from a previous prototype is required by this plan.
- Version 1 is English-only and limited to Mathematics, Physics, Chemistry, and Biology.
- SQL analysis, EDA, visualization, subject classification, intent annotation/classification, error analysis, and stakeholder reporting are core scope.
- The AI response and Streamlit demo are optional and cannot begin before the Phase 9 entry gate.
- Use only licensed, documented data; pin the dataset revision and record retrieval date.
- Do not collect or commit personal student data, credentials, API keys, or private stakeholder material.
- Keep the final test set isolated from model selection and tuning.
- Do not describe benchmark data as real user behavior or a prototype as a production system.
- Use random seed `42` for deterministic sampling and splits unless a documented experiment explicitly varies it.
- Final résumé language must be based on completed, attributable work rather than planned deliverables.

## 1. Executive Summary

This project evaluates a practical question:

> Can a small EdTech team use structured data analysis and interpretable machine-learning models to organize incoming STEM questions, identify data priorities, and support a future AI-assisted learning experience?

The project is a feasibility study, not a production deployment. Its required outcome is a defensible recommendation supported by data, model evaluation, error analysis, and clear communication. A successful project does not require every model to exceed a predetermined score. If the data or model is insufficient, identifying why and recommending the next data-collection step is itself a valid and useful result.

The first version is limited to four STEM subjects:

- Mathematics
- Physics
- Chemistry
- Biology

The project has two connected audiences:

1. **Technical audience:** needs reproducible data, SQL, model code, metrics, and limitations.
2. **Non-technical audience:** needs concise charts, plain-language findings, and actionable recommendations.

## 2. Problem Statement

An educational platform may receive questions from multiple STEM subjects, but an early-stage team may not know:

- which subjects dominate the incoming workload;
- whether the data is balanced and suitable for training;
- which records require cleaning or relabeling;
- whether a lightweight classifier can route questions reliably;
- where model errors are concentrated;
- which data or product feature should receive priority;
- whether an AI-generated hint or explanation is feasible and sufficiently grounded.

This study builds a miniature version of that decision process. It will not claim to represent production users or real platform demand unless real interaction data is later supplied.

## 3. Objectives

### 3.1 Required objectives

1. Create a documented and reproducible STEM question dataset.
2. Design a relational SQLite database and demonstrate meaningful SQL analysis.
3. Perform data cleaning, exploratory data analysis, and visualization.
4. Identify class imbalance, source effects, text-quality issues, and other training risks.
5. Train an interpretable subject-classification baseline.
6. Compare at least two suitable text-classification models.
7. Create and document a human annotation process for question intent.
8. Train an exploratory intent classifier using the reviewed annotations.
9. Evaluate models with cross-validation, macro-F1, per-class precision/recall, confusion matrices, and error analysis.
10. Translate technical findings into product and data recommendations.
11. Present the final report to teammates and project stakeholders using language understandable to people without statistics or mathematics backgrounds.

### 3.2 Optional objective

Build a constrained AI response demo that retrieves reviewed reference material and generates a learning hint or explanation. This component must not begin until the core SQL, EDA, subject model, intent model, and stakeholder review are complete.

### 3.3 Non-goals for version 1

- Production deployment or production-level scalability
- User authentication or collection of personal student data
- Image classification or object detection
- Coverage of non-STEM subjects
- Training a large language model from scratch
- Claiming that synthetic or benchmark data represents real user behavior
- Fully automated correctness grading of generated educational answers
- A mobile application or polished commercial frontend

## 4. Research Questions

The final report must answer the following questions:

### Data and SQL

1. How many usable questions are available for each STEM subject?
2. How do question length, source, answer format, and missingness differ by subject?
3. Are there duplicate, near-duplicate, malformed, or potentially leakage-prone records?
4. Which subjects or intent labels are underrepresented?
5. What should be collected, cleaned, or relabeled before a larger training effort?

### Machine learning

6. How much better do trained models perform than simple baselines?
7. Which model offers the best balance of macro-F1, interpretability, speed, and reproducibility?
8. Which subjects and intents have the lowest recall?
9. What errors remain after tuning, and what data issues explain them?
10. Can confidence be used to route uncertain questions to manual review?

### Product feasibility

11. Is automatic routing by STEM subject sufficiently reliable for a proof of concept?
12. Is intent classification feasible with the available human-labeled sample?
13. Which feature or dataset improvement should the team prioritize next?
14. Is a retrieval-grounded AI learning response worth prototyping after the classifier stage?

## 5. Recommended Dataset Strategy

### 5.1 Primary source

Use the four relevant categories from **MMLU-Pro**:

- math: 1,351 questions
- physics: 1,299 questions
- chemistry: 1,132 questions
- biology: 717 questions
- expected maximum before cleaning: 4,499 questions

The published dataset card describes approximately 12,000 questions across 14 domains, includes the four required categories, provides source metadata and answer explanations, and lists an MIT license. Before downloading data, preserve the dataset version or commit identifier and review the source-specific license notes. Sources: [MMLU-Pro dataset card](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) and [MMLU-Pro paper](https://arxiv.org/abs/2406.01574).

### 5.2 Why this is the recommended starting point

- All four required subjects already have subject labels.
- The dataset is large enough for a credible subject-classification experiment.
- Questions come from multiple sources, allowing analysis of source effects.
- It includes question text, answer options, correct answers, category, source, and explanation fields.
- The subject distribution is imperfect, which creates a realistic class-imbalance discussion.

### 5.3 Important limitations

- MMLU-Pro is a benchmark, not real student-submission or platform-interaction data.
- Most records are multiple-choice questions rather than conversational requests.
- The existing category is a subject label, not a question-intent label.
- The included explanation must not leak into classifier inputs.
- Source-specific wording may allow the model to learn shortcuts.
- Dataset updates and corrections mean that the exact version must be recorded.
- Licensing must be checked before redistribution or any future commercial use.

### 5.4 Intent annotation subset

Create a human-reviewed sample of **800 questions**, initially stratified as 200 per subject. Annotate one of the following intents:

1. `conceptual_explanation` — asks why, how, or for conceptual understanding.
2. `calculation_problem_solving` — requires calculation, derivation, or multi-step solving.
3. `definition_factual_recall` — asks for a definition, property, fact, or direct identification.
4. `application_interpretation` — applies knowledge to a scenario, experiment, graph, or observed result.
5. `ambiguous_exclude` — cannot be labeled reliably under the four intended classes.

At least 20% of the sample must be independently labeled by two people. Disagreements must be reviewed during an adjudication meeting. If agreement is weak, revise the label definitions before labeling the remainder.

## 6. Data Model

SQLite is sufficient for version 1 and makes the project easy to reproduce locally.

### 6.1 Core tables

#### `questions`

| Column | Type | Meaning |
|---|---|---|
| `question_id` | INTEGER PRIMARY KEY | Stable local identifier |
| `source_question_id` | TEXT | Identifier from the source dataset |
| `question_text` | TEXT NOT NULL | Input text used for classification |
| `subject` | TEXT NOT NULL | math, physics, chemistry, biology |
| `source_name` | TEXT NOT NULL | Original source metadata |
| `options_json` | TEXT | Serialized answer options |
| `correct_answer` | TEXT | Correct option or answer |
| `explanation_text` | TEXT | Held out from classifier features |
| `text_length` | INTEGER NOT NULL | Character count after normalization |
| `dataset_version` | TEXT NOT NULL | Dataset revision or commit identifier |
| `split_name` | TEXT NOT NULL | train, validation, or test |

#### `intent_annotations`

| Column | Type | Meaning |
|---|---|---|
| `annotation_id` | INTEGER PRIMARY KEY | Annotation identifier |
| `question_id` | INTEGER NOT NULL | Foreign key to questions |
| `annotator_id` | TEXT NOT NULL | Pseudonymous team identifier |
| `intent_label` | TEXT NOT NULL | One of the five annotation values |
| `confidence` | INTEGER NOT NULL | 1–3 annotation confidence |
| `notes` | TEXT | Short rationale for uncertain cases |
| `annotated_at` | TEXT NOT NULL | ISO timestamp |

#### `model_runs`

| Column | Type | Meaning |
|---|---|---|
| `run_id` | INTEGER PRIMARY KEY | Model run identifier |
| `task_name` | TEXT NOT NULL | subject or intent |
| `model_name` | TEXT NOT NULL | majority, logistic_regression, linear_svc |
| `parameters_json` | TEXT NOT NULL | Exact hyperparameters |
| `random_seed` | INTEGER NOT NULL | Reproducibility seed |
| `macro_f1` | REAL NOT NULL | Primary comparison metric |
| `weighted_f1` | REAL NOT NULL | Secondary metric |
| `accuracy` | REAL NOT NULL | Secondary metric |
| `trained_at` | TEXT NOT NULL | ISO timestamp |

#### `predictions`

| Column | Type | Meaning |
|---|---|---|
| `prediction_id` | INTEGER PRIMARY KEY | Prediction identifier |
| `run_id` | INTEGER NOT NULL | Foreign key to model_runs |
| `question_id` | INTEGER NOT NULL | Foreign key to questions |
| `true_label` | TEXT NOT NULL | Evaluation label |
| `predicted_label` | TEXT NOT NULL | Model prediction |
| `confidence` | REAL | Calibrated probability when available |
| `is_correct` | INTEGER NOT NULL | 0 or 1 |

#### `response_feedback` — optional AI demo only

| Column | Type | Meaning |
|---|---|---|
| `feedback_id` | INTEGER PRIMARY KEY | Feedback identifier |
| `question_id` | INTEGER NOT NULL | Question shown in the demo |
| `response_version` | TEXT NOT NULL | Prompt/model configuration |
| `helpfulness_rating` | INTEGER NOT NULL | 1–5 rating |
| `grounded_rating` | INTEGER NOT NULL | 1–5 rating |
| `review_notes` | TEXT | Human-review comments |

### 6.2 Required SQL analyses

The repository must include named `.sql` files that answer at least these questions:

1. Record count and percentage by subject.
2. Record count by subject and source.
3. Missing or empty field counts.
4. Question-length summary by subject.
5. Intent distribution by subject.
6. Annotation disagreements and low-confidence labels.
7. Model error rate by true subject.
8. Model error rate by source.
9. Confusion pairs with the highest counts.
10. Accuracy or error rate by confidence band.
11. Optional AI helpfulness and grounding ratings by subject.

Each query must include a short comment explaining why the result matters to a stakeholder decision.

## 7. EDA and Visualization Plan

### 7.1 Required figures

1. Subject distribution bar chart.
2. Subject-by-source heatmap or grouped bar chart.
3. Question-length distribution by subject.
4. Data-quality summary showing missing, duplicate, and excluded records.
5. Intent distribution by subject after annotation.
6. Subject-model confusion matrix.
7. Per-class precision, recall, and F1 comparison.
8. Model comparison chart with majority baseline, logistic regression, and LinearSVC.
9. Error rate by source or question-length band.
10. Confidence-versus-accuracy or confidence-versus-review-volume chart.

### 7.2 Communication rule for every chart

Every stakeholder-facing figure must include:

- one complete title stating the finding rather than only naming the variables;
- readable labels and units;
- the sample size;
- one or two sentences explaining the decision implication;
- a note when the chart is based on benchmark rather than real user data.

Example title:

> Biology has the fewest training examples and should receive priority in the next data-collection cycle (n = 4,499 before cleaning).

## 8. Machine-Learning Plan

### 8.1 Subject classification

**Input:** normalized `question_text` only.  
**Target:** math, physics, chemistry, biology.  
**Forbidden features:** correct answer, answer explanation, source-derived label tokens intentionally inserted into text, or any post-outcome information.

Models:

1. Majority-class baseline.
2. TF-IDF + multinomial Logistic Regression.
3. TF-IDF + LinearSVC.

The use of TF-IDF and sparse linear classifiers follows a standard, interpretable text-classification approach demonstrated in the [scikit-learn text classification documentation](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html).

### 8.2 Intent classification

**Input:** normalized `question_text`.  
**Target:** four reviewed intent classes; `ambiguous_exclude` is not used for training.  
**Data:** adjudicated subset only.

Use the same baseline and model family as subject classification so the team can compare tasks without introducing unnecessary complexity.

### 8.3 Splitting and leakage control

- Use deterministic random seed `42`.
- Remove exact duplicates before splitting.
- Check near-duplicates and repeated stems before splitting.
- Use stratified train/validation/test splits of approximately 70%/15%/15%.
- Fit TF-IDF only inside a scikit-learn `Pipeline` on training folds.
- Preserve a final test set that is not used for model selection.
- Report source effects and, if feasible, repeat evaluation with a source-aware holdout.

### 8.4 Tuning

Tune only parameters that have a clear purpose:

- TF-IDF: word n-grams `(1,1)` and `(1,2)`; minimum document frequency `1`, `2`, or `5`.
- Logistic Regression: `C` in `{0.1, 1.0, 10.0}`; class weight `None` or `balanced`.
- LinearSVC: `C` in `{0.1, 1.0, 10.0}`; class weight `None` or `balanced`.

Use stratified five-fold cross-validation on the training set. Select by macro-F1, not accuracy.

### 8.5 Evaluation

Required metrics:

- accuracy;
- macro-F1 — primary metric;
- weighted-F1;
- precision, recall, and F1 for every class;
- confusion matrix;
- model runtime;
- error counts by source and text-length band.

For Logistic Regression, evaluate whether predicted probabilities can support a review threshold. Do not call LinearSVC decision-function values probabilities unless calibrated.

### 8.6 Decision gates

These are feasibility gates, not promises:

- **Subject routing green:** test macro-F1 at or above 0.80 and every subject recall at or above 0.70.
- **Subject routing yellow:** macro-F1 from 0.65 to below 0.80, or one subject recall below 0.70. Recommend more data or manual review.
- **Subject routing red:** macro-F1 below 0.65. Do not recommend automatic routing.
- **Intent exploratory green:** macro-F1 at or above 0.65 with acceptable annotation agreement.
- **Intent exploratory yellow/red:** lower performance or weak agreement triggers taxonomy revision and additional labeling rather than forced deployment.

## 9. Optional Retrieval-Grounded AI Response

This component answers: “After a question is classified, can the system provide a useful learning hint without behaving like an unrestricted chatbot?”

### 9.1 Proposed flow

1. Receive a STEM question.
2. Predict subject and intent.
3. Retrieve a small number of reviewed reference explanations from the matching subject.
4. Send only the question, predicted labels, and retrieved evidence to the response model.
5. Return structured output:
   - short hint;
   - concise explanation;
   - one check-for-understanding question;
   - reference identifiers;
   - uncertainty or refusal when evidence is insufficient.

### 9.2 Guardrails

- Keep the response demo separate from classifier evaluation.
- Use only a curated reference subset with recorded licenses and attribution.
- Do not expose API keys in the repository.
- Do not send personal or real student information to an external API.
- Require human review of at least 40 stratified responses.
- Score correctness, grounding, clarity, and helpfulness separately.
- Do not claim the generated response is educationally validated.

OpenStax may be considered for noncommercial reference content, but its current textbook licensing is CC BY-NC-SA and requires attribution, noncommercial use, and share-alike treatment for adaptations. Review the license again at implementation time: [OpenStax licensing information](https://help.openstax.org/s/article/Licensing-information-of-OpenStax-textbooks).

## 10. Stakeholder Communication Plan

### 10.1 Review checkpoints

1. **Scope review:** confirm user scenario, four subjects, dataset, and success criteria.
2. **Data review:** present data quality, imbalance, and annotation guidance before training.
3. **Model review:** explain baseline, selected metrics, confusion matrix, and error examples.
4. **Recommendation review:** agree on go/no-go assessment, limitations, and next priority.
5. **Final presentation:** deliver the feasibility report and prototype demonstration.

### 10.2 Meeting artifacts

- Agenda sent before each review.
- Decision log recording what changed and why.
- Short meeting notes with owner and next action.
- Charts exported as PNG or SVG for reuse in slides and the report.
- Questions from non-technical participants recorded and addressed in the final report.

### 10.3 Plain-language translation examples

Avoid:

> The classifier achieved macro-F1 of 0.78 with heterogeneous per-class recall.

Prefer:

> The model performs reasonably overall, but it misses Biology questions more often than the other subjects. We should collect and review more Biology examples before using the model for automatic routing.

## 11. Deliverables

### Core deliverables

1. Versioned Git repository with installation and reproduction instructions.
2. Dataset manifest and license/source register.
3. SQLite schema and populated local database build script.
4. Ten or more meaningful SQL queries.
5. EDA notebook or reproducible analysis scripts.
6. Required stakeholder-ready figures.
7. Annotation guide and reviewed intent-label dataset.
8. Subject and intent classification pipelines.
9. Model comparison and error-analysis report.
10. Final feasibility report in Markdown, with PDF export optional.
11. Eight-to-ten-slide stakeholder presentation.
12. Decision log and limitation register.

### Optional deliverables

1. Retrieval-grounded AI response module.
2. Small Streamlit demonstration.
3. Human-review rubric and response-evaluation results.

## 12. Recommended Repository Structure

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
│  ├─ project_plan.md
│  ├─ data_dictionary.md
│  ├─ annotation_guide.md
│  ├─ decision_log.md
│  ├─ limitation_register.md
│  ├─ final_report.md
│  └─ stakeholder_presentation.md
├─ sql/
│  ├─ schema.sql
│  └─ analysis/
│     ├─ 01_subject_distribution.sql
│     ├─ 02_source_distribution.sql
│     ├─ 03_data_quality.sql
│     ├─ 04_text_length.sql
│     ├─ 05_intent_distribution.sql
│     ├─ 06_annotation_disagreement.sql
│     ├─ 07_model_errors_by_class.sql
│     ├─ 08_model_errors_by_source.sql
│     ├─ 09_confusion_pairs.sql
│     └─ 10_confidence_bands.sql
├─ src/stem_analytics/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ data_ingestion.py
│  ├─ data_validation.py
│  ├─ database.py
│  ├─ annotation.py
│  ├─ features.py
│  ├─ train.py
│  ├─ evaluate.py
│  ├─ reporting.py
│  └─ ai_response.py
├─ notebooks/
│  ├─ 01_data_audit.ipynb
│  ├─ 02_eda.ipynb
│  └─ 03_error_analysis.ipynb
├─ reports/
│  ├─ figures/.gitkeep
│  └─ tables/.gitkeep
├─ tests/
│  ├─ test_data_ingestion.py
│  ├─ test_data_validation.py
│  ├─ test_database.py
│  ├─ test_annotation.py
│  ├─ test_features.py
│  ├─ test_train.py
│  └─ test_evaluate.py
└─ app/
   └─ streamlit_app.py
```

`app/streamlit_app.py` and `src/stem_analytics/ai_response.py` remain absent until the optional extension is approved.

## 13. Implementation Roadmap

### Phase 0: Team alignment and project contract

**Output:** approved scope, roles, decision log, and target presentation date.

- [ ] Confirm team members and assign one owner for data/SQL, modeling, annotation coordination, and stakeholder reporting.
- [ ] Confirm that version 1 is English-only and limited to four subjects.
- [ ] Confirm MMLU-Pro as the initial dataset and record its exact revision.
- [ ] Agree on a repository owner, contribution rules, and how individual contributions will be documented.
- [ ] Approve the four intent labels and `ambiguous_exclude` rule.
- [ ] Set dates for the five review checkpoints.
- [ ] Record open decisions in `docs/decision_log.md`.

**Acceptance criteria:** every required role has an owner; the project scope and presentation target are written; no team member believes version 1 includes production deployment or image recognition.

### Phase 1: Repository and reproducibility setup

**Output:** a clean repository that installs and runs tests.

- [ ] Create the directory structure from Section 12.
- [ ] Define Python dependencies and exact supported version range in `pyproject.toml`.
- [ ] Add `.gitignore` rules for environments, raw data, databases, model artifacts, and secrets.
- [ ] Add a smoke test that imports `stem_analytics`.
- [ ] Add a `README.md` quick start using a clean virtual environment.
- [ ] Run `pytest -q`; expected result is all smoke tests passing.

**Acceptance criteria:** another teammate can clone the repository, install it, and run tests without receiving private files.

### Phase 2: Data acquisition, manifest, and validation

**Output:** deterministic, validated four-subject dataset.

Required interface contracts:

- `load_source_dataset(revision: str) -> pandas.DataFrame`
- `filter_stem_subjects(df: pandas.DataFrame) -> pandas.DataFrame`
- `validate_question_data(df: pandas.DataFrame) -> dict[str, int]`
- `assign_splits(df: pandas.DataFrame, seed: int = 42) -> pandas.DataFrame`

- [ ] Write tests using a five-row fixture with one invalid subject, one empty question, and one duplicate.
- [ ] Verify tests fail before implementation.
- [ ] Implement dataset loading with a pinned revision.
- [ ] Filter to math, physics, chemistry, and biology.
- [ ] Normalize only whitespace and Unicode; preserve scientifically meaningful punctuation and formulas.
- [ ] Detect empty text, exact duplicates, invalid labels, and missing source identifiers.
- [ ] Create deterministic stratified splits after duplicate removal.
- [ ] Save a manifest with source URL, revision, retrieval date, original row count, final row count, exclusions, and license note.
- [ ] Run validation tests and manually review at least ten records per subject.

**Acceptance criteria:** processed data contains only the four subjects; exclusions are explainable; identical input and seed produce identical splits.

### Phase 3: SQLite database and SQL analysis

**Output:** reproducible database plus stakeholder-relevant SQL outputs.

Required interface contracts:

- `create_database(db_path: str, schema_path: str) -> None`
- `load_questions(db_path: str, frame: pandas.DataFrame) -> int`
- `execute_named_query(db_path: str, query_path: str) -> pandas.DataFrame`

- [ ] Create `sql/schema.sql` with foreign keys and constraints matching Section 6.
- [ ] Write database tests for row counts, duplicate primary keys, invalid subjects, and broken foreign keys.
- [ ] Verify tests fail before implementation.
- [ ] Implement database creation and question loading in one transaction.
- [ ] Implement the ten required named SQL analyses.
- [ ] Export query results to `reports/tables/` as CSV.
- [ ] Review each query with at least one teammate and record what decision it informs.

**Acceptance criteria:** rebuilding the database twice produces the same counts; foreign-key checks return no violations; every SQL output has a documented stakeholder purpose.

### Phase 4: EDA and first stakeholder review

**Output:** reproducible figures and a data-priority recommendation.

Required interface contract:

- `build_eda_figures(db_path: str, output_dir: str) -> list[str]`

- [ ] Write a test confirming expected figure filenames are created from a small fixture database.
- [ ] Build the first five figures from Section 7.
- [ ] Include sample size and benchmark-data disclaimer in captions.
- [ ] Identify at least three concrete data risks.
- [ ] Produce a one-page non-technical summary recommending what to clean, label, or collect first.
- [ ] Present the summary to the team and record questions or requested changes.
- [ ] Revise the analysis based on valid feedback.

**Acceptance criteria:** the team can state the primary imbalance, primary data-quality risk, and recommended next data action without needing to read code.

### Phase 5: Intent annotation pilot

**Output:** reviewed annotation guide and reliable labeled subset.

Required interface contracts:

- `sample_for_annotation(df: pandas.DataFrame, n_per_subject: int, seed: int = 42) -> pandas.DataFrame`
- `validate_annotations(df: pandas.DataFrame) -> dict[str, object]`
- `compute_agreement(df: pandas.DataFrame) -> float`

- [ ] Draft `docs/annotation_guide.md` with definitions, inclusions, exclusions, and at least two examples per intent.
- [ ] Select a 40-question pilot balanced by subject.
- [ ] Have at least two teammates label the same pilot independently.
- [ ] Compute raw agreement and Cohen’s kappa where applicable.
- [ ] Discuss disagreements and revise ambiguous rules.
- [ ] Label the 800-question subset, double-labeling at least 20%.
- [ ] Store annotations in SQLite and export an adjudicated file.
- [ ] Exclude unresolved `ambiguous_exclude` cases from intent-model training.

**Acceptance criteria:** the guide has no unresolved label contradictions; agreement is reported honestly; every training label has an audit trail.

### Phase 6: Subject-classification baseline and model comparison

**Output:** reproducible baseline, tuned models, test metrics, and predictions in SQLite.

Required interface contracts:

- `build_pipeline(model_name: str, class_weight: str | None = None) -> sklearn.pipeline.Pipeline`
- `tune_model(X_train, y_train, model_name: str, seed: int = 42) -> sklearn.model_selection.GridSearchCV`
- `evaluate_model(model, X_test, y_test) -> dict[str, object]`
- `save_run(db_path: str, metrics: dict, predictions: pandas.DataFrame) -> int`

- [ ] Write unit tests confirming TF-IDF is fitted inside the pipeline and forbidden fields are never used.
- [ ] Add a majority-class baseline.
- [ ] Train Logistic Regression and LinearSVC pipelines.
- [ ] Tune only the parameter grid in Section 8.4.
- [ ] Select by cross-validated macro-F1.
- [ ] Evaluate once on the untouched test split.
- [ ] Save run metadata and per-record predictions to SQLite.
- [ ] Generate the required model-comparison, class-metric, and confusion-matrix figures.

**Acceptance criteria:** the report can reproduce every metric from saved configuration and data; the final test set was not used for tuning; model performance is compared with the majority baseline.

### Phase 7: Intent-classification experiment

**Output:** exploratory intent model and taxonomy recommendation.

- [ ] Build deterministic splits using only adjudicated labels.
- [ ] Reuse the tested pipeline family from Phase 6.
- [ ] Compare majority baseline, Logistic Regression, and LinearSVC.
- [ ] Report macro-F1, per-class recall, confusion matrix, and annotation agreement together.
- [ ] Review at least 20 incorrect predictions with the annotation team.
- [ ] Decide whether errors indicate insufficient data, overlapping intent definitions, or model limitations.
- [ ] Recommend whether to retain, merge, redefine, or expand intent classes.

**Acceptance criteria:** the intent result is described as exploratory; model limitations are not separated from annotation limitations; taxonomy recommendations cite observed error patterns.

### Phase 8: Decision analysis and stakeholder recommendation

**Output:** evidence-backed feasibility recommendation.

- [ ] Join questions, predictions, model runs, and annotations through SQL.
- [ ] Analyze errors by subject, intent, source, length, and confidence band.
- [ ] Select five representative correct cases and five representative errors.
- [ ] Estimate how different confidence thresholds change automated coverage and manual-review volume.
- [ ] Classify subject and intent components using the green/yellow/red gates in Section 8.6.
- [ ] Recommend one immediate data action, one model action, and one product action.
- [ ] Present results to mixed-background teammates and revise unclear explanations.

**Acceptance criteria:** recommendations follow directly from recorded evidence; limitations include benchmark-to-real-world transfer risk; non-technical teammates can explain the recommended next step.

### Phase 9: Optional AI response experiment

**Entry gate:** Phases 1–8 are complete and the team approves the extension.

**Output:** constrained demonstration and human-review results.

Required interface contract:

- `generate_learning_response(question: str, predicted_subject: str, predicted_intent: str, references: list[dict[str, str]]) -> dict[str, object]`

- [ ] Select and license-check a small reference collection.
- [ ] Implement retrieval independently from response generation.
- [ ] Require structured output with hint, explanation, check question, references, and uncertainty flag.
- [ ] Add tests ensuring empty evidence produces a refusal or uncertainty response.
- [ ] Review 40 responses stratified across subjects and intents.
- [ ] Store helpfulness, grounding, correctness, and clarity ratings.
- [ ] Report failures and decide whether the feature should proceed.

**Acceptance criteria:** responses contain traceable references; no secrets or personal data are committed; the report clearly separates human-rated demo results from classifier metrics.

### Phase 10: Final report and presentation

**Output:** complete technical report and stakeholder presentation.

- [ ] Draft the report using Section 14.
- [ ] Link each major claim to a table, figure, SQL result, or model artifact.
- [ ] Run all tests and rebuild data, database, figures, and metrics from a clean environment.
- [ ] Conduct a numerical consistency check across README, report, tables, and slides.
- [ ] Conduct a plain-language review with a teammate outside statistics or mathematics.
- [ ] Present the final report to project supervisors and partner stakeholders.
- [ ] Record questions, feedback, decisions, and follow-up actions.
- [ ] Update the limitation register and project conclusion after the presentation.

**Acceptance criteria:** a new reader can reproduce the analysis; all numbers agree; stakeholder feedback is documented separately from the team’s own conclusions.

## 14. Final Report Structure

1. **Executive Summary** — question, main finding, recommendation, and most important limitation.
2. **Background and Use Case** — who the proposed system serves and why classification matters.
3. **Scope and Non-goals** — four subjects, proof-of-concept status, no production claims.
4. **Data Sources and Governance** — dataset version, license, fields, exclusions, and representativeness.
5. **Database and SQL Analysis** — schema, key queries, and decision relevance.
6. **Data Preparation and EDA** — cleaning, imbalance, source effects, and visual findings.
7. **Intent Annotation** — taxonomy, annotators, agreement, adjudication, and limitations.
8. **Modeling Methodology** — baselines, features, splits, cross-validation, tuning, and leakage control.
9. **Results** — metrics, model comparison, confusion matrices, and runtime.
10. **Error and Limitation Analysis** — representative failures and real-world transfer risks.
11. **Stakeholder Recommendations** — data, model, and product priorities.
12. **Optional AI Response Evaluation** — only if completed.
13. **Conclusion and Next Phase** — go/no-go decision and required next evidence.
14. **Appendices** — data dictionary, SQL queries, hyperparameters, annotation guide, and reproducibility instructions.

## 15. Eight-Week Suggested Schedule

| Week | Focus | Review gate |
|---|---|---|
| 1 | Team alignment, repository, dataset revision, scope | Scope approved |
| 2 | Data validation, SQLite schema, SQL queries | Data build reproducible |
| 3 | EDA, visualization, data-priority briefing | Team agrees on data risks |
| 4 | Annotation pilot and guide revision | Label definitions approved |
| 5 | Complete annotations and subject models | Subject results reviewed |
| 6 | Intent models, tuning, error analysis | Model conclusions approved |
| 7 | Decision analysis and report draft; optional AI starts only if approved | Recommendation approved |
| 8 | Final report, presentation, rehearsal, stakeholder delivery | Final feedback recorded |

If annotation or team availability is limited, extend to ten weeks rather than reducing review quality.

## 16. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Benchmark questions differ from real student inputs | Results may not generalize | Label project as feasibility study; recommend later pilot with real consented data |
| Class imbalance, especially Biology | Lower minority-class recall | Use macro-F1, stratified splits, class-weight comparison, and collection recommendation |
| Source-specific wording creates shortcuts | Inflated test results | Analyze by source and attempt source-aware holdout |
| Intent labels overlap | Weak agreement and unstable models | Pilot, double annotation, adjudication, and taxonomy revision |
| Explanations leak answers or labels | Unrealistic classifier results | Prohibit answer and explanation fields from model features |
| Team members interpret metrics differently | Poor decision-making | Plain-language summaries, examples, and review checkpoints |
| Optional AI feature consumes the schedule | Core analysis remains incomplete | Enforce Phase 9 entry gate |
| External API cost or privacy concerns | Demo blocked or unsafe | Provider abstraction, local alternative, no personal data, budget approval |
| Dataset licensing changes | Redistribution risk | Pin version, maintain source register, recheck license before release |
| Resume claims exceed actual work | Credibility risk | Write final bullets only after deliverables and individual contributions are verified |

## 17. Decisions Still Required

### Must be decided before Week 1 ends

1. **Target learner level**  
   Recommended default: high-school through introductory university STEM, matching the source material more closely than elementary education.

2. **Team size and named owners**  
   Recommended default: three to four contributors with one accountable owner per required deliverable.

3. **Final presentation target and date**  
   Decide who will receive the report, expected duration, and whether a live demo is useful.

4. **Repository ownership and contribution policy**  
   Decide the organization/account, branch workflow, code review requirement, and attribution method.

5. **Intent taxonomy approval**  
   Confirm that the four proposed labels are understandable to all annotators before full labeling.

6. **Data redistribution policy**  
   Decide whether processed records will be committed, downloaded by script, or omitted from public release.

### Can be decided after the core model review

7. **Whether to build the optional AI response.**
8. **External API versus local model and maximum budget.**
9. **Whether a Streamlit demo adds value beyond report figures.**
10. **Whether to add a second dataset or real anonymized interaction data.**
11. **Whether to export the final report to PDF.**

## 18. Definition of Done

The core project is complete only when:

- [ ] the repository can be installed and tested from a clean environment;
- [ ] dataset version, source, license, and exclusions are documented;
- [ ] SQLite tables rebuild deterministically;
- [ ] ten required SQL analyses run successfully;
- [ ] required EDA and model figures are reproducible;
- [ ] the intent annotation process and agreement are documented;
- [ ] majority baseline, Logistic Regression, and LinearSVC are compared;
- [ ] tuning uses training data only and the final test set remains untouched until evaluation;
- [ ] macro-F1 and per-class metrics are reported with error analysis;
- [ ] conclusions include benchmark-data and generalization limitations;
- [ ] non-technical teammates review the plain-language explanation;
- [ ] the final report and presentation are delivered to the intended stakeholders;
- [ ] stakeholder questions and follow-up decisions are recorded;
- [ ] individual contributions are documented before résumé bullets are finalized.

The optional AI response is complete only if its separate entry gate and human-review criteria are met. It is not required for the core project to be considered successful.
