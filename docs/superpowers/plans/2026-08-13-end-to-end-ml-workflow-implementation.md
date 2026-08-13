# End-to-End STEM Subject Classification ML Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a public, reproducible machine-learning portfolio project that classifies MMLU-Pro questions into Mathematics, Physics, Chemistry, or Biology and produces evidence-grounded SQL analysis, model evaluation, error diagnosis, figures, README, Markdown report, and verified PDF.

**Architecture:** A Python CLI orchestrates immutable dataset retrieval, schema validation, duplicate-aware splitting, SQLite analysis, scikit-learn model selection, locked final-test evaluation, artifact-driven reporting, and PDF generation. Every downstream claim is generated from versioned configuration and traceable artifacts; GitHub presents finished evidence while local commands reproduce the workflow.

**Tech Stack:** Python 3.11–3.12, `datasets`, pandas, NumPy, scikit-learn, SQLite, PyYAML, Pydantic, matplotlib, seaborn, joblib, Jinja2, ReportLab, pypdf, pytest, Ruff, and GitHub Actions.

## Global Constraints

- Work in repository root `C:\Users\twpow\Desktop\RBC`; do not create a nested repository.
- Pin MMLU-Pro to Hugging Face revision `b189ec765aa7ed75c8acfea42df31fdae71f97be`.
- Load the official `test` split as a source pool, filter to `math`, `physics`, `chemistry`, and `biology`, then create project-specific development and final-test splits.
- Use only normalized `question` text as model input. Never use `options`, `answer`, `answer_index`, `cot_content`, `src`, or `category` as model features.
- Use random seed `42` throughout unless a robustness experiment explicitly records another seed.
- Keep near-duplicate groups within one split and one cross-validation fold.
- Select models using grouped five-fold cross-validated macro-F1. Do not use final-test results for model or threshold selection.
- Treat intent classification as future work. Do not train or report an intent model.
- Do not add a Web UI, REST API, Kubernetes, or microservices.
- Do not commit raw data, processed datasets, SQLite databases, trained models, credentials, or large regenerable outputs.
- Public documentation is English-first; add a concise Chinese summary to `README.md`.
- Every major result in README, Markdown report, and PDF must trace to a controlled CSV, JSON, SQL output, or manifest.
- Preserve `STEM_learning_analytics_ML_project_plan.md` and the approved design spec unchanged as historical inputs.

---

## Planned File Map

| Path | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, dependency bounds, CLI entry point, pytest and Ruff configuration |
| `configs/project.yaml` | Immutable dataset revision, labels, split, duplicate threshold, model grids, bootstrap, and paths |
| `src/stem_analytics/config.py` | Validated configuration types and canonical config hashing |
| `src/stem_analytics/provenance.py` | File hashes, artifact envelopes, cache-validity checks, run identifiers |
| `src/stem_analytics/data_ingestion.py` | Pinned Hugging Face retrieval and source manifest generation |
| `src/stem_analytics/data_validation.py` | Schema checks, normalization, exclusions, exact and near-duplicate grouping |
| `src/stem_analytics/splitting.py` | Deterministic grouped development/final-test split and integrity checks |
| `src/stem_analytics/database.py` | SQLite construction, transactional loads, named-query execution |
| `src/stem_analytics/analysis.py` | SQL-table export and reproducible EDA figures |
| `src/stem_analytics/modeling.py` | Baseline, scikit-learn pipelines, grouped CV, hyperparameter search |
| `src/stem_analytics/evaluation.py` | Locked evaluation, metrics, bootstrap interval, prediction artifact |
| `src/stem_analytics/error_analysis.py` | Error slices, confidence/coverage, features, learning curves, robustness |
| `src/stem_analytics/reporting.py` | Evidence registers, report context, Markdown and PDF source generation |
| `src/stem_analytics/cli.py` | Stage commands and `run-all` orchestration |
| `sql/schema.sql` | Relational schema and constraints |
| `sql/analysis/*.sql` | Ten decision-oriented SQL analyses |
| `templates/final_report.md.j2` | Artifact-driven technical-report template |
| `scripts/build_pdf.py` | ReportLab PDF builder from verified report context |
| `scripts/verify_clean_room.ps1` | Windows clean-environment verification |
| `tests/fixtures/` | Small deterministic source and annotation fixtures |
| `reports/` | Generated figures, tables, and QA registers suitable for GitHub |
| `artifacts/` | Locally generated manifests, runs, predictions, and models; ignored except documentation |

---

### Task 1: Initialize the Repository and Python Package

**Files:**
- Create: `.gitignore`
- Create: `LICENSE`
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `src/stem_analytics/__init__.py`
- Create: `src/stem_analytics/__main__.py`
- Create: `src/stem_analytics/cli.py`
- Create: `tests/test_package.py`
- Create: empty tracked directory markers under `data/`, `reports/`, and `artifacts/`

**Interfaces:**
- Produces CLI entry point `stem-analytics = stem_analytics.cli:main`.
- Produces `python -m stem_analytics --help` and `stem-analytics --help`.
- Establishes Python support `>=3.11,<3.13`.

- [ ] **Step 1: Initialize Git without altering existing files**

Run:

```powershell
git init -b main
git status --short
```

Expected: repository initializes on `main`; the original plan and `docs/` appear as untracked files.

- [ ] **Step 2: Write the failing package smoke test**

```python
# tests/test_package.py
from stem_analytics import __version__
from stem_analytics.cli import build_parser


def test_package_exposes_version_and_cli() -> None:
    assert __version__ == "0.1.0"
    assert build_parser().prog == "stem-analytics"
```

Run: `python -m pytest tests/test_package.py -q`

Expected: FAIL because the package and dependencies do not exist.

- [ ] **Step 3: Create package metadata and dependency bounds**

Use this dependency structure in `pyproject.toml`:

```toml
[project]
name = "stem-learning-analytics"
version = "0.1.0"
requires-python = ">=3.11,<3.13"
dependencies = [
  "datasets>=4.0,<5",
  "joblib>=1.4,<2",
  "jinja2>=3.1,<4",
  "matplotlib>=3.9,<4",
  "numpy>=2.0,<3",
  "pandas>=2.2,<3",
  "pydantic>=2.10,<3",
  "pyyaml>=6.0,<7",
  "reportlab>=4.2,<5",
  "scikit-learn>=1.6,<2",
  "seaborn>=0.13,<1",
]

[project.optional-dependencies]
dev = ["pypdf>=5,<7", "pytest>=8,<9", "pytest-cov>=6,<8", "ruff>=0.9,<1"]

[project.scripts]
stem-analytics = "stem_analytics.cli:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"

[tool.ruff]
line-length = 100
target-version = "py311"
```

Implement `__version__ = "0.1.0"`, `build_parser()`, and a `main()` that prints help when no subcommand is supplied.

- [ ] **Step 4: Add public-safe ignore rules and directory documentation**

`.gitignore` must exclude:

```gitignore
.venv/
.verify-venv/
__pycache__/
.pytest_cache/
.ruff_cache/
*.py[cod]
.env
data/raw/*
data/interim/*
data/processed/*
artifacts/predictions/*
artifacts/models/*
artifacts/runs/**/*.joblib
*.joblib
*.sqlite
*.sqlite3
```

Add negated `.gitkeep` and README exceptions. Use an MIT project-code license while clearly stating that upstream dataset licensing and attribution remain separate.

- [ ] **Step 5: Install and verify the foundation**

Run:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m pytest tests/test_package.py -q
.\.venv\Scripts\python -m stem_analytics --help
```

Expected: smoke test PASS and help output begins with `usage: stem-analytics`.

- [ ] **Step 6: Commit the foundation**

```powershell
git add .gitignore LICENSE README.md pyproject.toml src tests data reports artifacts docs STEM_learning_analytics_ML_project_plan.md
git commit -m "chore: initialize reproducible ML project"
```

---

### Task 2: Add Validated Configuration and Provenance

**Files:**
- Create: `configs/project.yaml`
- Create: `src/stem_analytics/config.py`
- Create: `src/stem_analytics/provenance.py`
- Create: `tests/test_config.py`
- Create: `tests/test_provenance.py`

**Interfaces:**
- Produces `load_config(path: Path) -> ProjectConfig`.
- Produces `config_hash(config: ProjectConfig) -> str`.
- Produces `sha256_file(path: Path) -> str`.
- Produces `write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None`.
- Produces `artifact_is_current(meta_path: Path, expected: Mapping[str, Any]) -> bool`.

- [ ] **Step 1: Write failing configuration tests**

```python
def test_project_config_pins_dataset_and_seed(tmp_path: Path) -> None:
    cfg = load_config(Path("configs/project.yaml"))
    assert cfg.dataset.revision == "b189ec765aa7ed75c8acfea42df31fdae71f97be"
    assert cfg.dataset.source_split == "test"
    assert cfg.subjects == ["math", "physics", "chemistry", "biology"]
    assert cfg.random_seed == 42
    assert cfg.split.final_test_folds == 7


def test_config_hash_is_key_order_independent() -> None:
    first = canonical_sha256({"b": 2, "a": 1})
    second = canonical_sha256({"a": 1, "b": 2})
    assert first == second
```

Run: `.\.venv\Scripts\python -m pytest tests/test_config.py tests/test_provenance.py -q`

Expected: FAIL because configuration types and hashing do not exist.

- [ ] **Step 2: Define the exact project configuration**

`configs/project.yaml` must contain:

```yaml
project_name: stem-learning-analytics
random_seed: 42
subjects: [math, physics, chemistry, biology]
dataset:
  name: TIGER-Lab/MMLU-Pro
  revision: b189ec765aa7ed75c8acfea42df31fdae71f97be
  source_split: test
  required_columns: [question_id, question, options, answer, answer_index, cot_content, category, src]
split:
  final_test_folds: 7
  cv_folds: 5
duplicates:
  char_ngram_range: [3, 5]
  cosine_similarity_threshold: 0.92
models:
  tfidf_ngram_ranges: [[1, 1], [1, 2]]
  tfidf_min_df: [1, 2, 5]
  c_values: [0.1, 1.0, 10.0]
  class_weights: [null, balanced]
evaluation:
  primary_metric: macro_f1
  bootstrap_iterations: 2000
paths:
  raw_data: data/raw/mmlu_pro_source.parquet
  processed_data: data/processed/questions.parquet
  database: data/processed/stem_analytics.sqlite
  artifacts: artifacts
  reports: reports
```

Use strict Pydantic models with `extra="forbid"` so misspelled keys fail.

- [ ] **Step 3: Implement deterministic provenance helpers**

Canonical hashing must JSON-serialize with sorted keys and compact separators. Atomic JSON writes must write to a sibling temporary file and replace the target only after serialization succeeds. Artifact metadata must compare stage name, config hash, input hashes, dataset revision, and package version.

- [ ] **Step 4: Run tests and lint**

Run:

```powershell
.\.venv\Scripts\python -m pytest tests/test_config.py tests/test_provenance.py -q
.\.venv\Scripts\python -m ruff check src tests
```

Expected: all tests PASS and Ruff reports no errors.

- [ ] **Step 5: Commit configuration and provenance**

```powershell
git add configs src/stem_analytics/config.py src/stem_analytics/provenance.py tests/test_config.py tests/test_provenance.py
git commit -m "feat: add validated configuration and provenance"
```

---

### Task 3: Implement Pinned Dataset Retrieval and Source Manifest

**Files:**
- Create: `src/stem_analytics/data_ingestion.py`
- Create: `tests/fixtures/mmlu_pro_sample.jsonl`
- Create: `tests/test_data_ingestion.py`
- Create: `data/README.md`

**Interfaces:**
- Produces `load_source_dataset(config: ProjectConfig) -> pd.DataFrame`.
- Produces `select_stem_subjects(frame: pd.DataFrame, subjects: Sequence[str]) -> pd.DataFrame`.
- Produces `build_source_manifest(frame: pd.DataFrame, config: ProjectConfig, retrieved_at: datetime) -> dict[str, Any]`.
- Produces CLI stage `stem-analytics fetch --config configs/project.yaml`.

- [ ] **Step 1: Create a nine-row fixture and failing tests**

The fixture must use the official columns and contain two rows per target subject plus one non-target row. Use synthetic question text; do not copy benchmark questions.

```python
def test_select_stem_subjects_preserves_official_ids(sample_source_frame: pd.DataFrame) -> None:
    selected = select_stem_subjects(sample_source_frame, TARGET_SUBJECTS)
    assert set(selected["category"]) == set(TARGET_SUBJECTS)
    assert selected["question_id"].is_unique
    assert len(selected) == 8


def test_manifest_records_immutable_source(sample_source_frame: pd.DataFrame) -> None:
    manifest = build_source_manifest(sample_source_frame, config, FIXED_TIME)
    assert manifest["dataset_revision"] == config.dataset.revision
    assert manifest["source_split"] == "test"
    assert manifest["retrieved_at"] == "2026-08-13T00:00:00+00:00"
```

Run: `.\.venv\Scripts\python -m pytest tests/test_data_ingestion.py -q`

Expected: FAIL because ingestion functions do not exist.

- [ ] **Step 2: Implement pinned Hugging Face retrieval**

Call:

```python
dataset = load_dataset(
    config.dataset.name,
    revision=config.dataset.revision,
    split=config.dataset.source_split,
)
frame = dataset.to_pandas()
```

Verify all required columns before saving. Sort by `question_id`, filter the four categories, write the source pool to configured Parquet, and write `artifacts/manifests/source_manifest.json` atomically.

- [ ] **Step 3: Add the `fetch` CLI command**

The command accepts `--config`, prints dataset name, immutable revision, source rows, selected rows, and output paths, and exits non-zero on network, revision, schema, or write failure. It must never fall back from the configured revision to `main`.

- [ ] **Step 4: Run fixture tests and one real retrieval smoke test**

Run:

```powershell
.\.venv\Scripts\python -m pytest tests/test_data_ingestion.py -q
.\.venv\Scripts\stem-analytics fetch --config configs/project.yaml
```

Expected: tests PASS; real retrieval records revision `b189ec765aa7ed75c8acfea42df31fdae71f97be`, selects 4,499 target rows before cleaning if the pinned revision matches the official dataset card, and creates the Parquet plus manifest. If the selected count differs, stop and reconcile against the pinned source rather than changing the assertion silently.

- [ ] **Step 5: Document data licensing and redistribution**

`data/README.md` must explain that MMLU-Pro is loaded from Hugging Face, identify the pinned revision, link the official dataset card and paper, record the dataset card's MIT license, warn that source metadata and upstream terms must be reviewed, and state why raw/processed data are not committed.

- [ ] **Step 6: Commit ingestion**

```powershell
git add src/stem_analytics/data_ingestion.py tests/fixtures tests/test_data_ingestion.py data/README.md src/stem_analytics/cli.py artifacts/manifests/source_manifest.json
git commit -m "feat: add pinned MMLU-Pro ingestion"
```

---

### Task 4: Validate, Clean, Group Duplicates, and Split Safely

**Files:**
- Create: `src/stem_analytics/data_validation.py`
- Create: `src/stem_analytics/splitting.py`
- Create: `tests/test_data_validation.py`
- Create: `tests/test_splitting.py`

**Interfaces:**
- Produces `normalize_question(text: str) -> str`.
- Produces `validate_and_clean(frame: pd.DataFrame, config: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame]` where the second frame is the exclusion register.
- Produces `assign_duplicate_groups(frame: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame` with `duplicate_group_id`.
- Produces `assign_project_split(frame: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame` with `split_name` in `development|test`.
- Produces `assert_split_integrity(frame: pd.DataFrame) -> None`.
- Produces CLI stage `stem-analytics validate`.

- [ ] **Step 1: Write failing normalization and exclusion tests**

```python
def test_normalization_preserves_formula_punctuation() -> None:
    assert normalize_question("  If  E = mc²,\n why?  ") == "If E = mc², why?"


def test_validation_records_reason_for_each_exclusion(invalid_frame: pd.DataFrame) -> None:
    clean, excluded = validate_and_clean(invalid_frame, config)
    assert set(excluded["exclusion_reason"]) == {
        "empty_question", "invalid_subject", "duplicate_question_id"
    }
    assert clean["question_id"].is_unique
```

- [ ] **Step 2: Write failing duplicate and split-integrity tests**

```python
def test_near_duplicates_share_group_and_split(near_duplicate_frame: pd.DataFrame) -> None:
    grouped = assign_duplicate_groups(near_duplicate_frame, config)
    split = assign_project_split(grouped, config)
    pair = split.loc[split["question_id"].isin([101, 102])]
    assert pair["duplicate_group_id"].nunique() == 1
    assert pair["split_name"].nunique() == 1


def test_integrity_rejects_group_contamination(valid_split: pd.DataFrame) -> None:
    contaminated = valid_split.copy()
    repeated_group = contaminated["duplicate_group_id"].value_counts().loc[lambda s: s >= 2].index[0]
    member_index = contaminated.index[contaminated["duplicate_group_id"] == repeated_group][0]
    current = contaminated.loc[member_index, "split_name"]
    contaminated.loc[member_index, "split_name"] = "test" if current == "development" else "development"
    with pytest.raises(ValueError, match="duplicate group crosses splits"):
        assert_split_integrity(contaminated)
```

Run: `.\.venv\Scripts\python -m pytest tests/test_data_validation.py tests/test_splitting.py -q`

Expected: FAIL because cleaning and splitting functions do not exist.

- [ ] **Step 3: Implement conservative cleaning and exact deduplication**

Require official columns, normalize Unicode with NFKC and collapse whitespace, retain punctuation, derive `question_text` and `text_length`, and create stable local IDs from sorted `question_id`. Keep the first exact normalized question deterministically and record later copies as `exact_duplicate` exclusions.

- [ ] **Step 4: Implement near-duplicate connected components**

Build character TF-IDF with `analyzer="char_wb"` and configured 3–5 grams. Use cosine-radius neighbors with radius `1 - 0.92`. Combine qualifying pairs with union-find; assign the minimum member `question_id` as `duplicate_group_id`. Record group size and maximum observed similarity for audit.

- [ ] **Step 5: Implement deterministic grouped split**

Use `StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)` with `category` as `y` and `duplicate_group_id` as groups. Assign fold zero to final `test` and the other folds to `development`. Verify each subject appears in both partitions and no group crosses partitions.

- [ ] **Step 6: Save controlled artifacts and validation summary**

`validate` writes:

- `data/processed/questions.parquet`;
- `reports/tables/exclusion_register.csv`;
- `artifacts/manifests/validation_manifest.json` containing source count, clean count, exclusion counts, per-subject counts, duplicate groups, split counts, hashes, and threshold.

- [ ] **Step 7: Run tests and the real validation stage**

Run:

```powershell
.\.venv\Scripts\python -m pytest tests/test_data_validation.py tests/test_splitting.py -q
.\.venv\Scripts\stem-analytics validate --config configs/project.yaml
```

Expected: tests PASS; integrity check reports zero duplicate groups crossing development and test.

- [ ] **Step 8: Commit validation and splitting**

```powershell
git add src/stem_analytics/data_validation.py src/stem_analytics/splitting.py src/stem_analytics/cli.py tests reports/tables/exclusion_register.csv artifacts/manifests/validation_manifest.json
git commit -m "feat: add leakage-safe validation and splitting"
```

---

### Task 5: Build SQLite and Decision-Oriented SQL Analyses

**Files:**
- Create: `sql/schema.sql`
- Create: `sql/analysis/01_subject_distribution.sql`
- Create: `sql/analysis/02_source_distribution.sql`
- Create: `sql/analysis/03_data_quality.sql`
- Create: `sql/analysis/04_text_length.sql`
- Create: `sql/analysis/05_split_integrity.sql`
- Create: `sql/analysis/06_model_errors_by_class.sql`
- Create: `sql/analysis/07_model_errors_by_source.sql`
- Create: `sql/analysis/08_confusion_pairs.sql`
- Create: `sql/analysis/09_errors_by_length.sql`
- Create: `sql/analysis/10_confidence_bands.sql`
- Create: `src/stem_analytics/database.py`
- Create: `tests/test_database.py`

**Interfaces:**
- Produces `create_database(db_path: Path, schema_path: Path) -> None`.
- Produces `load_questions(db_path: Path, frame: pd.DataFrame) -> int`.
- Produces `load_model_run(db_path: Path, run: Mapping[str, Any], predictions: pd.DataFrame) -> int`.
- Produces `execute_named_query(db_path: Path, query_path: Path) -> pd.DataFrame`.
- Produces CLI stage `stem-analytics build-db`.

- [ ] **Step 1: Write failing integrity tests**

```python
def test_database_enforces_subject_and_foreign_keys(tmp_path: Path) -> None:
    db = tmp_path / "test.sqlite"
    create_database(db, Path("sql/schema.sql"))
    with sqlite3.connect(db) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(INVALID_SUBJECT_INSERT)


def test_load_questions_is_transactional(tmp_path: Path, clean_frame: pd.DataFrame) -> None:
    db = tmp_path / "test.sqlite"
    create_database(db, Path("sql/schema.sql"))
    assert load_questions(db, clean_frame) == len(clean_frame)
```

Run: `.\.venv\Scripts\python -m pytest tests/test_database.py -q`

Expected: FAIL because schema and database helpers do not exist.

- [ ] **Step 2: Implement the relational schema**

Create `dataset_manifests`, `questions`, `model_runs`, and `predictions`. Enforce:

```sql
subject TEXT NOT NULL CHECK(subject IN ('math','physics','chemistry','biology')),
split_name TEXT NOT NULL CHECK(split_name IN ('development','test')),
FOREIGN KEY(run_id) REFERENCES model_runs(run_id),
FOREIGN KEY(question_id) REFERENCES questions(question_id)
```

Store `source_question_id`, `question_text`, `source_name`, `text_length`, `duplicate_group_id`, dataset revision, and split. Store model parameters and metrics as canonical JSON plus searchable scalar metrics.

- [ ] **Step 3: Implement transactional database construction**

Rebuilding writes to a sibling temporary SQLite file, enables foreign keys, loads all rows in one transaction, executes `PRAGMA foreign_key_check`, and atomically replaces the configured database only after checks pass.

- [ ] **Step 4: Implement ten named SQL queries**

Each file begins with `-- Decision:` and a plain-language stakeholder purpose. Queries 6–10 may return zero rows before predictions are loaded but must run successfully. Confidence bands use `[0.0,0.5)`, `[0.5,0.6)`, `[0.6,0.7)`, `[0.7,0.8)`, `[0.8,0.9)`, and `[0.9,1.0]`.

- [ ] **Step 5: Add query-export behavior**

`build-db` constructs the database and exports every SQL result to `reports/tables/<query_stem>.csv` using stable sort order and UTF-8 encoding.

- [ ] **Step 6: Run tests and inspect database integrity**

Run:

```powershell
.\.venv\Scripts\python -m pytest tests/test_database.py -q
.\.venv\Scripts\stem-analytics build-db --config configs/project.yaml
.\.venv\Scripts\python -c "import sqlite3; c=sqlite3.connect('data/processed/stem_analytics.sqlite'); print(c.execute('PRAGMA foreign_key_check').fetchall())"
```

Expected: tests PASS and foreign-key output is `[]`.

- [ ] **Step 7: Commit database and SQL**

```powershell
git add sql src/stem_analytics/database.py src/stem_analytics/cli.py tests/test_database.py reports/tables
git commit -m "feat: add SQLite analytics and named SQL queries"
```

---

### Task 6: Generate Reproducible EDA Tables and Figures

**Files:**
- Create: `src/stem_analytics/analysis.py`
- Create: `tests/test_analysis.py`
- Generate: `reports/figures/01_subject_distribution.png`
- Generate: `reports/figures/02_subject_by_source.png`
- Generate: `reports/figures/03_question_length.png`
- Generate: `reports/figures/04_data_quality.png`
- Generate: `reports/figures/05_split_distribution.png`
- Generate: `reports/qa/figure_source_register.csv`

**Interfaces:**
- Produces `build_eda_figures(db_path: Path, output_dir: Path) -> list[FigureRecord]`.
- Produces `FigureRecord(number, title, path, source_table, sample_size, verification_status)`.
- Produces CLI stage `stem-analytics analyze`.

- [ ] **Step 1: Write a failing figure-generation test**

```python
def test_build_eda_figures_creates_expected_files(fixture_db: Path, tmp_path: Path) -> None:
    records = build_eda_figures(fixture_db, tmp_path)
    assert [record.path.name for record in records] == [
        "01_subject_distribution.png",
        "02_subject_by_source.png",
        "03_question_length.png",
        "04_data_quality.png",
        "05_split_distribution.png",
    ]
    assert all(record.sample_size > 0 for record in records)
```

Run: `.\.venv\Scripts\python -m pytest tests/test_analysis.py -q`

Expected: FAIL because the analysis module does not exist.

- [ ] **Step 2: Implement deterministic chart styling**

Set matplotlib backend to `Agg`, fixed figure sizes, 150 DPI, colorblind-safe colors, and explicit category order `math, physics, chemistry, biology`. Every chart must include a finding-led title, `n`, axis labels, units, and footer: `Benchmark questions; not real student demand.`

- [ ] **Step 3: Build figures from exported SQL tables**

Do not query ad hoc data separately for captions. Use named SQL outputs as figure sources and store each source CSV path, SHA-256, row count, figure path, title, and verification status in `figure_source_register.csv`.

- [ ] **Step 4: Run tests and build real EDA**

Run:

```powershell
.\.venv\Scripts\python -m pytest tests/test_analysis.py -q
.\.venv\Scripts\stem-analytics analyze --config configs/project.yaml
```

Expected: test PASS; five non-empty PNGs and a complete figure source register exist.

- [ ] **Step 5: Commit EDA code and public-safe results**

```powershell
git add src/stem_analytics/analysis.py src/stem_analytics/cli.py tests/test_analysis.py reports/figures reports/qa/figure_source_register.csv reports/tables
git commit -m "feat: add reproducible EDA and figure provenance"
```

---

### Task 7: Implement Baselines, Pipelines, and Grouped Hyperparameter Search

**Files:**
- Create: `src/stem_analytics/modeling.py`
- Create: `tests/test_modeling.py`

**Interfaces:**
- Produces `build_pipeline(model_name: Literal["logistic_regression", "linear_svc"], seed: int = 42) -> Pipeline`.
- Produces `parameter_grid(model_name: str, config: ProjectConfig) -> dict[str, list[Any]]`.
- Produces `build_cv(frame: pd.DataFrame, config: ProjectConfig) -> StratifiedGroupKFold`.
- Produces `fit_candidates(frame: pd.DataFrame, config: ProjectConfig) -> ModelSelectionResult`.
- `ModelSelectionResult` contains baseline metrics, candidate CV summaries, best estimator, selection rationale, and exact search results.
- Produces CLI stage `stem-analytics train`.

- [ ] **Step 1: Write failing pipeline and feature-safety tests**

```python
def test_pipeline_fits_tfidf_inside_pipeline() -> None:
    pipeline = build_pipeline("logistic_regression", seed=42)
    assert list(pipeline.named_steps) == ["tfidf", "classifier"]
    assert isinstance(pipeline.named_steps["tfidf"], TfidfVectorizer)


def test_training_uses_question_text_only(clean_frame: pd.DataFrame) -> None:
    X, y, groups = modeling.prepare_xyg(clean_frame)
    assert X.name == "question_text"
    assert X.tolist() == clean_frame["question_text"].tolist()
    assert y.tolist() == clean_frame["category"].tolist()
    assert groups.tolist() == clean_frame["duplicate_group_id"].tolist()
```

- [ ] **Step 2: Write failing grid and grouped-CV tests**

```python
def test_parameter_grid_matches_predeclared_space(config: ProjectConfig) -> None:
    grid = parameter_grid("linear_svc", config)
    assert grid["tfidf__ngram_range"] == [(1, 1), (1, 2)]
    assert grid["tfidf__min_df"] == [1, 2, 5]
    assert grid["classifier__C"] == [0.1, 1.0, 10.0]
    assert grid["classifier__class_weight"] == [None, "balanced"]


def test_grouped_cv_never_splits_duplicate_group(clean_frame: pd.DataFrame, config) -> None:
    X, y, groups = prepare_xyg(clean_frame)
    for train_idx, valid_idx in build_cv(clean_frame, config).split(X, y, groups):
        assert set(groups.iloc[train_idx]).isdisjoint(set(groups.iloc[valid_idx]))
```

Run: `.\.venv\Scripts\python -m pytest tests/test_modeling.py -q`

Expected: FAIL because modeling functions do not exist.

- [ ] **Step 3: Implement baseline and candidate pipelines**

Use `DummyClassifier(strategy="most_frequent")`, `LogisticRegression(max_iter=5000, random_state=42)`, and `LinearSVC(random_state=42)`. Use `TfidfVectorizer(sublinear_tf=True, strip_accents=None, lowercase=True)` inside both candidate pipelines.

- [ ] **Step 4: Implement grouped model selection**

Use `GridSearchCV(scoring="f1_macro", cv=StratifiedGroupKFold(5, shuffle=True, random_state=42), n_jobs=-1, return_train_score=True, refit=True)`. Pass duplicate groups to `.fit`. Record per-fold validation scores, mean, standard deviation, fit time, score time, parameters, and rank for every candidate.

- [ ] **Step 5: Make the selection decision before final-test evaluation**

Write `artifacts/runs/<run_id>/selection_decision.json` containing selected model, selected parameter set, CV macro-F1 mean/std, per-class CV recall, runtime, config hash, development-data hash, and an evidence-based rationale. The `evaluate` command must refuse to run if this file is missing or its hashes do not match current inputs.

- [ ] **Step 6: Run tests and train on the real development partition**

Run:

```powershell
.\.venv\Scripts\python -m pytest tests/test_modeling.py -q
.\.venv\Scripts\stem-analytics train --config configs/project.yaml
```

Expected: tests PASS; the run directory contains `cv_results.csv`, `selection_decision.json`, and development-fitted candidate artifacts.

- [ ] **Step 7: Commit modeling code and compact CV evidence**

Commit code and small CSV/JSON evidence; do not commit `.joblib` model files.

```powershell
git add src/stem_analytics/modeling.py src/stem_analytics/cli.py tests/test_modeling.py artifacts/runs reports/tables
git commit -m "feat: add grouped model selection and tuning"
```

---

### Task 8: Lock and Evaluate the Final Model

**Files:**
- Create: `src/stem_analytics/evaluation.py`
- Create: `tests/test_evaluation.py`
- Generate: `reports/tables/model_comparison.csv`
- Generate: `reports/tables/per_class_metrics.csv`
- Generate: `artifacts/metrics/final_test_metrics.json`
- Generate: `artifacts/predictions/final_test_predictions.csv`

**Interfaces:**
- Produces `classification_metrics(y_true, y_pred, labels) -> dict[str, Any]`.
- Produces `stratified_bootstrap_macro_f1(y_true, y_pred, labels, iterations=2000, seed=42) -> tuple[float, float]`.
- Produces `evaluate_locked_model(selection: SelectionDecision, frame: pd.DataFrame, config: ProjectConfig) -> EvaluationResult`.
- Produces CLI stage `stem-analytics evaluate`.

- [ ] **Step 1: Write failing metric tests with known values**

```python
def test_classification_metrics_reports_macro_and_per_class() -> None:
    result = classification_metrics(
        ["math", "math", "physics", "physics"],
        ["math", "physics", "physics", "physics"],
        ["math", "physics"],
    )
    assert result["accuracy"] == pytest.approx(0.75)
    assert result["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert set(result["per_class"]) == {"math", "physics"}


def test_bootstrap_interval_is_deterministic() -> None:
    first = stratified_bootstrap_macro_f1(Y_TRUE, Y_PRED, LABELS, 200, 42)
    second = stratified_bootstrap_macro_f1(Y_TRUE, Y_PRED, LABELS, 200, 42)
    assert first == second
```

- [ ] **Step 2: Write failing lock-enforcement tests**

```python
def test_evaluation_rejects_changed_config(selection_decision, changed_config) -> None:
    with pytest.raises(ValueError, match="selection decision does not match"):
        evaluate_locked_model(selection_decision, test_frame, changed_config)
```

Run: `.\.venv\Scripts\python -m pytest tests/test_evaluation.py -q`

Expected: FAIL because evaluation functions do not exist.

- [ ] **Step 3: Implement metrics and stratified bootstrap**

Calculate accuracy, macro-F1, weighted-F1, per-class precision/recall/F1/support, and confusion matrix with fixed label order. For each bootstrap iteration, sample with replacement inside each true class, concatenate indices, compute macro-F1, and return percentile 2.5% and 97.5% bounds.

- [ ] **Step 4: Implement one-time locked evaluation**

Verify selection-decision hashes, refit the exact selected pipeline on all development rows, predict final-test rows, and store predictions with `question_id`, true label, predicted label, confidence if available, correctness, source, length, duplicate group, run ID, and model name. If final metrics already exist with matching hashes, report them without reevaluating; `--force` must be an explicit CLI option recorded in metadata.

- [ ] **Step 5: Build a fair model-comparison table without reopening selection**

Use grouped cross-validation results for the majority baseline and both tuned candidates. Populate final-test columns only for the preselected locked model, label the other test cells `not_evaluated`, and explain that this preserves the test set as a one-time confirmation rather than another selection surface. Do not switch the final designation after seeing test results.

- [ ] **Step 6: Persist to SQLite and re-export SQL results**

Load run metadata and predictions transactionally, rerun SQL queries 6–10, and update their CSVs. Verify row counts equal final-test prediction counts.

- [ ] **Step 7: Run tests and evaluation**

Run:

```powershell
.\.venv\Scripts\python -m pytest tests/test_evaluation.py -q
.\.venv\Scripts\stem-analytics evaluate --config configs/project.yaml
```

Expected: tests PASS; final metrics, predictions, comparison table, per-class table, and database rows agree.

- [ ] **Step 8: Commit evaluation code and public-safe evidence**

```powershell
git add src/stem_analytics/evaluation.py src/stem_analytics/database.py src/stem_analytics/cli.py tests/test_evaluation.py reports/tables artifacts/metrics
git commit -m "feat: add locked final-test evaluation"
```

---

### Task 9: Add Error Analysis, Confidence Trade-offs, Features, Learning Curves, and Robustness

**Files:**
- Create: `src/stem_analytics/error_analysis.py`
- Create: `tests/test_error_analysis.py`
- Generate: `reports/tables/error_slices.csv`
- Generate: `reports/tables/representative_cases.csv`
- Generate: `reports/tables/top_features.csv`
- Generate: `reports/tables/learning_curve.csv`
- Generate: `reports/tables/source_holdout_results.csv`
- Generate: model-result figures `06` through `12`

**Interfaces:**
- Produces `build_error_slices(predictions: pd.DataFrame) -> pd.DataFrame`.
- Produces `confidence_coverage_curve(predictions: pd.DataFrame, thresholds: Sequence[float]) -> pd.DataFrame`.
- Produces `extract_linear_features(pipeline: Pipeline, labels: Sequence[str], top_n: int = 20) -> pd.DataFrame`.
- Produces `compute_learning_curve(frame: pd.DataFrame, selected: SelectionDecision, config: ProjectConfig) -> pd.DataFrame`.
- Produces `run_source_holdout(frame: pd.DataFrame, selected: SelectionDecision, config: ProjectConfig) -> pd.DataFrame`.
- Produces `build_model_figures(...) -> list[FigureRecord]`.

- [ ] **Step 1: Write failing slice and confidence tests**

```python
def test_error_slices_include_subject_source_and_length(prediction_frame) -> None:
    slices = build_error_slices(prediction_frame)
    assert set(slices["slice_type"]) >= {"subject", "source", "length_band"}
    assert slices["n"].sum() >= len(prediction_frame)


def test_confidence_threshold_reduces_coverage(prediction_frame) -> None:
    curve = confidence_coverage_curve(prediction_frame, [0.0, 0.7, 0.9])
    assert curve["automated_coverage"].is_monotonic_decreasing
    assert (curve["manual_review_volume"] == len(prediction_frame) - curve["automated_count"]).all()
```

- [ ] **Step 2: Write failing feature and learning-curve tests**

```python
def test_feature_table_has_both_directions(fitted_logistic_pipeline) -> None:
    table = extract_linear_features(fitted_logistic_pipeline, TARGET_SUBJECTS, top_n=3)
    assert set(table["direction"]) == {"positive", "negative"}
    assert table.groupby(["subject", "direction"]).size().eq(3).all()
```

Run: `.\.venv\Scripts\python -m pytest tests/test_error_analysis.py -q`

Expected: FAIL because diagnostic functions do not exist.

- [ ] **Step 3: Implement deterministic diagnostic tables**

Use fixed length bands `<100`, `100–199`, `200–399`, `400+` characters. Confidence thresholds are `0.00, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95`. Representative cases select five correct and five incorrect examples deterministically by confidence and class coverage, and include question excerpts capped at 240 characters.

- [ ] **Step 4: Implement top-feature analysis**

For linear classifiers, map coefficients to TF-IDF feature names and export the top 20 positive and negative features per subject. Add flags when a feature exactly matches normalized source names or appears disproportionately in one source, so the report can discuss shortcut risk.

- [ ] **Step 5: Implement grouped learning curves**

Use development fractions `0.2, 0.4, 0.6, 0.8, 1.0`, grouped stratified folds, and the locked parameter set. Record training and validation macro-F1 mean/std. The final test set remains untouched.

- [ ] **Step 6: Implement source-aware robustness**

For each source with at least 80 rows and all four target subjects represented, hold that source out, train the locked pipeline on all other development sources, and report macro-F1/per-class recall on the held-out source. If no source meets the rule, write a one-row table with `status=not_feasible` and the exact eligibility reason; do not fabricate results.

- [ ] **Step 7: Generate model-result figures**

Create:

```text
06_model_comparison.png
07_confusion_matrix.png
08_per_class_metrics.png
09_cv_fold_variation.png
10_error_slices.png
11_confidence_coverage.png
12_learning_curve.png
```

Register every figure source and checksum.

- [ ] **Step 8: Run tests and diagnostics**

Run:

```powershell
.\.venv\Scripts\python -m pytest tests/test_error_analysis.py -q
.\.venv\Scripts\stem-analytics diagnose --config configs/project.yaml
```

Expected: tests PASS; diagnostic tables and figures exist; robustness status is either verified or explicitly not feasible.

- [ ] **Step 9: Commit diagnostics**

```powershell
git add src/stem_analytics/error_analysis.py src/stem_analytics/cli.py tests/test_error_analysis.py reports
git commit -m "feat: add model diagnostics and robustness analysis"
```

---

### Task 10: Implement Stage Orchestration, Cache Validation, and End-to-End Test

**Files:**
- Modify: `src/stem_analytics/cli.py`
- Modify: `src/stem_analytics/provenance.py`
- Create: `tests/test_pipeline_integration.py`

**Interfaces:**
- Produces subcommands `fetch`, `validate`, `build-db`, `analyze`, `train`, `evaluate`, `diagnose`, `report`, and `run-all`.
- Produces `run_stage(stage: Stage, config: ProjectConfig, force: bool = False) -> StageResult`.
- Produces stage order `fetch -> validate -> build-db -> analyze -> train -> evaluate -> diagnose -> report`.

- [ ] **Step 1: Write a failing fixture-based end-to-end test**

```python
def test_run_all_builds_expected_fixture_outputs(tmp_path: Path, fixture_config: Path) -> None:
    result = cli.main(["run-all", "--config", str(fixture_config)])
    assert result == 0
    for relative in [
        "data/processed/questions.parquet",
        "data/processed/stem_analytics.sqlite",
        "artifacts/metrics/final_test_metrics.json",
        "reports/tables/model_comparison.csv",
        "reports/figures/06_model_comparison.png",
    ]:
        assert (tmp_path / relative).exists()
```

The fixture config uses local JSONL data, a reduced grid, two CV folds, and 50 bootstrap iterations so CI remains fast.

- [ ] **Step 2: Write cache invalidation tests**

```python
def test_changed_input_invalidates_downstream_stage(stage_meta: Path) -> None:
    assert artifact_is_current(stage_meta, ORIGINAL_EXPECTATION)
    assert not artifact_is_current(stage_meta, CHANGED_INPUT_HASH)
```

Run: `.\.venv\Scripts\python -m pytest tests/test_pipeline_integration.py -q`

Expected: FAIL because orchestration is incomplete.

- [ ] **Step 3: Implement stage contracts and dependency checks**

Each stage declares input artifacts, output artifacts, and metadata path. A current stage may be skipped with a visible `CACHED` message. If an upstream hash changes, dependent stages rerun. `--force` reruns the requested stage and all downstream stages, and records the reason.

- [ ] **Step 4: Implement consistent failure behavior**

Catch known configuration, dataset, schema, split, database, model-lock, and reporting exceptions at the CLI boundary. Print `ERROR [stage]: actionable message` to stderr and return exit code `2`. Unexpected exceptions retain tracebacks under `--debug`.

- [ ] **Step 5: Run integration and complete test suites**

Run:

```powershell
.\.venv\Scripts\python -m pytest tests/test_pipeline_integration.py -q
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m ruff check src tests
```

Expected: all tests PASS and lint succeeds.

- [ ] **Step 6: Commit orchestration**

```powershell
git add src/stem_analytics/cli.py src/stem_analytics/provenance.py tests/test_pipeline_integration.py tests/fixtures
git commit -m "feat: orchestrate the reproducible ML workflow"
```

---

### Task 11: Generate Evidence Registers, Markdown Report, and Portfolio README

**Files:**
- Create: `src/stem_analytics/reporting.py`
- Create: `templates/final_report.md.j2`
- Create: `tests/test_reporting.py`
- Create: `docs/data_dictionary.md`
- Create: `docs/limitation_register.md`
- Create: `docs/reproducibility.md`
- Generate: `docs/final_report.md`
- Generate: `reports/qa/evidence_matrix.csv`
- Generate: `reports/qa/numerical_consistency_register.csv`
- Generate: `reports/qa/report_completeness_register.csv`
- Replace: `README.md` with artifact-driven portfolio content

**Interfaces:**
- Produces `build_report_context(config: ProjectConfig) -> dict[str, Any]`.
- Produces `build_evidence_matrix(context) -> pd.DataFrame`.
- Produces `verify_numerical_consistency(context) -> pd.DataFrame` and raises on unresolved major discrepancies.
- Produces `render_markdown_report(context, template_path, output_path) -> Path`.
- Produces CLI stage `stem-analytics report`.

- [ ] **Step 1: Write failing evidence and report tests**

```python
def test_major_claims_have_authoritative_sources(report_context) -> None:
    matrix = build_evidence_matrix(report_context)
    major = matrix.loc[matrix["claim_priority"] == "major"]
    assert major["authoritative_source"].notna().all()
    assert major["verification_status"].eq("verified").all()


def test_report_rejects_metric_disagreement(report_context) -> None:
    report_context["readme_metrics"]["macro_f1"] += 0.01
    with pytest.raises(ValueError, match="numerical consistency"):
        verify_numerical_consistency(report_context)
```

Run: `.\.venv\Scripts\python -m pytest tests/test_reporting.py -q`

Expected: FAIL because reporting functions do not exist.

- [ ] **Step 2: Implement authoritative-source routing**

Use this priority:

1. locked final metric JSON;
2. generated result CSVs;
3. selection decision JSON;
4. validation/source manifests;
5. SQLite named-query exports;
6. figure register.

Every report claim records claim text, metric/conclusion, source, source priority, source value, displayed value, transformation, tolerance, verification status, and discrepancy note.

- [ ] **Step 3: Implement the report template**

Render the 16 chapters in the approved design. Define macro-F1, precision, recall, weighted-F1, confusion matrix, bootstrap interval, confidence/coverage, and learning curve before interpreting them. Every major result paragraph must state value, direction, meaning, uncertainty, and whether it changed the feasibility decision.

- [ ] **Step 4: Implement capability classification and decision language**

Classify the finished system as `validated offline prototype` only if clean-room reproduction and final evaluation pass; otherwise use `offline research prototype`. Explicitly state that it cannot validate real student behavior, calibrated educational decisions, production readiness, or guaranteed generalization.

- [ ] **Step 5: Build the GitHub README**

README order:

1. one-paragraph English summary;
2. concise Chinese summary;
3. final feasibility result and top three metrics;
4. workflow diagram;
5. two to four representative figures;
6. key technical decisions and leakage controls;
7. error findings and limits;
8. repository map;
9. local quick start and stage commands;
10. report links, data/license note, and reproducibility status.

All displayed numbers must come from the report context; do not hand-type metrics.

- [ ] **Step 6: Run report tests and generate documentation**

Run:

```powershell
.\.venv\Scripts\python -m pytest tests/test_reporting.py -q
.\.venv\Scripts\stem-analytics report --config configs/project.yaml
```

Expected: tests PASS; Markdown report, README, evidence matrix, numerical register, and completeness register are generated with no unresolved major item.

- [ ] **Step 7: Commit report sources and generated public artifacts**

```powershell
git add src/stem_analytics/reporting.py templates tests/test_reporting.py README.md docs reports/qa reports/figures reports/tables
git commit -m "docs: add evidence-grounded report and portfolio README"
```

---

### Task 12: Build and Visually Verify the Publication PDF

**Required skill during execution:** Use `pdf:pdf` before creating or validating the PDF.

**Files:**
- Create: `scripts/build_pdf.py`
- Create: `tests/test_pdf_report.py`
- Generate: `docs/final_report.pdf`
- Generate: `reports/qa/pdf_page_audit.csv`
- Generate: `reports/qa/pdf_qa_report.md`
- Generate locally: `reports/qa/pdf_pages/*.png` (do not commit unless needed to document a defect)

**Interfaces:**
- Produces `build_pdf(context_path: Path, output_path: Path) -> Path`.
- Produces `audit_pdf(pdf_path: Path, render_dir: Path) -> pd.DataFrame`.
- Requires title page, table of contents, numbered headings, page numbers, consistent header/footer, figure/table captions, source notes, and appendices.

- [ ] **Step 1: Read the PDF skill and bundled dependency locations**

Read `pdf:pdf` completely, call the workspace-dependency locator, and use its Poppler/Python paths. Record the actual commands in `docs/reproducibility.md`.

- [ ] **Step 2: Write failing structural PDF tests**

```python
def test_pdf_has_expected_structure(generated_pdf: Path) -> None:
    reader = PdfReader(generated_pdf)
    assert len(reader.pages) >= 12
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert "Executive Summary" in text
    assert "Model Capability Boundaries" in text
    assert "Reproducibility" in text
    assert "Intent Annotation Future Work" in text
```

Run: `.\.venv\Scripts\python -m pytest tests/test_pdf_report.py -q`

Expected: FAIL because the PDF does not exist.

- [ ] **Step 3: Implement ReportLab document generation**

Build the PDF from the same verified JSON report context used by Markdown. Use Platypus paragraph/table/image flowables, repeated table headers, landscape pages only for wide appendices, bookmarks for numbered chapters, and captions that include source file and verification status.

- [ ] **Step 4: Render every PDF page to PNG**

Use Poppler `pdftoppm -png -r 150 docs/final_report.pdf reports/qa/pdf_pages/page`. Confirm rendered image count equals pypdf page count.

- [ ] **Step 5: Inspect every rendered page and record QA**

Check clipping, overlap, table overflow, broken glyphs, missing images, blank charts, raw Markdown, unreadable labels, and orphaned headings. Record one row per page in `pdf_page_audit.csv` with `page_number`, `status`, `issues`, and `resolution`. Correct the source and repeat build/render/audit until every row is `pass`.

- [ ] **Step 6: Run PDF tests and checksum the final file**

Run:

```powershell
.\.venv\Scripts\python -m pytest tests/test_pdf_report.py -q
.\.venv\Scripts\python -c "from pathlib import Path; import hashlib; p=Path('docs/final_report.pdf'); print(hashlib.sha256(p.read_bytes()).hexdigest())"
```

Expected: structural tests PASS; QA report contains no unresolved page failure.

- [ ] **Step 7: Commit the PDF and QA evidence**

```powershell
git add scripts/build_pdf.py tests/test_pdf_report.py docs/final_report.pdf reports/qa/pdf_page_audit.csv reports/qa/pdf_qa_report.md docs/reproducibility.md
git commit -m "docs: publish and verify technical report PDF"
```

---

### Task 13: Document Intent Annotation as Honest Future Work

**Files:**
- Create: `docs/annotation_guide.md`
- Create: `src/stem_analytics/annotation.py`
- Create: `tests/test_annotation.py`
- Generate: `reports/tables/intent_pilot_sample.csv` from the local controlled dataset; commit only if upstream redistribution terms permit, otherwise commit a schema-only example.

**Interfaces:**
- Produces `sample_intent_pilot(frame: pd.DataFrame, n_per_subject: int = 10, seed: int = 42) -> pd.DataFrame`.
- Produces `validate_annotation_sheet(frame: pd.DataFrame) -> dict[str, Any]`.
- No model-training interface is permitted in this task.

- [ ] **Step 1: Write failing sampling and validation tests**

```python
def test_intent_pilot_is_balanced_and_deterministic(clean_frame) -> None:
    first = sample_intent_pilot(clean_frame, n_per_subject=2, seed=42)
    second = sample_intent_pilot(clean_frame, n_per_subject=2, seed=42)
    assert first["question_id"].tolist() == second["question_id"].tolist()
    assert first.groupby("subject").size().eq(2).all()


def test_annotation_validation_rejects_unknown_intent(annotation_frame) -> None:
    annotation_frame.loc[0, "intent_label"] = "unknown"
    result = validate_annotation_sheet(annotation_frame)
    assert result["valid"] is False
```

- [ ] **Step 2: Write the annotation guide**

Define `conceptual_explanation`, `calculation_problem_solving`, `definition_factual_recall`, `application_interpretation`, and `ambiguous_exclude`, with inclusion/exclusion rules and at least two synthetic examples per label. Require a 40-question pilot, independent double labeling, raw agreement and Cohen's kappa, adjudication, taxonomy revision if agreement is weak, and at least 20% double labeling before any later 800-question study.

- [ ] **Step 3: Implement sampling and sheet validation only**

The output schema is `question_id, subject, question_text, annotator_id, intent_label, confidence, notes`. Validation enforces allowed labels, confidence `1–3`, unique `(question_id, annotator_id)`, and non-empty annotator IDs. It reports agreement readiness but does not claim human work occurred.

- [ ] **Step 4: Run tests and generate the pilot template**

Run:

```powershell
.\.venv\Scripts\python -m pytest tests/test_annotation.py -q
.\.venv\Scripts\stem-analytics sample-intent-pilot --config configs/project.yaml
```

Expected: tests PASS; a balanced deterministic 40-row local sheet or redistribution-safe schema/example is produced.

- [ ] **Step 5: Commit future-work artifacts**

```powershell
git add docs/annotation_guide.md src/stem_analytics/annotation.py src/stem_analytics/cli.py tests/test_annotation.py reports/tables
git commit -m "docs: define intent annotation future work"
```

---

### Task 14: Add CI, Clean-Room Verification, and Public-Release QA

**Required skill before completion claims:** Use `superpowers:verification-before-completion`.

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `scripts/verify_clean_room.ps1`
- Create: `reports/qa/release_checklist.md`
- Modify: `README.md`
- Modify: `docs/reproducibility.md`

**Interfaces:**
- CI runs fixture-based tests and Ruff on Python 3.11 and 3.12.
- Clean-room script creates a workspace-local temporary virtual environment, installs the package, runs tests, runs the fixture pipeline, and verifies expected outputs.

- [ ] **Step 1: Add GitHub Actions CI**

Workflow requirements:

```yaml
strategy:
  matrix:
    python-version: ["3.11", "3.12"]
steps:
  - uses: actions/checkout@v4
  - uses: actions/setup-python@v5
    with:
      python-version: ${{ matrix.python-version }}
      cache: pip
  - run: python -m pip install -e ".[dev]"
  - run: python -m ruff check src tests
  - run: python -m pytest -q
```

CI must use local fixtures and must not download MMLU-Pro or execute the full search grid.

- [ ] **Step 2: Implement clean-room verification**

`scripts/verify_clean_room.ps1` creates `.verify-venv`, installs `.[dev]`, runs Ruff, the complete test suite, and fixture `run-all`, then checks the required fixture outputs. It must stop on first failure with `$ErrorActionPreference = 'Stop'`. It does not delete user data or the main `.venv`.

- [ ] **Step 3: Run full local workflow from current pinned data**

Run:

```powershell
.\.venv\Scripts\stem-analytics run-all --config configs/project.yaml
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m ruff check src tests
```

Expected: workflow completes or correctly reuses provenance-matching artifacts; tests and lint pass.

- [ ] **Step 4: Run clean-room verification**

Run: `powershell -ExecutionPolicy Bypass -File scripts/verify_clean_room.ps1`

Expected: clean-room installation, tests, and fixture pipeline all PASS. Record command, timestamp, Python version, test count, duration, and output hashes in `reports/qa/release_checklist.md`.

- [ ] **Step 5: Audit the repository for secrets and oversized files**

Run:

```powershell
git status --short
git ls-files | Select-String -Pattern '\.env$|\.sqlite3?$|\.joblib$|data/raw/|data/processed/'
git ls-files | ForEach-Object { Get-Item $_ } | Where-Object Length -gt 10MB | Select-Object FullName,Length
```

Expected: no credentials, databases, model binaries, dataset rows, or unintended file over 10 MB are tracked. Under `data/raw/` and `data/processed/`, only intentionally negated `.gitkeep` or documentation files may appear.

- [ ] **Step 6: Run numerical, completeness, and PDF gates again**

Require:

- zero unresolved major rows in `numerical_consistency_register.csv`;
- all required report sections marked complete;
- every figure marked verified;
- every PDF page marked pass;
- final-test and CV claims trace to authoritative artifacts;
- README commands match tested commands.

- [ ] **Step 7: Verify GitHub and local presentation paths**

Render README locally to inspect links, ensure every linked figure/report path exists, run `python -m stem_analytics --help`, and rehearse a short terminal demonstration of `validate`, `train` cache status, `evaluate` cache status, and report opening.

- [ ] **Step 8: Commit release readiness**

```powershell
git add .github scripts README.md docs reports/qa
git commit -m "ci: verify reproducible public release"
git status --short
```

Expected: clean working tree.

---

## Final Verification Commands

Before claiming completion, run these commands fresh and preserve their outputs in the release checklist:

```powershell
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m ruff check src tests
.\.venv\Scripts\stem-analytics run-all --config configs/project.yaml
powershell -ExecutionPolicy Bypass -File scripts/verify_clean_room.ps1
git status --short
```

Then verify manually:

- README opens with the project question, result, and limitations.
- All README links and images resolve from GitHub-relative paths.
- `docs/final_report.md` and `docs/final_report.pdf` display the same major numbers.
- The public Git tree excludes raw data, databases, model binaries, and secrets.
- The project is described as an offline feasibility study, not a production or real-student system.
- Intent classification is labeled future work everywhere.

## Suggested Public Repository Metadata

- Repository name: `stem-learning-analytics`
- Description: `Reproducible STEM question classification workflow with SQL analytics, leakage-safe model selection, hyperparameter tuning, error analysis, and an evidence-grounded report.`
- Topics: `machine-learning`, `data-science`, `scikit-learn`, `sql`, `nlp`, `mlops`, `model-evaluation`, `error-analysis`, `reproducibility`

Creating the remote GitHub repository and pushing are external mutations and require explicit user authorization at execution time. Local implementation and verification do not imply permission to publish.
