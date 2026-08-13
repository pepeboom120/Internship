PRAGMA foreign_keys = ON;

CREATE TABLE questions (
    question_id INTEGER PRIMARY KEY,
    source_question_id INTEGER NOT NULL UNIQUE,
    question_text TEXT NOT NULL CHECK(length(trim(question_text)) > 0),
    subject TEXT NOT NULL CHECK(subject IN ('math','physics','chemistry','biology')),
    source_name TEXT NOT NULL CHECK(length(trim(source_name)) > 0),
    text_length INTEGER NOT NULL CHECK(text_length > 0),
    duplicate_group_id INTEGER NOT NULL,
    duplicate_group_size INTEGER NOT NULL CHECK(duplicate_group_size >= 1),
    duplicate_group_max_similarity REAL NOT NULL CHECK(duplicate_group_max_similarity BETWEEN 0 AND 1),
    split_name TEXT NOT NULL CHECK(split_name IN ('development','test')),
    dataset_revision TEXT NOT NULL
);

CREATE TABLE model_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_name TEXT NOT NULL CHECK(task_name = 'subject'),
    model_name TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    random_seed INTEGER NOT NULL,
    cv_macro_f1 REAL,
    test_macro_f1 REAL,
    test_weighted_f1 REAL,
    test_accuracy REAL,
    runtime_seconds REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    question_id INTEGER NOT NULL,
    true_label TEXT NOT NULL,
    predicted_label TEXT NOT NULL,
    confidence REAL,
    is_correct INTEGER NOT NULL CHECK(is_correct IN (0,1)),
    FOREIGN KEY(run_id) REFERENCES model_runs(run_id),
    FOREIGN KEY(question_id) REFERENCES questions(question_id),
    UNIQUE(run_id, question_id)
);

CREATE INDEX idx_questions_subject ON questions(subject);
CREATE INDEX idx_questions_source ON questions(source_name);
CREATE INDEX idx_questions_split ON questions(split_name);
CREATE INDEX idx_predictions_run ON predictions(run_id);

