# Data Dictionary

The controlled `questions` table contains `source_question_id`, normalized `question_text`, four-class `subject`, `source_name`, `text_length`, duplicate-group diagnostics, split name, and pinned dataset revision. Model features use `question_text` only. Answers, options, explanations, and answer indices remain provenance fields outside the controlled model input.

The `model_runs` table stores the external run ID, model and parameter metadata, seed, development CV macro-F1, final metrics, runtime, and timestamp. `predictions` links one locked run to each final-test question and stores true/predicted labels, optional calibrated confidence, and correctness.
