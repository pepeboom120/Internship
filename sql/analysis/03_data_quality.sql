-- Decision: confirm the controlled modeling table has no empty required fields or split leakage.
SELECT COUNT(*) AS total_records,
       SUM(CASE WHEN trim(question_text) = '' THEN 1 ELSE 0 END) AS empty_questions,
       SUM(CASE WHEN trim(source_name) = '' THEN 1 ELSE 0 END) AS missing_sources,
       COUNT(*) - COUNT(DISTINCT source_question_id) AS duplicate_source_ids
FROM questions;
