-- Decision: identify source composition that may create shortcut features or robustness risk.
SELECT subject, source_name, COUNT(*) AS record_count
FROM questions
GROUP BY subject, source_name
ORDER BY subject, record_count DESC, source_name;
