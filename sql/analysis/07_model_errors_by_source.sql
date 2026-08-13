-- Decision: detect source-specific performance degradation and shortcut dependence.
SELECT q.source_name, COUNT(*) AS n,
       SUM(1 - p.is_correct) AS error_count,
       ROUND(1.0 * SUM(1 - p.is_correct) / COUNT(*), 4) AS error_rate
FROM predictions p JOIN questions q USING(question_id)
GROUP BY q.source_name ORDER BY error_rate DESC, n DESC;
