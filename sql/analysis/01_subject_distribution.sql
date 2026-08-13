-- Decision: quantify class imbalance before model selection and interpret per-class metrics.
SELECT subject,
       COUNT(*) AS record_count,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS percentage
FROM questions
GROUP BY subject
ORDER BY CASE subject WHEN 'math' THEN 1 WHEN 'physics' THEN 2 WHEN 'chemistry' THEN 3 ELSE 4 END;
