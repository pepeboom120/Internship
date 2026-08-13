-- Decision: determine which subject requires more data or manual review.
SELECT p.true_label AS subject, COUNT(*) AS n,
       SUM(1 - p.is_correct) AS error_count,
       ROUND(1.0 * SUM(1 - p.is_correct) / COUNT(*), 4) AS error_rate
FROM predictions p GROUP BY p.true_label ORDER BY error_rate DESC;
