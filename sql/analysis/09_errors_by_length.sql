-- Decision: test whether unusually short or long questions need separate handling.
SELECT CASE WHEN q.text_length < 100 THEN '<100'
            WHEN q.text_length < 200 THEN '100-199'
            WHEN q.text_length < 400 THEN '200-399'
            ELSE '400+' END AS length_band,
       COUNT(*) AS n, SUM(1 - p.is_correct) AS error_count,
       ROUND(1.0 * SUM(1 - p.is_correct) / COUNT(*), 4) AS error_rate
FROM predictions p JOIN questions q USING(question_id)
GROUP BY length_band ORDER BY MIN(q.text_length);
