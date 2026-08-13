-- Decision: estimate automatic coverage and manual-review volume at different confidence levels.
SELECT CASE WHEN confidence < 0.5 THEN '[0.0,0.5)'
            WHEN confidence < 0.6 THEN '[0.5,0.6)'
            WHEN confidence < 0.7 THEN '[0.6,0.7)'
            WHEN confidence < 0.8 THEN '[0.7,0.8)'
            WHEN confidence < 0.9 THEN '[0.8,0.9)'
            ELSE '[0.9,1.0]' END AS confidence_band,
       COUNT(*) AS n, ROUND(AVG(is_correct), 4) AS accuracy
FROM predictions WHERE confidence IS NOT NULL
GROUP BY confidence_band ORDER BY MIN(confidence);
