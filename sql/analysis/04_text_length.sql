-- Decision: compare text complexity across subjects and define diagnostic length bands.
SELECT subject, COUNT(*) AS n,
       ROUND(AVG(text_length), 2) AS mean_length,
       MIN(text_length) AS min_length,
       MAX(text_length) AS max_length
FROM questions GROUP BY subject ORDER BY subject;
