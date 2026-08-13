-- Decision: verify class breadth and duplicate-group isolation in development and final test data.
SELECT split_name, subject, COUNT(*) AS record_count,
       COUNT(DISTINCT duplicate_group_id) AS duplicate_groups
FROM questions GROUP BY split_name, subject ORDER BY split_name, subject;
