-- Decision: identify the most consequential subject pairs for targeted error analysis.
SELECT true_label, predicted_label, COUNT(*) AS confusion_count
FROM predictions WHERE is_correct = 0
GROUP BY true_label, predicted_label ORDER BY confusion_count DESC, true_label, predicted_label;
