-- Section 6 INFORMATION_SCHEMA cost pivot from Medium post #2:
-- "Your Agent Events Table Is Also a Test Suite"
--
-- The BigQuery Agent Analytics SDK labels every query it issues
-- with the feature that triggered it. This pivot groups BQ jobs
-- from the last 24 hours by `sdk_feature`, so you can see what the
-- CI gate cost in BQ compute and what the developer trace-reads
-- after a failing run cost on top of that.
--
-- Swap `region-us` for the region your dataset lives in
-- (`region-us-central1`, `region-europe-west4`, etc.).

SELECT
  (SELECT value FROM UNNEST(labels) WHERE key = 'sdk_feature') AS sdk_feature,
  COUNT(*) AS runs,
  ROUND(SUM(total_bytes_processed) / POW(1024, 3), 3) AS gb_processed,
  ROUND(AVG(total_slot_ms), 0) AS avg_slot_ms
FROM `region-us`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
WHERE creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
  AND EXISTS (
    SELECT 1 FROM UNNEST(labels)
    WHERE key = 'sdk' AND value = 'bigquery-agent-analytics'
  )
GROUP BY sdk_feature
ORDER BY runs DESC;
