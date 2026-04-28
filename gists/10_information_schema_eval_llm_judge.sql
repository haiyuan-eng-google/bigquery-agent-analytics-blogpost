-- Section 6 INFORMATION_SCHEMA cost pivot from Medium post #3:
-- "Some agent regressions don't fit a budget. Score them in
-- BigQuery instead."
--
-- The BigQuery Agent Analytics SDK labels every job it issues
-- with ``sdk_feature``. Run this pivot after a day of CI gating
-- and you'll see one row per feature: ``eval-code`` (post #2's
-- deterministic budgets), ``eval-llm-judge`` (this post's
-- semantic gate), ``trace-read`` (developers pulling individual
-- failing sessions with ``client.get_session_trace(...).render()``
-- from post #1).
--
-- One gotcha: the BigQuery side of an AI.GENERATE call shows up
-- here, but Vertex AI inference is billed separately on the AI
-- Platform side. Use this pivot for BQ slot cost; check the AI
-- Platform billing report for the inference cost.
--
-- Swap ``region-us`` for the region your dataset lives in:
-- ``region-us-central1``, ``region-europe-west4``, etc.

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
