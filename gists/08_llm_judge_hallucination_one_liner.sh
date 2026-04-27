#!/usr/bin/env bash
# Section 3 hero command from Medium post #3:
# "Some agent regressions don't fit a budget. Score them in BigQuery instead."
#
# Runs the SDK's pre-built hallucination/faithfulness judge against
# the last 24 hours of traces in your agent_events table. The judge
# executes via BigQuery AI.GENERATE — BigQuery submits the model
# call from SQL; no app-side export, no custom judge service. When
# AI.GENERATE isn't available (wrong region, missing model
# permissions), the SDK transparently falls back to direct Gemini
# API calls via google-genai. ``report.details["execution_mode"]``
# names the path that actually ran.
#
# Exit codes:
#   0 — every evaluated session scored at or above the threshold
#   1 — at least one session fell below (your CI gate fires here)
#   2 — configuration error (bad dataset, missing auth, etc.)
#
# Requires bigquery-agent-analytics >= 0.2.3 (the first release
# with the live-verified AI.GENERATE judge path; 0.2.2 shipped a
# broken table-valued template).

bq-agent-sdk evaluate \
  --project-id="$PROJECT_ID" \
  --dataset-id="$DATASET_ID" \
  --evaluator=llm-judge \
  --criterion=hallucination \
  --threshold=0.7 \
  --last=24h \
  --agent-id=calendar_assistant \
  --exit-code

# Tune the threshold once you've seen the score distribution your
# agent actually produces — run the same command without
# ``--exit-code`` over the last 30 days first and look at the
# faithfulness scores.
#
# Add ``--strict`` later if you want a dashboard to distinguish
# empty-score failures from low-score failures. The gate exits
# red the same way either way.
