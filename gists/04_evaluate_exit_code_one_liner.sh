#!/usr/bin/env bash
# Section 3 hero command from Medium post #2:
# "Your Agent Events Table Is Also a Test Suite"
#
# Runs the deterministic latency gate against the last 24 hours of
# production traces. Exit 0 = all sessions within budget; exit 1 =
# at least one session regressed; exit 2 = configuration error.
#
# The SDK's `evaluate --exit-code` path also prints one readable
# FAIL session=... observed=... budget=... line on stderr per failing
# (session, metric) pair so CI logs are scannable without scrolling.
#
# Requires bigquery-agent-analytics >= 0.2.2 (raw-budget --threshold
# semantics + tight --exit-code failure output).

bq-agent-sdk evaluate \
  --project-id="$PROJECT_ID" \
  --dataset-id="$DATASET_ID" \
  --evaluator=latency \
  --threshold=5000 \
  --last=24h \
  --agent-id=calendar_assistant \
  --exit-code
