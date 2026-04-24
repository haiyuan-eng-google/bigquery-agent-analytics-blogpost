#!/usr/bin/env bash
# capture_assets.sh — Turnkey helper for the three screenshots
# Medium post #2 needs ("Your Agent Events Table Is Also a Test Suite").
#
# Usage:
#   export GOOGLE_CLOUD_API_KEY=AQ.your-vertex-express-key
#   export GOOGLE_GENAI_USE_VERTEXAI=true
#   export PROJECT_ID=test-project-0728-467323
#   export DATASET_ID=agent_analytics_demo
#   ./capture_assets.sh
#
# The script runs the exact commands needed for the two terminal
# shots (cover setup + section 3) and prints step-by-step for the
# GitHub Actions UI shot and the BigQuery console shot.

set -eu

: "${GOOGLE_CLOUD_API_KEY:?set to your Vertex AI Express API key}"
: "${GOOGLE_GENAI_USE_VERTEXAI:?set to 'true'}"
: "${PROJECT_ID:?set to test-project-0728-467323 (or your sandbox project)}"
: "${DATASET_ID:?set to agent_analytics_demo (or your sandbox dataset)}"

SANDBOX_REPO="caohy1988/bqaa-ci-sandbox"

# Colors for the section banners (dark-terminal-friendly)
B=$'\033[1m'
Y=$'\033[33m'
C=$'\033[36m'
R=$'\033[0m'

banner() { printf '\n%s===== %s =====%s\n' "$B" "$1" "$R"; }

banner "STEP 1 — Seed ~20 regressed sessions into agent_events"
echo "About to run demo_calendar_assistant_regressed.py three times."
echo "Each run emits 5 regressed sessions, so you end up with ~15 in total."
echo "This pushes token usage past the 50k/session budget and makes the"
echo "CI workflow's 'Token budget' step go red."
echo
read -p "Press ENTER to start, or Ctrl-C to skip. " _
for i in 1 2 3; do
  echo
  echo "${C}-- run $i / 3 --${R}"
  python demo_calendar_assistant_regressed.py
done

banner "STEP 2 — Kick off the CI runs in the sandbox repo"
cat <<EOT
Open the sandbox PRs and re-run the workflow so the gate sees the
freshly-seeded regressed fleet:

  Red PR (regressed prompt, should fail Token budget step):
    https://github.com/${SANDBOX_REPO}/pull/1

  Green PR (baseline, should pass all four steps):
    https://github.com/${SANDBOX_REPO}/pull/2

On each PR page:
  1. Click "Checks" tab.
  2. Find the most recent "Agent quality gate" run.
  3. Click "Re-run all jobs" (top right).
  4. Wait ~1–2 minutes for completion.

Press ENTER when both runs have finished (red on PR #1, green on PR #2).
EOT
read _

banner "SHOT #2 — terminal with 'evaluate --exit-code' FAIL lines (section 3 of draft)"
echo "Running the same command the workflow runs, locally, so you can"
echo "screenshot the output in your terminal:"
echo
CMD="bq-agent-sdk evaluate --project-id=$PROJECT_ID --dataset-id=$DATASET_ID --evaluator=token_efficiency --threshold=50000 --last=24h --agent-id=calendar_assistant --exit-code"
echo "${Y}\$ ${CMD}${R}"
echo
# Run it; let the non-zero exit just flow through without stopping the script.
set +e
${CMD}
STATUS=$?
set -e
echo
echo "(Exit code: ${STATUS}. Expected: 1, meaning at least one session blew budget.)"
echo
cat <<EOT
Screenshot framing guidance:
  - Dark terminal theme, 14pt+ monospace, full window width so no wrap.
  - Crop tight: the command line + the summary line +
    the first 3–5 FAIL lines. Leave the '... more failing session(s)'
    footer visible.
  - Save as PNG at retina resolution.
EOT

banner "COVER and SHOT #3 — GitHub Actions failing/passing run (section 4 of draft)"
cat <<EOT
Cover image (feed hero):
  1. Open https://github.com/${SANDBOX_REPO}/pull/1
  2. Click "Checks" tab.
  3. Select the most recent "Agent quality gate" run.
  4. Click the red "Token budget" step to expand it.
  5. Scroll to the block where '--exit-code:' appears followed by
     FAIL lines. Frame the screenshot on:
       - The step header (red X, 'Token budget', timestamps)
       - The '--exit-code: N session(s) failed' summary
       - 3–5 FAIL session=... observed=... budget=... lines
  6. Crop tight. Dark GHA theme if available.

Inline shot #3 (section 4):
  Same run, zoomed out one level. Show the PR status pill with
  three green checks + one red X, and the full list of step names
  on the left panel. This proves 'each gate is its own step.'

Green-counterpart shot (optional for series plan):
  Same process on https://github.com/${SANDBOX_REPO}/pull/2
  All four steps should show green checks.
EOT

banner "SHOT #4 — BigQuery INFORMATION_SCHEMA cost pivot (section 6 of draft)"
cat <<EOT
After the CI runs finish (so eval-code and eval-categorical rows
exist in JOBS_BY_PROJECT), open the BigQuery console in the
sandbox project and run the query below. Screenshot the result
pane.

  BigQuery console:
    https://console.cloud.google.com/bigquery?project=${PROJECT_ID}

  SQL (paste this):
EOT
cat <<'SQL'
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
SQL
cat <<EOT

Screenshot framing:
  - Light browser theme, no extensions visible.
  - Crop tight to the result table: column headers + 3–5 rows.
  - Save as PNG.

EOT

banner "DONE — next step: upload to the Medium draft"
cat <<EOT
Put the four PNGs in docs/assets/ as:
  docs/assets/cover_gha_failure.png
  docs/assets/shot2_evaluate_stderr.png
  docs/assets/shot3_gha_pr_checks.png
  docs/assets/shot4_information_schema.png

Then in the Medium draft, swap each *[SCREENSHOT: ...]* placeholder
for an image upload. The cover slot is set at publication time via
the "Add a cover image" button.
EOT
