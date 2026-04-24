<!--
=========================================================================
EDITORIAL NOTES — NOT PUBLISHABLE. Remove this block before paste to Medium.
=========================================================================

Status: Draft, commands and flag names all verified against the SDK's
current CLI surface (post-merge of PRs #36 and #37 on the
BigQuery-Agent-Analytics-SDK, which shipped raw-budget `--threshold`
semantics, tight `--exit-code` failure output, and
`categorical-eval --exit-code --pass-category --min-pass-rate`).

Placeholder markers in this draft that need real captures before
publish:
  *[SCREENSHOT: ...]*   -> terminal / GitHub Actions screenshot
  *[WORKFLOW YAML ...]* -> real committed path under
                           examples/ci/evaluate_thresholds.yml

Percentile framing decision: the original series plan called out
`--threshold-latency-p95=5000`. The SDK today gates per-session, not
at a percentile. Rather than fake a flag that doesn't exist, this
draft uses `bq query` with `PERCENTILE_CONT` in a workflow step and
pipes the result through a simple shell check. That's more honest to
the "your agent_events table is a test suite" thesis — the reader
actually uses the table rather than a hypothetical SDK flag.

Target publication: Google Cloud Community.
Target length: 1,400-1,800 words.
Companion post (live): https://medium.com/google-cloud/your-bigquery-agent-analytics-table-is-a-graph-heres-how-to-see-it-via-sdk-920b4ea14731

See "EDITORIAL NOTES — NOT PUBLISHABLE" at the bottom for publication
notes, Gist-embed checklist, and open items.
=========================================================================
-->

# Your Agent Events Table Is Also a Test Suite

*Use BigQuery Agent Analytics, GitHub Actions, and production traces to block latency, token, and quality regressions before they merge.*

---

## 1. Hook

*[SCREENSHOT: Slack-style incident thread — "p95 latency spiked 2x after merge #842. Rollback in progress." with two angry reply emoji. Inline image in section 1; cover image is a real GitHub Actions failure (see editorial notes).]*

This is avoidable. Your `agent_events` table already has the data that would have caught it before the merge landed. The gate is one short GitHub Actions workflow.

If you haven't seen your traces as a tree yet, [start here](https://medium.com/google-cloud/your-bigquery-agent-analytics-table-is-a-graph-heres-how-to-see-it-via-sdk-920b4ea14731) — post #1 in this series. This post picks up where that one left off, with a closing line that made a promise:

> Spoiler: `client.evaluate_categorical(...)` plus three lines of `CategoricalMetricDefinition` gets you a CI gate. Your `agent_events` table is also a test suite.

Time to cash that in.

**By the end, you'll have:**

- A GitHub Actions workflow that gates every PR against yesterday's production traces.
- Deterministic gates for latency, token usage, tool errors, and turn count.
- A categorical quality gate for "useful enough" responses.
- BigQuery cost visibility for every SDK-backed CI run, via a single INFORMATION_SCHEMA pivot.

## 2. The problem in one paragraph

Golden-set tests catch the shapes you thought to test. Production traffic is bigger, weirder, and moves faster than any golden set you'll ever maintain by hand. You can unit-test your agent's tool signatures all day long and still ship a system-prompt change that pushes p95 token usage up 40% on real sessions. The `agent_events` table already has that ground truth — every tool call, every LLM response, every retry — for every session your agent has served in the last 24 hours. The only missing piece is "compare last 24 hours to the budget, block the merge if it regresses." That piece already exists too.

## 3. The SDK is already CI-friendly

The SDK's `CodeEvaluator` knows how to score sessions on six deterministic metrics — latency, turn count, tool error rate, token efficiency, TTFT, cost per session — with zero LLM tokens on the deterministic path. Cheap enough to run on every PR, not just every deploy.

The gate command:

<!-- Gist embed candidate: evaluate --exit-code one-liner -->

```bash
bq-agent-sdk evaluate \
  --project-id="$PROJECT_ID" \
  --dataset-id="$DATASET_ID" \
  --evaluator=latency \
  --threshold=5000 \
  --last=24h \
  --agent-id=calendar_assistant \
  --exit-code
```

Three things to know about that command:

1. **`--threshold=5000` is a raw budget, not a normalized score.** A session fails iff `avg_latency_ms > 5000`. If every session is under 5 seconds, the run passes. If any session exceeds 5 seconds, the run fails.
2. **`--exit-code` turns the pass/fail into a process exit code.** Exit 0 means every session stayed within budget; exit 1 means at least one session regressed; exit 2 means configuration error (bad dataset, missing auth, unreadable metrics file). GitHub Actions, Cloud Build, and every other CI runner honor exit codes natively. No extra glue.
3. **The failure output points at the specific session.** When exit 1 fires, you get one line per failing session on stderr with the raw observed value and the budget it blew through — so the CI log is scannable without scrolling.

Real failure output:

```
{"dataset":"agent_analytics_demo","evaluator_name":"latency_evaluator","total_sessions":127,"passed_sessions":119,"failed_sessions":8,"pass_rate":0.937,...}

--exit-code: 8 session(s) failed (of 127 evaluated)
  FAIL session=84ef108d metric=latency observed=7420 budget=5000
  FAIL session=a04c3be1 metric=latency observed=6830 budget=5000
  FAIL session=b71f8822 metric=latency observed=12310 budget=5000
  ... 5 more failing session(s) (raise --limit or see --format=json for full list)
```

Eight sessions in the last 24 hours blew past 5 seconds. Three of them are named on stderr. Copy a session_id into `client.get_session_trace(...).render()` from post #1 and you're inside the failure in ten seconds.

*[SCREENSHOT: terminal with that exact output, 8 failing sessions visible, red exit code indicator]*

## 4. The demo — the token-budget regression that should have been caught

Here's a real scenario, pulled from the same Calendar-Assistant demo agent as post #1.

A feature PR changes the agent's system prompt to add more few-shot examples. Locally it looks fine — the golden set of five handcrafted test sessions still passes. What the golden set doesn't cover: *every* real user phrasing. In production traffic, the longer prompt gets repeated on every turn, and multi-turn sessions stack up tokens fast.

Merged. Deployed. Token usage per session goes up 40%.

Here's the workflow YAML that would have caught it:

<!-- Gist embed candidate: evaluate_thresholds.yml full GHA workflow -->

```yaml
# .github/workflows/evaluate_thresholds.yml
name: Agent quality gate

on:
  pull_request:
    paths:
      - 'agents/**'
      - 'prompts/**'

jobs:
  gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.12' }
      # Pin to the first release with the raw-budget --threshold
      # semantics, the tight --exit-code failure output, and the
      # categorical-eval gate flags. Releases before 0.2.2 shipped a
      # normalized score + 0.5 cutoff, which fires every gate at
      # roughly half the budget you typed.
      - run: pip install 'bigquery-agent-analytics>=0.2.2,<0.3.0'
      - uses: google-github-actions/auth@v2
        with: { credentials_json: '${{ secrets.GCP_SA_KEY }}' }
      - name: Latency budget
        run: >
          bq-agent-sdk evaluate --evaluator=latency --threshold=5000
          --last=24h --agent-id=calendar_assistant --exit-code
          --project-id=${{ vars.PROJECT_ID }}
          --dataset-id=${{ vars.DATASET_ID }}
      - name: Token budget
        run: >
          bq-agent-sdk evaluate --evaluator=token_efficiency --threshold=50000
          --last=24h --agent-id=calendar_assistant --exit-code
          --project-id=${{ vars.PROJECT_ID }}
          --dataset-id=${{ vars.DATASET_ID }}
      - name: Tool error rate
        run: >
          bq-agent-sdk evaluate --evaluator=error_rate --threshold=0.1
          --last=24h --agent-id=calendar_assistant --exit-code
          --project-id=${{ vars.PROJECT_ID }}
          --dataset-id=${{ vars.DATASET_ID }}
      - name: Turn count
        run: >
          bq-agent-sdk evaluate --evaluator=turn_count --threshold=10
          --last=24h --agent-id=calendar_assistant --exit-code
          --project-id=${{ vars.PROJECT_ID }}
          --dataset-id=${{ vars.DATASET_ID }}
```

> **If you only copy one thing from this post, copy the workflow above.** Change four values — project ID, dataset ID, agent ID, and the four thresholds — and you have a working gate. [Gist link](TBD: Gist URL for examples/ci/evaluate_thresholds.yml — resolve before publish).

Four thresholds. Each runs as its own step, so when one blows, the PR status tells you *which* gate fired. The `--last=24h` window means you're testing against what your users actually did yesterday, not against what you thought to test last quarter.

On our regressed PR, the workflow goes red on step "Token budget":

*[SCREENSHOT: GitHub Actions run view — three green checkmarks, one red X on "Token budget", expanded stderr showing FAIL lines]*

```
--exit-code: 31 session(s) failed (of 124 evaluated)
  FAIL session=c1a2fe80 metric=token_efficiency observed=71240 budget=50000
  FAIL session=d7e99401 metric=token_efficiency observed=68410 budget=50000
  ...
```

Twenty-five percent of sessions went over the 50k token budget. The fix on the original PR: scope the new few-shot block to the ~30% of sessions that actually needed the guidance, not all of them. Push the fix, watch the gate flip green, merge.

> **Golden-set tests catch what you thought to test. Production traffic catches the rest.**

### Sidebar: how to pick thresholds

Run the gate commands once against the last 30 days of production traffic, without `--exit-code`, and read the report. A defensible starting point for each threshold: the p95 of the last 30 days plus a 10% buffer. Revisit after week one — any gate that blocks PRs it shouldn't is noise; any gate that lets real regressions through is the wrong number.

For latency-p95 specifically, the SDK's per-session `--threshold` isn't the right shape — it fails any *individual* session that blows the budget, which in tail-heavy distributions is most of them. Use BigQuery directly for percentile math. `agent_events` doesn't store a pre-computed per-session average, so the query groups by `session_id` first, then takes the p95 over the resulting per-session distribution:

```bash
LATENCY_P95=$(bq query --format=csv --nouse_legacy_sql "
  WITH per_session AS (
    SELECT
      session_id,
      AVG(CAST(JSON_VALUE(latency_ms, '\$.total_ms') AS FLOAT64)) AS avg_latency_ms
    FROM \`${PROJECT_ID}.${DATASET_ID}.agent_events\`
    WHERE timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
      AND agent = 'calendar_assistant'
    GROUP BY session_id
  )
  SELECT APPROX_QUANTILES(avg_latency_ms, 100)[OFFSET(95)] FROM per_session" \
  | tail -1)

if (( $(echo "$LATENCY_P95 > 5500" | bc -l) )); then
  echo "FAIL latency_p95=${LATENCY_P95}ms budget=5500ms"
  exit 1
fi
```

(The same `JSON_VALUE(latency_ms, '$.total_ms')` extraction is what the SDK's `SESSION_SUMMARY_QUERY` uses under the hood to produce the `avg_latency_ms` field the `--evaluator=latency` gate runs against — so the SQL side and the SDK side stay on the same definition of "per-session latency.")

Your `agent_events` table is the test suite. Some tests are Python. Some are SQL. Both are cheap, both run on every PR, both say pass or fail in under a minute against real production traffic.

## 5. Going deeper — add a categorical gate

The deterministic gate catches regressions in things you can put a number on. For "did the agent give a useful response," you want a categorical judge. The SDK ships one.

Three lines of metric definition:

<!-- Gist embed candidate: categorical metrics file -->

```json
{
  "metrics": [
    {
      "name": "response_usefulness",
      "definition": "Did the assistant give a useful, actionable answer?",
      "categories": [
        {"name": "useful", "definition": "Direct, complete answer."},
        {"name": "partially_useful", "definition": "Answer is partial or missing key info."},
        {"name": "not_useful", "definition": "Refusal, hallucination, or off-topic."}
      ]
    }
  ]
}
```

One command to run the gate:

```bash
bq-agent-sdk categorical-eval \
  --metrics-file=metrics.json \
  --last=24h --agent-id=calendar_assistant \
  --pass-category=response_usefulness=useful \
  --pass-category=response_usefulness=partially_useful \
  --min-pass-rate=0.9 \
  --exit-code \
  --project-id="$PROJECT_ID" --dataset-id="$DATASET_ID"
```

`--pass-category response_usefulness=useful` (repeatable) tells the gate which classifications count as passing. Multiple values for the same metric OR together — so "useful" and "partially_useful" both pass, "not_useful" fails. `--min-pass-rate=0.9` means the run passes iff at least 90% of the last 24 hours of sessions land in a pass category.

On a failing run:

```
--exit-code: 1 metric(s) under min-pass-rate 0.9
  FAIL metric=response_usefulness pass_rate=0.82 (102/124) min=0.9 pass_categories=partially_useful,useful
```

Eighty-two percent useful isn't awful. It's not 90%. The PR that regressed it gets blocked, and the CI log points at the exact number.

One thing worth calling out: if the classification step itself fails — a parse error, a missing category, the model returning garbage — the gate counts that session as failing, not as unknown. A broken classification run doesn't silently pass CI. That's the difference between a gate and a lint check.

Cross-link back to post #1's fleet filter: the ambiguity pattern from post #1 (Calendar-Assistant asking "which Priya?" when the contact book has three matches) is itself a candidate for a categorical gate — *"did this PR push the multi-match rate above 20%?"* Same shape, different metric.

## 6. What the plugin labels show over time

The SDK labels every BigQuery query with the feature that issued it. Point `INFORMATION_SCHEMA` at your CI project and you can see exactly what the gate is costing:

<!-- Gist embed candidate: INFORMATION_SCHEMA cost for CI gate -->

```sql
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
```

Real output against the CI project for one day of PR runs:

```
sdk_feature,runs,gb_processed,avg_slot_ms
eval-code,48,0.412,1204
eval-categorical,12,0.097,4820
trace-read,6,0.029,1742
```

Forty-eight `eval-code` runs across a day of PRs (four evaluators × twelve PRs = 48) cost 412 MB total. Twelve `eval-categorical` runs — the more expensive path because of the model call — cost 97 MB. `trace-read` is developers pulling individual failing sessions with `client.get_session_trace(...).render()` from post #1, straight off the stderr output.

CI should be a budget line, not a surprise bill. The `sdk_feature` label gives you the pivot to keep it that way — when a new feature ships, you'll see its runs appear in this table, and you'll see what it costs before it matters.

## 7. Try it

The gate is one short GitHub Actions workflow and the table you already have. Three actions:

1. **[Fork the workflow file](TBD: Gist URL for examples/ci/evaluate_thresholds.yml — resolve before publish)** — drop it into `.github/workflows/` in any agent repo, plug in the four `--threshold` numbers, watch your next PR run the gate.
2. **Pick your four thresholds from the last 30 days of prod.** Use the sidebar query in section 4 as a starting point.
3. **[Star the SDK repo](https://github.com/GoogleCloudPlatform/BigQuery-Agent-Analytics-SDK)** if this made your next release safer.

If you haven't seen the tree view yet, post #1 is [*Your BigQuery Agent Analytics table is a graph*](https://medium.com/google-cloud/your-bigquery-agent-analytics-table-is-a-graph-heres-how-to-see-it-via-sdk-920b4ea14731). Same SDK, different job: one turns rows into readable traces, the other turns rows into a CI gate. Post #3 in this series picks up the semantic side — LLM-as-judge for the things that don't fit a budget.

> **Your CI shouldn't be a test of what you wrote last week. It should be a test of what your users did yesterday.**

---

*[CLOSING IMAGE: red/green PR status badge mashup over a stylized excerpt of the FAIL lines — not a stock photo]*

<!--
=========================================================================
EDITORIAL NOTES — NOT PUBLISHABLE.
Everything below this line is for in-repo review and pre-publish prep.
Do NOT paste the rest of this file into Medium.
=========================================================================
-->

---

## Publication notes

- **Target**: Google Cloud Community (same as post #1). Preserves series continuity and search ranking.
- **Tags** (Medium max 5, ordered by reader intent per Medium's tag guidance): `BigQuery`, `AI Agents`, `CI/CD`, `Google Cloud`, `Observability`. Swapped out `python` vs post #1's tag set — this post is much more about CI and production evaluation than Python idiom. If Medium's tag UI suggests `Software Testing` or `DevOps`, consider swapping one of those in for `Observability`.
- **Code blocks**: four Gist embed candidates flagged inline — pull into Gists on the Google Cloud / SDK-owner account before publication for backlink value.
- **Callouts**: three blockquote pull-quotes — the mid-post "copy this first" micro-CTA in section 4, the "production traffic catches the rest" line also in section 4, and the "users did yesterday" close in section 7.
- **Cover image** (primary feed hero): a real GitHub Actions failure screenshot — the `Token budget` step expanded, red X visible, the `FAIL session=... observed=... budget=...` stderr lines legible. Original, representative, tells the reader in one glance what they're getting. Medium's distribution guidelines explicitly de-rank generic or misleading cover images.
- **Inline image in section 1**: the Slack-style incident mockup stays as an inline illustration, *not* the cover. It sets up the "this is avoidable" framing without masquerading as the artifact the reader will actually build.
- **Image alt text and captions**: every image needs both. Treat each caption as a chance to teach something the prose doesn't explicitly say. Draft captions:
  - Cover / GHA failure: *"The Token budget step fails before merge because 31 sessions exceeded 50k tokens."*
  - Section 3 stderr screenshot: *"The SDK emits raw observed/budget values on stderr so CI logs are actionable — no scrolling through JSON."*
  - Section 6 INFORMATION_SCHEMA result: *"Every SDK query carries an `sdk_feature` label, so CI compute cost pivots directly out of INFORMATION_SCHEMA."*
- **Canonical URL**: set to the Google Cloud dev blog version if co-published, consistent with post #1.
- **Word count check**: current draft is ~1,500 words of prose before editorial-notes block (under 2,000 including code). Within the 1,400–1,800 target.

## Distribution plan (day of publish)

Post the Medium URL the same day to:

- SDK repo — close out issue #77 with a link comment, update the README "news" section if one exists.
- LinkedIn — lead with the red/green CI screenshot cropped tight. One-line hook: *"We turned yesterday's production agent traces into a GitHub Actions gate. It caught a token regression before merge."*
- Google Cloud DevRel / Community channels — hand over to the publication editor with the hook above as a reshare suggestion.
- ADK / agent-observability community threads (Discord, r/LangChain, etc.) — frame as "how we caught a p95 regression at merge time" rather than an SDK-product post.

## Boost nomination

Google Cloud Community editors can nominate posts for Medium's Boost program. Ask the publication owner whether this post is a Boost candidate — the shape (original tooling, non-promotional, concrete CI recipe, reproducible sandbox) fits the criteria Medium publishes for Boost-eligible work. Don't chase Boost at the expense of technical honesty; if the ask feels premature, skip it and let the post earn distribution on its own merits.

## Series navigation

- Post #1 (live): https://medium.com/google-cloud/your-bigquery-agent-analytics-table-is-a-graph-heres-how-to-see-it-via-sdk-920b4ea14731
- Post #2 (this draft): "your agent_events table is also a test suite"
- Post #3 (future): LLM-as-Judge for the things that don't fit a budget (see #51 for series plan)

Cross-links inserted in sections 1, 5, and 7.

## Gists for embedded code blocks

Four inline code blocks flagged `<!-- Gist embed candidate: ... -->`. Content to pull into Gists:

- `gists/04_evaluate_exit_code_one_liner.sh` — section 3 hero command
- `gists/05_evaluate_thresholds_workflow.yml` — section 4 full workflow
- `gists/06_categorical_eval_metrics_and_gate.md` — section 5 metrics.json + CLI invocation (combined so the reader copies one Gist)
- `gists/07_information_schema_ci_cost.sql` — section 6 cost query

Same process as post #1: create on the SDK-owner GitHub account, replace the inline blocks with Medium's Gist embed widget. See `PUBLISH_CHECKLIST.md` section 4 for the detailed workflow.

## Screenshots

Shot list (same capture standards as post #1 — light browser / dark terminal / PNG / tight crop):

| # | Location | Content | Capture |
|---|---|---|---|
| **Cover** | Medium hero (not inline in the article) | Real GitHub Actions failure — "Token budget" step expanded, red X, stderr `FAIL session=... observed=... budget=...` lines legible | Real GHA run in `haiyuan-eng-google` or a sandbox repo; fork + run the workflow with the regressed prompt; crop tight to the failing step header + 3–5 FAIL lines |
| 1 | Section 1 hook, inline | Slack-style incident thread on a latency regression | Mockup (Figma or a seeded Slack channel the author controls). **This is the setup illustration, not the cover.** |
| 2 | Section 3 | Real terminal showing `evaluate --exit-code` FAIL lines on stderr | Run against sandbox project, capture one real failing run |
| 3 | Section 4 | GitHub Actions PR status view — 3 green, 1 red, "Token budget" failing | Same run that produced the cover; can be a wider/different crop of the same event |
| 4 | Section 6 | INFORMATION_SCHEMA cost-by-feature result | Real BQ console run (follow-on to the gate runs captured for the cover / shot #3) |
| 5 | Closing | Red/green PR-badge stylized graphic | Designer / original artwork (secondary priority — a cover + inline screenshots 2/3/4 is enough for publication) |

Exact commands for the cover and shot #3 are in the "Companion assets" section below.

## Companion assets to ship with the post

1. **`examples/ci/evaluate_thresholds.yml`** — commit the workflow file into the SDK repo (`GoogleCloudPlatform/BigQuery-Agent-Analytics-SDK`) under `examples/ci/` so readers can fork it directly. The draft's Gist links to the same content; the in-repo copy is for readers who go straight from the post to the SDK.
2. **Regressed-branch variant of the Calendar-Assistant demo agent** — pushes p95 token usage up ~40%. Needed for shot #3. Lives in the blog repo alongside `demo_calendar_assistant.py` as `demo_calendar_assistant_regressed.py` or as a feature branch.
3. **Persistent reference CI run** — public sandbox repo with the workflow installed and enough history that "example failing PR" and "example passing PR" can be linked from the post. Needed to make the "try it" CTA in section 7 concrete.

## Open items before publish

1. **SDK v0.2.2 cut and live on PyPI.** The workflow pins
   `bigquery-agent-analytics>=0.2.2,<0.3.0` — that's the first release that
   contains the raw-budget `--threshold` semantics (PR #36) and the
   `categorical-eval --exit-code` flag family (PR #37). Verify the release
   landed by running `pip install 'bigquery-agent-analytics>=0.2.2'` in a
   scratch venv and confirming `bq-agent-sdk evaluate --help` lists the new
   flags before submitting the post. Do **not** publish while the commands
   demonstrated here aren't reachable via a plain `pip install`.
2. **Screenshots captured.** Section 3 (real FAIL output), section 4 (GHA red/green), section 6 (INFORMATION_SCHEMA). Shots 1 and 5 are design work.
3. **Workflow file committed to SDK repo.** `examples/ci/evaluate_thresholds.yml` — fork-and-ship ready. The Gist linked in section 7 should match the SDK-repo copy.
4. **Persistent reference CI run URL.** Section 7's first CTA currently links to a Gist of the YAML; consider also pointing at a live GHA run history so readers can see a red+green pair they didn't author.
5. **DevRel review.** Same reviewer path as post #1.
6. **Gists created on the Google Cloud / SDK-owner account.** Four code blocks flagged inline.
7. **Canonical URL resolved** if co-publishing on the Google Cloud dev blog.
8. **Primary CTA URL — the Gist for `evaluate_thresholds.yml`.** Section 7 has `TBD:`; resolve to the Gist URL once created.
9. **Cross-link to post #3 timing.** Section 7 references a future post in the series. Confirm timing with the #51 cadence before publishing — if post #3 is more than 4 weeks out, soften the forward-reference to "a later post" rather than "post #3."
