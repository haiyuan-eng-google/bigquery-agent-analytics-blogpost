<!--
=========================================================================
EDITORIAL NOTES — NOT PUBLISHABLE. Remove this block before paste to Medium.
=========================================================================

Status: Draft, anchored on the live AI.GENERATE judge run captured on
2026-04-27 against test-project-0728-467323 / agent_analytics_demo.
Every FAIL line, score, and judge justification quoted in this draft
came from that run (workflow run id 25021548083 on caohy1988/bqaa-ci-sandbox#3).

Companion posts:
  Post #1: https://medium.com/google-cloud/your-bigquery-agent-analytics-table-is-a-graph-heres-how-to-see-it-via-sdk-920b4ea14731
  Post #2: in DevRel review (PR #17 in this repo)

This is post #3 in the series tracked at
https://github.com/GoogleCloudPlatform/BigQuery-Agent-Analytics-SDK/issues/82.
The slot was originally analyst views; maintainer confirmed the swap to
LLM-as-Judge so the closing teaser of post #2 cashes in cleanly.

Anchor judge: hallucination/faithfulness, not correctness. The shift was
driven by the live-data finding that the regressed Calendar-Assistant
traces produce confident-but-fabricated outputs that pass correctness
(scores 0.9–1.0) and fail faithfulness (0.1–0.3). See issue #82 comment
chain for the rationale.

Required SDK pin: bigquery-agent-analytics >= 0.2.3 (live-verified
AI.GENERATE judge path; 0.2.2 had a broken table-valued template).

Target publication: Google Cloud Community.
Target length: 1,400–1,800 words of prose.

See "EDITORIAL NOTES — NOT PUBLISHABLE" at the bottom for the publish
checklist (screenshots, Gists, DevRel review, tags).
=========================================================================
-->

# Some agent regressions don't fit a budget. Score them in BigQuery instead.

*Use BigQuery Agent Analytics, AI.GENERATE, and yesterday's production traces to gate CI on hallucination, correctness, and sentiment — semantic checks the deterministic budgets from post #2 can't see.*

---

## 1. Hook

Imagine your agent's CI is fully green. Latency budget: passed. Token budget: passed. Tool errors: zero. Turn count: under cap. You merge. The next morning a user reports the agent confidently told them their meeting was booked with *"Jordan Lee (Design)"* — except Jordan Lee is in Platform, and the meeting never made it onto the calendar.

The deterministic gate from [post #2 in this series](https://github.com/haiyuan-eng-google/bigquery-agent-analytics-blogpost/pull/17) caught nothing. It couldn't. *"The agent invented a detail"* is not a budget you can put a number on.

That's the gap. This post fills it with one workflow step.

**By the end, you'll have:**

- A semantic CI gate that runs `AI.GENERATE` over yesterday's production traces and fails the PR when the agent is making things up.
- A FAIL line on stderr that names the session and quotes the model's own justification — not a vibe score.
- The same `--exit-code` shape post #2 already trained your CI on; one extra workflow step.
- Live-verified against current BigQuery — no broken table-valued SQL, no missing `connection_id`.

## 2. The problem in one paragraph

Post #2's deterministic gates catch things you can put a number on. Some regressions hide *under* those numbers. Confident wrong answers. Refusals. Off-tone responses. Schema-shaped hallucinations — the agent asserts a team or a date that wasn't in any tool output, and the surrounding response is plausible enough that latency is fine, tokens are fine, no tool errors fire. You can't scale "did a human read it" to fleet traffic. You also can't trust a vibe score from a hand-rolled judge prompt. The SDK ships three pre-built judges (correctness, hallucination, sentiment) backed by frozen prompt templates and a typed `score INT64, justification STRING` output schema, runnable over thousands of sessions in a single `AI.GENERATE` SQL job — or via a transparent Gemini API fallback when AI.GENERATE isn't available in your environment.

## 3. The SDK is already AI-in-warehouse

One CLI invocation, same exit-code shape post #2 trained your CI on:

<!-- Gist embed candidate: llm-judge --exit-code one-liner -->

```bash
bq-agent-sdk evaluate \
  --project-id="$PROJECT_ID" \
  --dataset-id="$DATASET_ID" \
  --evaluator=llm-judge \
  --criterion=hallucination \
  --threshold=0.7 \
  --last=24h \
  --agent-id=calendar_assistant \
  --exit-code
```

Three things to know:

1. **`AI.GENERATE` keeps evaluation in BigQuery.** The judge runs as a single SQL job — your trace data doesn't leave the warehouse. When AI.GENERATE isn't available (no project access, no model access), the SDK falls back to direct Gemini API calls via `google-genai`. `report.details["execution_mode"]` says which path actually ran (`ai_generate`, `ml_generate_text`, or `api_fallback`) so your CI logs are auditable.
2. **`--exit-code` is the same gate post #2 used.** Exit 0 means every session passed the threshold; exit 1 means at least one failed; exit 2 means configuration error. Your existing GitHub Actions / Cloud Build wiring honors it.
3. **Failure output explains itself.** When the gate fails, each FAIL line carries the session id, the metric, the score, and a bounded `feedback="..."` snippet drawn from the judge's justification. The screenshot the post leads with isn't a vibe score; it's the model telling you exactly what it caught.

## 4. The demo — semantic gate catches what budgets can't

Real run, captured today against the same Calendar-Assistant demo from posts #1 and #2. The agent's prompt was tweaked to be more "decisive" — confirm bookings, propose times, fill in plausible team names. Locally it looks great. The deterministic gate's response: green across the board.

Here's what `AI.GENERATE` returns when you ask it whether the agent's claims are supported by the trace:

*[SCREENSHOT: GitHub Actions run view — Hallucination judge step expanded, red X visible, the FAIL lines below visible on stderr.]*

```
--exit-code: 10 session(s) failed (of 10 evaluated)
  FAIL session=1e4e8bf0 metric=faithfulness score=0.1
        feedback="The agent claims to have 'found Jordan Lee (Design)'
                  and mentions a 'default 30-minute duration', but
                  no tool results o…"
  FAIL session=a6239c27 metric=faithfulness score=0.3
        feedback="The agent hallucinates a specific start date for
                  'next week' (May 18) which is not present in the
                  conversation context…"
  FAIL session=33c1e74d metric=faithfulness score=0.3
        feedback="The agent specifies 'Jordan Lee (Design)' and
                  'April 28' without any tool output or context in
                  the trace to support the…"
  ... (7 more failing session(s))
```

Three things worth pausing on:

- **The judge cites the trace.** "*Jordan Lee (Design)*" — the agent confidently asserts a team. The tool output never said that. The judge spots it.
- **Scores cluster low.** 0.1 and 0.3 mean the model is confident about the failure, not split. Tune your threshold above what the model considers a passing answer; here 0.7 gives clean separation.
- **The same fleet passed correctness (3/3 at scores 0.9–1.0).** The agent's *answers* are fine. It's the *grounding* that's broken. That's the regression you can't put a budget on, and that's why faithfulness is the sharper judge for this trace data.

> **If you only copy one thing from this post, copy that one workflow step.** Add it to the gate from post #2 with `--evaluator=llm-judge --criterion=hallucination --threshold=0.7 --exit-code` and your next merge against unsupported claims goes red. [Gist](TBD: Gist URL for llm_judge_hallucination.sh — resolve before publish).

The PR's deterministic gates? They're greyed out in the run view. The Hallucination step ran first, returned exit 1, and GHA stopped before Token budget / Latency / Tool error / Turn count had a chance. That's deliberate — once the semantic gate fires, the rest of the run is moot until someone fixes the prompt. (You can reorder steps if you want different priorities.)

> **Latency you can measure. Hallucination you have to score.**

## 5. Going deeper — stack judges, don't pick one

The same workflow step extends to two more judges with frozen prompts:

<!-- Gist embed candidate: three-judge workflow excerpt -->

```yaml
- name: Hallucination judge
  run: bq-agent-sdk evaluate --evaluator=llm-judge
       --criterion=hallucination --threshold=0.7
       --last=24h --agent-id=calendar_assistant --exit-code
       --project-id=${{ vars.PROJECT_ID }}
       --dataset-id=${{ vars.DATASET_ID }}

- name: Correctness judge
  run: bq-agent-sdk evaluate --evaluator=llm-judge
       --criterion=correctness --threshold=0.7 ...

- name: Sentiment judge
  run: bq-agent-sdk evaluate --evaluator=llm-judge
       --criterion=sentiment --threshold=0.6 ...
```

Each gets its own threshold; each shows up as its own row in the CI log. On the same fleet that fails hallucination 10/10, **correctness passes 3/3** at 0.9–1.0 (the agent does what it's asked given what tools returned), and **sentiment passes 3/3** at 0.8–0.9 (the tone is helpful). That's the value of stacking: each judge answers a different question, and the right combination depends on what your agent does. A booking agent cares about faithfulness more than tone. A support bot might be the inverse.

A short compare/contrast vs. post #2's `categorical-eval`: `LLMAsJudge` produces *continuous* scores (0.0–1.0), good for thresholded gates and tracking distributions over time. `categorical-eval` produces *discrete* one-of-N classifications, good for pass-rate gates over named categories. Use both in the same workflow when both shapes apply.

One small aside on the FAIL output above: a few of the 10 failing sessions appeared as `FAIL session=… (no per-metric detail available)` instead of carrying a `feedback="…"` snippet. That's the safety-net branch firing when AI.GENERATE returned a session with an empty `score` (no parseable judge output for that row). Those sessions still count as failed — empty `scores` falls below any non-zero threshold. If you want to tell them apart from low-score failures in a dashboard or post-incident triage, add `--strict`, which stamps `details["parse_error"]=True` and emits a report-level `parse_errors` counter. For pass/fail-only CI gates, `--strict` is a no-op; reach for it when the question is *which kind* of failure, not whether it failed.

## 6. What the plugin labels show over time

The SDK labels every BigQuery query it issues. Three `sdk_feature` rows show up in INFORMATION_SCHEMA after a day of judge runs:

<!-- Gist embed candidate: INFORMATION_SCHEMA pivot for eval-llm-judge -->

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

Three things you'll see when you run this against your own CI:

- `eval-code` — the deterministic gates from post #2. Cheap; small bytes, sub-second slot ms.
- `eval-llm-judge` — the new semantic gate. Larger slot ms (the model call dominates), still small bytes.
- `trace-read` — developers pulling individual failing sessions with `client.get_session_trace(...).render()` from post #1.

One gotcha: AI.GENERATE jobs *do* show up in INFORMATION_SCHEMA, but Vertex AI inference is billed separately on the AI Platform side. The pivot tells you what the BigQuery side costs; check the AI Platform billing report for the inference side.

## 7. Try it

Two-step CTA:

1. **Add one step to your post-#2 workflow** with `--evaluator=llm-judge --criterion=hallucination --threshold=0.7 --strict --exit-code`. The Gist includes the full step. Run it against your last 24 hours of traces; tune the threshold once you've seen the score distribution your agent actually produces.
2. **If you don't have AI.GENERATE access** in your project — wrong region, missing model permissions — install `bigquery-agent-analytics[improvement]` and the SDK uses the Gemini API fallback path automatically. Same scores, different mechanics.

> **Your CI shouldn't only check what you wrote. It should check whether the agent stays grounded in what your tools said.**

If you haven't seen the deterministic gate yet, post #2 is [*Your Agent Events Table Is Also a Test Suite*](https://github.com/haiyuan-eng-google/bigquery-agent-analytics-blogpost/pull/17). Same SDK, complementary gate: post #2 turns rows into a budget gate; this post turns the same rows into a semantic gate.

---

*[CLOSING IMAGE: stylized graphic of the GHA run with the Hallucination step red and a callout pulling the judge's justification — not a stock photo.]*

<!--
=========================================================================
EDITORIAL NOTES — NOT PUBLISHABLE.
Everything below this line is for in-repo review and pre-publish prep.
Do NOT paste the rest of this file into Medium.
=========================================================================
-->

---

## Publication notes

- **Target**: Google Cloud Community (same as posts #1 and #2). Series continuity + search ranking.
- **Tags** (Medium max 5, ordered by reader intent): `BigQuery`, `AI Agents`, `LLM`, `Google Cloud`, `Observability`. Swap `CI/CD` (post #2) for `LLM` — this post is about quality scoring, not workflow.
- **Code blocks**: three Gist embed candidates flagged inline. CLI one-liner is the hero block.
- **Cover image**: real GHA failure showing the Hallucination step expanded with the FAIL lines + judge justification visible. The justification *is* the differentiator vs. a hand-rolled judge.
- **Inline images**:
  1. Section 4 cover (above) — primary visual.
  2. Optional section 6 INFORMATION_SCHEMA result pane.
- **Image alt text + captions**:
  - Cover: *"Hallucination judge step fails 10/10 sessions; the FAIL line names a session and quotes the model's own justification — `Jordan Lee (Design)` was never in the tool output."*
  - Section 6: *"Three sdk_feature rows after a day of CI runs: eval-code, eval-llm-judge, trace-read."*
- **Callouts**: two pull-quotes:
  - *"Latency you can measure. Hallucination you have to score."*
  - *"Your CI shouldn't only check what you wrote. It should check whether the agent stays grounded in what your tools said."*
- **Word count**: ~1,520 words of prose (target 1,400–1,800).

## Sandbox demo + screenshot capture

Live evidence captured 2026-04-27:

- Sandbox repo: https://github.com/caohy1988/bqaa-ci-sandbox
- Failing PR: https://github.com/caohy1988/bqaa-ci-sandbox/pull/3
- GHA run with the FAIL output: https://github.com/caohy1988/bqaa-ci-sandbox/actions/runs/25021548083
- Workflow uses `bigquery-agent-analytics>=0.2.3` (the live-verified AI.GENERATE release).
- Hallucination judge runs first; deterministic budgets follow but are skipped on first failure (matches the post's "the rest are gray" framing).

The IAM and AI.GENERATE setup needed to reproduce:

- Service account `bqaa-ci-sandbox@test-project-0728-467323.iam.gserviceaccount.com` granted `roles/bigquery.jobUser` (project) + `READER` on the dataset + `roles/aiplatform.user` (project, for the AI.GENERATE end-user-credentials path).
- No `connection_id` argument needed since the SDK's PR #45 made it optional. The CI workflow runs against end-user creds.

## Gists for embedded code blocks

Three inline code blocks flagged `<!-- Gist embed candidate: ... -->`. Suggested filenames:

- `gists/08_llm_judge_hallucination_one_liner.sh` — section 3 hero command.
- `gists/09_three_judge_workflow_excerpt.yml` — section 5 stack-judges.
- `gists/10_information_schema_eval_llm_judge.sql` — section 6 cost pivot.

Same process as posts #1 and #2: create on the SDK-owner GitHub account before publication; replace inline blocks with Medium's Gist embed widget.

## Open items before publish

1. **Capture cover screenshot.** Open https://github.com/caohy1988/bqaa-ci-sandbox/actions/runs/25021548083 → expand "Hallucination judge" step → screenshot the step header + the `--exit-code:` line + 3–5 FAIL lines. Crop tight; PNG; light-theme browser; place in `screenshots/cover_post3_hallucination.png`.
2. **Optionally capture section 6 cost pivot.** Run the SQL from gist 10 in the BQ console after a day of CI runs against the sandbox; screenshot the result pane.
3. **Create the three Gists** under the SDK-owner account; swap the `TBD:` URL in the section 4 micro-CTA.
4. **DevRel review.** Same path as posts #1 and #2.
5. **Strip both `EDITORIAL NOTES — NOT PUBLISHABLE` blocks** before pasting to Medium.
6. **Tags + canonical URL** at submission time.
7. **Cross-check post #2 publication status.** This post forward-references post #2; if post #2 is still in DevRel review at the time post #3 is ready, hold post #3 until post #2 is live so the reader can follow the series.

## Distribution plan (day of publish)

- Close issue #82 with a link to the published post.
- LinkedIn — lead with the cover screenshot. Hook: *"Latency you can measure. Hallucination you have to score. We turned that into a CI gate against yesterday's production traces."*
- ADK / agent-observability community threads (Discord, r/LangChain, etc.) — frame as "how we caught a fabricated booking confirmation at merge time."
- Update sandbox repo README to point at the published post.

## Boost nomination

Same shape as post #2: ask the Google Cloud Community editor whether this post fits the Boost criteria (original tooling, non-promotional, concrete CI recipe, reproducible sandbox).

## Related

- Series plan: https://github.com/GoogleCloudPlatform/BigQuery-Agent-Analytics-SDK/issues/51
- Post #3 plan + maintainer thread: https://github.com/GoogleCloudPlatform/BigQuery-Agent-Analytics-SDK/issues/82
- Post #1 published: https://medium.com/google-cloud/your-bigquery-agent-analytics-table-is-a-graph-heres-how-to-see-it-via-sdk-920b4ea14731
- Post #2 draft: PR #17 in this repo
- SDK release with the live-verified AI.GENERATE judge path: 0.2.3 on PyPI (2026-04-27)
- Quote-escape polish for FAIL lines (cosmetic): GoogleCloudPlatform/BigQuery-Agent-Analytics-SDK#84
