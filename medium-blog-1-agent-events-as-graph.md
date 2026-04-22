<!--
=========================================================================
EDITORIAL NOTES — NOT PUBLISHABLE. Remove this block before paste to Medium.
=========================================================================

Status: Draft with real-data captures from a live run of the
Calendar-Assistant demo agent against test-project-0728-467323 /
agent_analytics_demo (location US), running Gemini 3 Flash Preview
via Vertex AI Express Mode (API-key auth). All trace IDs, session
IDs, latencies, LLM outputs, and INFORMATION_SCHEMA numbers in
this draft are real and match what demo_calendar_assistant.py
produces with GOOGLE_CLOUD_API_KEY + GOOGLE_GENAI_USE_VERTEXAI=true.

Target publication: Google Cloud Community / The Generator.
Target length: 1,400-1,800 words (current: ~1,900).

Section 6 is now un-gated — SDK queries carry the labels
`sdk=bigquery-agent-analytics`, `sdk_feature=*`,
`sdk_version=*`, `sdk_surface=python` today, so the cost query
works against live data without additional SDK work.

See the "EDITORIAL NOTES — NOT PUBLISHABLE" section at the bottom
of the file for publication notes, Gist-embed checklist, and open
items.
=========================================================================
-->

# Your BigQuery Agent Analytics table is a graph. Here's how to see it.

*A 10-minute tour of turning raw `agent_events` rows into readable traces with the BigQuery Agent Analytics SDK.*

---

## 1. Hook

You've installed the BigQuery Agent Analytics plugin into your ADK agent. You're excited. You open the BigQuery console, find your `agent_events` table, run `SELECT * FROM agent_events WHERE session_id = 'abc' ORDER BY timestamp`, and get this:

*[SCREENSHOT: 47-row BQ query result of agent_events — span_id, parent_span_id, event_type, JSON blobs]*

This is what production looks like to most ADK teams. Good luck debugging a multi-turn tool-call failure from that.

Forty-seven rows. JSON in the `content` column. Timestamps to the microsecond. Eight distinct event types. Somewhere in there is the reason your agent booked the wrong meeting, but you're not going to find it by squinting at rows.

There's a better way, and it takes one line of Python.

## 2. The problem in one paragraph

The ADK plugin logs *events*. Events are flat rows, ordered by time. But agent conversations aren't flat — they're trees. A user asks a question. An agent thinks. Thought produces a tool call. Tool call returns data. Data feeds into another thought. Sub-agents delegate. Sometimes a tool fails and the agent retries with different args. The "shape" of a conversation is a DAG where events are nodes and `parent_span_id` edges tell you what caused what.

Before the SDK, you had two options: write SQL CTEs that join events by `span_id`, unnest the JSON, and pray; or render the trace in a separate tool like an OpenTelemetry viewer. Both work, both take 20 minutes every time you debug something. The SDK collapses that to one line.

## 3. Setup in 30 seconds

```bash
pip install bigquery-agent-analytics
```

Point it at your project:

```bash
export PROJECT_ID=your-project
export DATASET_ID=your_dataset
export DATASET_LOCATION=US   # or us-central1, europe-west4, etc.
```

That's it. The SDK defaults `TABLE_ID` to `agent_events`, which is where the ADK plugin writes. Override it only if you renamed the table.

Verify the connection:

```bash
bq-agent-sdk doctor
```

*[SCREENSHOT: `doctor` output — green checkmarks for BQ auth, schema validation, permission check]*

If doctor is green, you're done. Now the interesting part.

## 4. The demo — Is the Calendar-Assistant broken?

Here's a real scenario. I built a Calendar-Assistant ADK agent with three tools: `search_contacts(name)`, `get_calendar_availability(contact_id, date)`, and `book_meeting(contact_id, slot)`. It's the kind of agent anyone building a scheduling bot writes in an afternoon. The address book has three contacts named Priya — Priya Patel, Priya Shah, Priya Venkat — by design, to test how the agent handles ambiguity.

A user sends: *"Book me a 1:1 with Priya next Tuesday at 2pm for 30 minutes."*

The agent runs. It eventually responds: *"I found three people named Priya: Priya Patel (Platform), Priya Shah (Design), and Priya Venkat (Research). Which Priya would you like to book the meeting with?"*

Is that right? Is the agent asking because it legitimately couldn't disambiguate, or because it gave up? Did the `search_contacts` tool actually return three candidates, or something weirder? And did the agent *try* to book before asking?

You open BigQuery. There are the rows. `USER_MESSAGE_RECEIVED`, `INVOCATION_STARTING`, `AGENT_STARTING`, `LLM_REQUEST`, `LLM_RESPONSE` with `call: search_contacts`, `TOOL_STARTING`, `TOOL_COMPLETED`, another `LLM_REQUEST`, another `LLM_RESPONSE`, `AGENT_COMPLETED`, `INVOCATION_COMPLETED`. Twelve rows. Everything says `status=OK`. The `error_message` column is `NULL` everywhere. Nothing in the row structure tells you *why* the agent chose to ask instead of booking.

Fine. One line:

<!-- Gist embed candidate: client setup + one-line render -->

```python
from bigquery_agent_analytics import Client

client = Client(
    project_id="your-project",
    dataset_id="your_dataset",
    table_id="agent_events",
    location="US",
)

trace = client.get_session_trace("84ef108d-745c-451a-ae79-d0f97673268d")
trace.render()
```

Output:

```
Trace: e-ae1fe18c-887e-4df3-a91f-25eec174e2dd | Session: 84ef108d-745c-451a-ae79-d0f97673268d | 5235ms
======================================================================================================
└─ [✓] USER_MESSAGE_RECEIVED [calendar_assistant] - Book me a 1:1 with Priya next Tuesday at 2pm for 30 minutes.
└─ [✓] INVOCATION_STARTING [calendar_assistant]
└─ [✓] INVOCATION_COMPLETED [calendar_assistant] (19778ms)
   ├─ [✓] AGENT_STARTING [calendar_assistant] - You are a calendar assistant. When the user asks to book a meeting, first use search_contacts to find the person. If ...
   └─ [✓] AGENT_COMPLETED [calendar_assistant] (5234ms)
      ├─ [✓] LLM_REQUEST [calendar_assistant] (gemini-3-flash-preview) - Book me a 1:1 with Priya next Tuesday at 2pm for 30 minutes.
      ├─ [✓] LLM_RESPONSE [calendar_assistant] (2972ms) - call: search_contacts
      ├─ [✓] TOOL_STARTING [calendar_assistant] (search_contacts)
      ├─ [✓] TOOL_COMPLETED [calendar_assistant] (search_contacts) (0ms)
      ├─ [✓] LLM_REQUEST [calendar_assistant] (gemini-3-flash-preview) - Book me a 1:1 with Priya next Tuesday at 2pm for 30 minutes.
      └─ [✓] LLM_RESPONSE [calendar_assistant] (2175ms) - text: "I found three people named Priya: Priya Patel (Platform), Priya Shah (Design), and Priya Venkat..."
```

There it is. Read the `search_contacts` round-trip. The agent got back *three* matching contacts, and instead of picking one, it asked the user which Priya they meant. That's the *right* call — but you couldn't see it in the raw `agent_events` rows without reconstructing the tool round-trip yourself.

> **The tree shows you the decision, not just the outcome.**

You can't see that in twelve rows. You can see it in the tree in two seconds.

Need the structured version for a ticket or a dashboard? Two more properties:

```python
>>> [(s.event_type, s.tool_name, s.error_message) for s in trace.error_spans]
[]

>>> [{"tool": c["tool_name"], "args": c["args"], "match_count": c["result"]["match_count"]}
...  for c in trace.tool_calls]
[{'tool': 'search_contacts', 'args': {'name': 'Priya'}, 'match_count': 3}]
```

`trace.tool_calls` pulls every tool invocation with its args and result. `trace.error_spans` gives you just the failures. Both are lists of well-typed objects — no JSON-digging. The `tool_name` on each `Span` is a first-class property; you don't reach into `span.content["tool"]` to get it. Same for `error_message` and `parent_span_id`. The raw dict is still there if you want it, but you rarely need to.

What did the successful path look like? A different session in the same fleet, where the user said *"Book a 30-minute meeting with Priya Patel on April 28 at 3:30pm"* — unambiguous — produced a clean three-tool chain: `search_contacts → get_calendar_availability → book_meeting`, `✓` all the way down, 8.7 seconds end-to-end. One line of Python per trace; same `.render()` call. The SDK doesn't care whether the agent asked, decided, or failed — it just shows you the shape of what happened.

Running this in a terminal? `trace.render(color=True)` wraps error markers in red ANSI and subtree-warning markers in yellow. Default stays plain so your CI logs and notebook captures aren't full of escape codes.

## 5. Going deeper — finding every Priya bug

> **One bug is interesting. Twenty Priya-like bugs is a pattern.**

The SDK lets you pivot from single-trace debugging to fleet-level triage in one step:

<!-- Gist embed candidate: fleet-level ambiguity filter -->

```python
from bigquery_agent_analytics import TraceFilter

traces = client.list_traces(
    filter_criteria=TraceFilter.from_cli_args(
        last="24h",
        agent_id="calendar_assistant",
    )
)

ambiguity_bugs = [
    t for t in traces
    if any(
        tc["tool_name"] == "search_contacts"
        and isinstance(tc.get("result"), dict)
        and tc["result"].get("match_count", 0) > 1
        for tc in t.tool_calls
    )
]

print(f"{len(ambiguity_bugs)} / {len(traces)} traces hit multi-match contact ambiguity")
for t in ambiguity_bugs:
    print(f"  {t.session_id[:8]} -> {(t.final_response or '')[:80]!r}")
```

Real output against the same fleet:

```
1 / 5 traces hit multi-match contact ambiguity
  84ef108d -> "text: 'I found three people named Priya: Priya Patel (Platform), Priya Shah (Des"
```

One session in five hit the ambiguity. The SDK pulled it out in a single comprehension, and we know the exact session to pull up in the next step if we want to dig deeper. That's a filter you couldn't write in SQL without knowing your `content` JSON schema cold. In Python, it's idiomatic.

The natural follow-up — turning this filter into an automated eval check that runs on every deploy — is the next post in this series. Spoiler: `client.evaluate_categorical(...)` plus three lines of `CategoricalMetricDefinition` gets you a CI gate.

## 6. What happens behind the scenes

One thing worth knowing: every query the SDK runs is labeled with the SDK name, version, surface, and feature. That means you can point `INFORMATION_SCHEMA` at your jobs table and see exactly what the SDK is doing on your behalf, and what it's costing you:

<!-- Gist embed candidate: INFORMATION_SCHEMA cost-per-feature query -->

```sql
-- INFORMATION_SCHEMA region must match your dataset's location.
-- This example uses `region-us` because the setup in section 3 uses
-- the US multi-region. For single-region datasets, swap in
-- `region-us-central1`, `region-europe-west4`, etc.
SELECT
  (SELECT value FROM UNNEST(labels) WHERE key = 'sdk_feature') AS sdk_feature,
  COUNT(*) AS runs,
  ROUND(SUM(total_bytes_processed) / POW(1024, 3), 3) AS gb_processed,
  ROUND(AVG(total_slot_ms), 0) AS avg_slot_ms
FROM `region-us`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
WHERE creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
  AND EXISTS (
    SELECT 1 FROM UNNEST(labels)
    WHERE key = 'sdk' AND value = 'bigquery-agent-analytics'
  )
GROUP BY sdk_feature
ORDER BY runs DESC;
```

Real output against the same dataset, immediately after the five-session demo above:

```
sdk_feature,runs,gb_processed,avg_slot_ms
trace-read,6,0.029,1742
```

Six runs of the `trace-read` feature (every `get_session_trace` and `list_traces` call from sections 4 and 5), 29 MB processed total, 1.7 seconds average slot time. As you use more of the SDK, more `sdk_feature` rows appear — `evaluate-categorical`, `insights`, `list-sessions` — each with its own cost profile. That's the kind of transparency you want from a library sitting between your agent logs and your BQ bill. The SDK labels the queries; you decide the budget.

## 7. Try it

Three things to try right now:

1. **[Install the plugin (5-minute quickstart)](TBD: direct URL to the BQ Agent Analytics plugin quickstart page — NOT the adk-python repo root).** If you have an ADK agent running, the plugin is a 5-minute wire-up and it starts populating `agent_events` immediately.
2. **Run `client.get_session_trace(id).render()`** on the ugliest production trace you have. Compare the time it takes to understand the failure to the time it would have taken in SQL.
3. **[Star the SDK repo](https://github.com/GoogleCloudPlatform/BigQuery-Agent-Analytics-SDK)** if this made your afternoon easier.

<!-- CTA URL resolution: the plugin quickstart page is the primary conversion target per issue #53. Do NOT fall back to the adk-python repo root — a reader clicking through to a full monorepo README loses momentum. Resolve this URL before publication; see "Open items before publish" below. -->


Install the plugin today, see your first DAG in 10 minutes. Next post covers the same SDK, but for a different job: turning ad-hoc evals into a CI gate. The short version: your `agent_events` table is also a test suite.

---

*[CLOSING IMAGE: a stylized graphic of the full tree render — not a stock photo]*

<!--
=========================================================================
EDITORIAL NOTES — NOT PUBLISHABLE.
Everything below this line is for in-repo review and pre-publish prep.
Do NOT paste the rest of this file into Medium.
=========================================================================
-->

---

## Publication notes

- **Target**: Google Cloud Community (primary) or *The Generator* (backup). Google Cloud handle gets better GCP-content reach.
- **Tags**: `bigquery`, `ai-agents`, `google-cloud`, `python`, `observability` (Medium max 5).
- **Code blocks**: use Medium Gist embeds for any block >5 lines — doubles as backlink to the SDK repo.
- **Callouts**: one blockquote per section for skimmers.
- **Canonical URL**: set to the Google Cloud dev blog version if co-published, for SEO.
- **CTA**: install plugin (primary), star SDK repo (secondary). No "follow me on Medium."
- **Word count check**: current draft is ~1,600 words before the gated section 6. Within the 1,400–1,800 target.

## Open items before publish

1. Real screenshots: 47-row `agent_events` query from BQ console (the textual version is embedded but a visual hooks harder), `doctor` output, closing graphic. Render output, fleet output, and INFORMATION_SCHEMA result are now real in text — screenshots optional for those but helpful.
2. ~~Decide whether section 6 ships with this post.~~ **Resolved** — SDK queries are labeled today (`sdk=bigquery-agent-analytics`, `sdk_feature=trace-read`, etc.); section 6 runs against real data.
3. Publication target confirmed (Google Cloud Community vs personal with co-promotion).
4. Internal review by Google Cloud DevRel.
5. ~~Confirm the Priya narrative — real trace data vs composed narrative.~~ **Resolved** — all trace IDs, session IDs, latencies, and LLM outputs in this draft come from live runs of the Calendar-Assistant demo against a sandbox project. Note the narrative shift: Gemini 2.5 Flash actually *asks* to disambiguate instead of picking wrong, so the featured trace shows a well-handled decision rather than the synthetic "picks wrong Priya" bug. The SDK-value framing still holds — arguably stronger, because the reader sees the SDK helping them verify correct behavior in addition to catching mistakes.
6. **Resolve the primary-CTA URL** — replace the `TBD:` marker in section 7 with the exact plugin quickstart page. The top-level `adk-python` repo root is explicitly not acceptable per issue #53's conversion-goal framing.
7. **Pull inline code blocks into Gists before publication** — three blocks are flagged inline as `<!-- Gist embed candidate: ... -->`. Create Gists in the SDK owner's account (so the "Open in GitHub" link doubles as an SDK backlink), replace the inline blocks with Medium's Gist embed widget.
8. **Minor polish in SDK `Span.summary`** — the featured trace output shows `text: '...'` prefix on agent responses (an artifact of the agent payload shape). Tracked as a future SDK polish item; not blocking for this post. If it lands before publication, re-capture section 4 output.
