<!--
=========================================================================
EDITORIAL NOTES — NOT PUBLISHABLE. Remove this block before paste to Medium.
=========================================================================

Status: Draft for internal review (issue #53).
Target publication: Google Cloud Community / The Generator.
Target length: 1,400-1,800 words.
Screenshots and opening image TBD.
Section 6 depends on query-labeling work tracked separately; can ship
without it by removing that section.

See the "EDITORIAL NOTES — NOT PUBLISHABLE" section at the bottom of the
file for publication notes, Gist-embed checklist, and open items.
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
export DATASET_LOCATION=us-central1
```

That's it. The SDK defaults `TABLE_ID` to `agent_events`, which is where the ADK plugin writes. Override it only if you renamed the table.

Verify the connection:

```bash
bq-agent-sdk doctor
```

*[SCREENSHOT: `doctor` output — green checkmarks for BQ auth, schema validation, permission check]*

If doctor is green, you're done. Now the interesting part.

## 4. The demo — The Calendar-Assistant bug

Here's a real scenario. I built a Calendar-Assistant ADK agent with three tools: `search_contacts(name)`, `get_calendar_availability(contact_id, date_range)`, and `book_meeting(contact_id, slot)`. It's the kind of agent anyone building a scheduling bot writes in an afternoon.

A user sends: *"Book me a 1:1 with Priya next Tuesday."*

The agent runs. It eventually responds: *"Done! 1:1 with Priya P. booked for Tuesday at 2pm."*

Except the meeting got booked with the wrong Priya.

You open BigQuery. There are the rows. `USER_MESSAGE_RECEIVED`, `AGENT_STARTING`, a `LLM_REQUEST`, some `LLM_RESPONSE` spans, three `TOOL_STARTING` rows, three `TOOL_COMPLETED` rows, one more `LLM_RESPONSE`, `AGENT_COMPLETED`. Everything says `status=OK`. The `error_message` column is `NULL` everywhere. There is no bug in these rows — there's a *judgment* bug somewhere in the tool results.

Fine. One line:

<!-- Gist embed candidate: client setup + one-line render -->

```python
from bigquery_agent_analytics import Client

client = Client(
    project_id="your-project",
    dataset_id="your_dataset",
    table_id="agent_events",
    location="us-central1",
)

trace = client.get_session_trace("d4f2-a1b3-book-priya")
trace.render()
```

Output:

```
Trace: d4f2-a1b3-book-priya | Session: sess-041 | 4382ms
===========================================================
└─ [✓] USER_MESSAGE_RECEIVED - Book me a 1:1 with Priya next Tuesday
   └─ [✓] AGENT_STARTING [calendar_assistant]
      ├─ [✓] LLM_REQUEST [calendar_assistant] (gemini-2.5-flash) (820ms)
      ├─ [✓] LLM_RESPONSE [calendar_assistant] (120ms) - I'll search for Priya's contact info...
      ├─ [✓] TOOL_STARTING [calendar_assistant] (search_contacts)
      ├─ [✓] TOOL_COMPLETED [calendar_assistant] (search_contacts) (180ms) - 3 matches: Priya P., Priya S., Priya V.
      ├─ [✓] LLM_REQUEST [calendar_assistant] (gemini-2.5-flash) (640ms)
      ├─ [✓] LLM_RESPONSE [calendar_assistant] (95ms) - Priya P. is probably who they mean.
      ├─ [✓] TOOL_STARTING [calendar_assistant] (get_calendar_availability)
      ├─ [✓] TOOL_COMPLETED [calendar_assistant] (get_calendar_availability) (210ms) - Tue 2pm free
      ├─ [✓] TOOL_STARTING [calendar_assistant] (book_meeting)
      ├─ [✓] TOOL_COMPLETED [calendar_assistant] (book_meeting) (1100ms) - booked
      └─ [✓] AGENT_COMPLETED [calendar_assistant] - Done! 1:1 with Priya P. booked for Tuesday at 2pm.
```

There it is. Read the third-from-top tool result: *"3 matches: Priya P., Priya S., Priya V."* The agent got three candidates back, then picked one without asking the user which Priya they meant. The book_meeting tool succeeded because the agent gave it a valid `contact_id`. Nothing errored.

> **The bug is the decision, not the failure.**

You can't see that in rows. You can see it in the tree in two seconds.

Need the structured version for a ticket or a dashboard? Two more properties:

```python
>>> [(s.event_type, s.tool_name, s.error_message) for s in trace.error_spans]
[]

>>> [{"tool": c["tool_name"], "args": c["args"]} for c in trace.tool_calls]
[{"tool": "search_contacts", "args": {"name": "Priya"}},
 {"tool": "get_calendar_availability", "args": {"contact_id": "pp-412", "date_range": "..."}},
 {"tool": "book_meeting", "args": {"contact_id": "pp-412", "slot": "2024-04-23T14:00"}}]
```

`trace.tool_calls` pulls every tool invocation with its args and result. `trace.error_spans` gives you just the failures. Both are lists of well-typed objects — no JSON-digging. The `tool_name` on each `Span` is a first-class property; you don't reach into `span.content["tool"]` to get it. Same for `error_message` and `parent_span_id`. The raw dict is still there if you want it, but you rarely need to.

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

print(f"{len(ambiguity_bugs)} / {len(traces)} traces matched an ambiguous contact")
```

That's a filter you couldn't write in SQL without knowing your `content` JSON schema cold. In Python, it's idiomatic.

The natural follow-up — turning this filter into an automated eval check that runs on every deploy — is the next post in this series. Spoiler: `client.evaluate_categorical(...)` plus three lines of `CategoricalMetricDefinition` gets you a CI gate.

## 6. What happens behind the scenes

*[Section gated on query labeling work — see companion issue. Remove this section if shipping before that lands.]*

One thing worth knowing: every query the SDK runs is labeled with the SDK version and the call site. That means you can point `INFORMATION_SCHEMA` at your jobs table and see exactly what the SDK is doing on your behalf, and what it's costing you:

<!-- Gist embed candidate: INFORMATION_SCHEMA cost-per-call-site query -->

```sql
-- INFORMATION_SCHEMA region must match your dataset's location.
-- This example uses us-central1 to match the setup in section 3;
-- swap in region-us, region-europe-west4, etc. as appropriate.
SELECT
  labels.value AS sdk_call_site,
  COUNT(*) AS runs,
  SUM(total_bytes_processed) / POW(1024, 3) AS gb_processed,
  AVG(total_slot_ms) AS avg_slot_ms
FROM `region-us-central1`.INFORMATION_SCHEMA.JOBS
LEFT JOIN UNNEST(labels) AS labels
WHERE labels.key = 'bq_agent_sdk_call'
  AND creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
GROUP BY sdk_call_site
ORDER BY gb_processed DESC;
```

*[SCREENSHOT: INFORMATION_SCHEMA query result — top five SDK call sites, bytes processed, runs]*

That's the kind of transparency you want from a library sitting between your agent logs and your BQ bill. The SDK labels the queries; you decide the budget.

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

1. Real screenshots: 47-row `agent_events` query, `doctor` output, render output, INFORMATION_SCHEMA result, closing graphic.
2. Decide whether section 6 ships with this post or is held for a companion "how the SDK stays observable" post later.
3. Publication target confirmed (Google Cloud Community vs personal with co-promotion).
4. Internal review by Google Cloud DevRel.
5. Confirm the Priya narrative — do we use real trace data (dogfooded Calendar-Assistant in a sandbox project) or a composed narrative that reads real?
6. **Resolve the primary-CTA URL** — replace the `TBD:` marker in section 7 with the exact plugin quickstart page. The top-level `adk-python` repo root is explicitly not acceptable per issue #53's conversion-goal framing.
7. **Pull inline code blocks into Gists before publication** — three blocks are flagged inline as `<!-- Gist embed candidate: ... -->`. Create Gists in the SDK owner's account (so the "Open in GitHub" link doubles as an SDK backlink), replace the inline blocks with Medium's Gist embed widget.
