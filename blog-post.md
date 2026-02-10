# Your AI Agent Is a Black Box — BigQuery Agent Analytics Fixes That

*How to add production-grade observability to your Agent Starter Pack project in one flag*

---

You've built your AI agent with [Agent Starter Pack](https://github.com/GoogleCloudPlatform/agent-starter-pack). It's deployed. Users are talking to it. But can you answer these questions?

- Which tools does your agent call most frequently?
- How many tokens is each conversation burning?
- When a user reports a bad answer, can you trace exactly what happened?
- Are error rates going up or down this week?

If not, you're flying blind — and that's a problem once your agent hits production.

The **BigQuery Agent Analytics Plugin**, now integrated directly into Agent Starter Pack, gives you a full event-level audit trail of every agent interaction — queryable with SQL, visualizable in Looker Studio, and analyzable with BigQuery's built-in AI functions. And enabling it takes exactly one CLI flag.

---

## What Is BigQuery Agent Analytics?

BigQuery Agent Analytics is an [ADK (Agent Development Kit)](https://google.github.io/adk-docs/) plugin that streams detailed event data from your agent directly into a BigQuery table. Every LLM request, tool call, agent response, and error gets captured as a structured row in the `agent_events_v2` table.

Think of it as an event-sourced log of everything your agent does, stored in a warehouse built for analytics at any scale.

**Here's what a single user conversation generates:**

| Event Type | Description |
|---|---|
| `INVOCATION_STARTING` | A new user turn begins |
| `AGENT_STARTING` | The agent starts processing |
| `LLM_REQUEST` | Prompt sent to the model |
| `LLM_RESPONSE` | Model response received |
| `TOOL_STARTING` | Agent calls a tool |
| `TOOL_COMPLETED` | Tool returns a result |
| `AGENT_COMPLETED` | Agent finishes processing |
| `INVOCATION_COMPLETED` | The turn is done |

For a simple two-turn conversation with tool calls, that's ~22 events — each with timestamps, session IDs, trace IDs, token usage, latency, and the full content payload. All in BigQuery, ready to query.

---

## Why Should You Care?

If you're building a demo or prototype, you probably don't need this. But the moment your agent serves real users, you need answers to questions that logs and traces alone can't efficiently provide:

### 1. Cost Visibility
LLM tokens cost money. BigQuery Agent Analytics lets you track token consumption per user, per session, per model — and spot runaway costs before they hit your bill.

```sql
SELECT
  agent,
  JSON_VALUE(attributes, '$.model') AS model,
  COUNT(*) AS total_requests,
  SUM(CAST(JSON_VALUE(attributes, '$.usage_metadata.prompt') AS INT64)) AS prompt_tokens,
  SUM(CAST(JSON_VALUE(attributes, '$.usage_metadata.completion') AS INT64)) AS completion_tokens
FROM `my_project.my_dataset.agent_events_v2`
WHERE event_type = 'LLM_RESPONSE'
GROUP BY agent, model
ORDER BY prompt_tokens DESC;
```

### 2. Debugging Production Issues
When a user says "the agent gave me a wrong answer," you can trace the exact sequence of events — what prompt was sent, which tools were called, what they returned, and what the model decided.

```sql
SELECT event_type, agent, timestamp, content, status, error_message
FROM `my_project.my_dataset.agent_events_v2`
WHERE session_id = 'the-problematic-session-id'
ORDER BY timestamp;
```

### 3. Understanding Agent Behavior at Scale
Which tools are used most? Which are failing? How does latency change over time? These are business-level questions that require aggregate analysis — exactly what BigQuery excels at.

```sql
SELECT
  JSON_VALUE(content, '$.tool') AS tool_name,
  COUNT(*) AS call_count,
  COUNTIF(status = 'ERROR') AS error_count,
  ROUND(COUNTIF(status = 'ERROR') / COUNT(*) * 100, 2) AS error_rate_pct
FROM `my_project.my_dataset.agent_events_v2`
WHERE event_type IN ('TOOL_COMPLETED', 'TOOL_ERROR')
GROUP BY tool_name
ORDER BY call_count DESC;
```

### 4. LLM-Powered Analysis
Because your data lives in BigQuery, you can use BigQuery's built-in AI functions to analyze agent conversations *with another LLM* — semantic clustering, quality scoring, topic classification — all without moving data.

---

## Getting Started: One Flag, Full Observability

### Prerequisites
- [Agent Starter Pack](https://github.com/GoogleCloudPlatform/agent-starter-pack) installed (`pipx install agent-starter-pack`)
- A Google Cloud project with BigQuery API enabled
- `gcloud` authenticated with your project

### Step 1: Generate Your Project with BQ Analytics

Add `--bq-analytics` to your `create` command:

```bash
agent-starter-pack create my-agent \
  -a adk \
  -d cloud_run \
  --bq-analytics
```

That's it. The flag does three things automatically:
1. Adds the `google-adk[bigquery-analytics]` dependency to your project
2. Injects the plugin initialization code into `app/agent.py`
3. Configures Terraform infrastructure for BigQuery dataset, GCS bucket, and logging

> **Tip:** The `--bq-analytics` flag works with any ADK-based agent template: `adk`, `adk_a2a`, `agentic_rag`, and `adk_live`.

### Step 2: Look at What Was Generated

Open `app/agent.py` and you'll see the plugin code at the bottom:

```python
from google.adk.plugins.bigquery_agent_analytics_plugin import (
    BigQueryAgentAnalyticsPlugin,
    BigQueryLoggerConfig,
)
from google.cloud import bigquery

# Initialize BigQuery Analytics
_plugins = []
_project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
_dataset_id = os.environ.get("BQ_ANALYTICS_DATASET_ID", "adk_agent_analytics")
_location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

if _project_id:
    try:
        bq = bigquery.Client(project=_project_id)
        bq.create_dataset(f"{_project_id}.{_dataset_id}", exists_ok=True)

        _plugins.append(
            BigQueryAgentAnalyticsPlugin(
                project_id=_project_id,
                dataset_id=_dataset_id,
                location=_location,
                config=BigQueryLoggerConfig(
                    gcs_bucket_name=os.environ.get("BQ_ANALYTICS_GCS_BUCKET"),
                    connection_id=os.environ.get("BQ_ANALYTICS_CONNECTION_ID"),
                ),
            )
        )
    except Exception as e:
        logging.warning(f"Failed to initialize BigQuery Analytics: {e}")

app = App(
    root_agent=root_agent,
    name="app",
    plugins=_plugins,
)
```

The key points:
- The plugin is passed to the `App` constructor via the `plugins` parameter
- The BigQuery dataset is auto-created if it doesn't exist
- If initialization fails (e.g., missing permissions), the agent still works — analytics just won't be captured
- Configuration is driven by environment variables, making it easy to adjust per environment

### Step 3: Test Locally

Set environment variables and start the playground:

```bash
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_CLOUD_LOCATION=us-central1
export BQ_ANALYTICS_DATASET_ID=my_agent_analytics

make install && make playground
```

Send a few queries through the ADK web UI, then check BigQuery:

```bash
bq query --use_legacy_sql=false \
  "SELECT event_type, agent, user_id, timestamp, status
   FROM \`your-project-id.my_agent_analytics.agent_events_v2\`
   ORDER BY timestamp DESC LIMIT 20"
```

You should see rows for each event in your conversations:

```
+----------------------+------------+---------+---------------------+--------+
|     event_type       |   agent    | user_id |     timestamp       | status |
+----------------------+------------+---------+---------------------+--------+
| INVOCATION_COMPLETED | root_agent | user-1  | 2026-02-10 20:26:00 | OK     |
| AGENT_COMPLETED      | root_agent | user-1  | 2026-02-10 20:26:00 | OK     |
| LLM_RESPONSE         | root_agent | user-1  | 2026-02-10 20:26:00 | OK     |
| TOOL_COMPLETED       | root_agent | user-1  | 2026-02-10 20:25:59 | OK     |
| TOOL_STARTING        | root_agent | user-1  | 2026-02-10 20:25:59 | OK     |
| LLM_REQUEST          | root_agent | user-1  | 2026-02-10 20:25:27 | OK     |
| AGENT_STARTING       | root_agent | user-1  | 2026-02-10 20:25:27 | OK     |
| INVOCATION_STARTING  | root_agent | user-1  | 2026-02-10 20:25:27 | OK     |
+----------------------+------------+---------+---------------------+--------+
```

### Step 4: Deploy to Cloud Run

Deploy with the BQ environment variables:

```bash
gcloud run deploy my-agent \
  --source . \
  --memory 4Gi \
  --region us-central1 \
  --no-allow-unauthenticated \
  --set-env-vars "GOOGLE_CLOUD_PROJECT=your-project-id,GOOGLE_CLOUD_LOCATION=us-central1,BQ_ANALYTICS_DATASET_ID=my_agent_analytics"
```

Or, if you generated with CI/CD (`--cicd-runner google_cloud_build`), just push to your repo — the pipeline deploys automatically and Terraform provisions the BigQuery infrastructure.

---

## Advanced: Customizing What Gets Logged

The default configuration logs everything. For production, you might want to:

### Filter Event Types

Only log LLM interactions and tool calls (skip internal agent lifecycle events):

```python
config = BigQueryLoggerConfig(
    event_allowlist=["LLM_REQUEST", "LLM_RESPONSE", "TOOL_COMPLETED", "TOOL_ERROR"],
)
```

### Handle Multimodal Content

For agents that process images, audio, or video (like `adk_live`), large content is automatically offloaded to GCS:

```python
config = BigQueryLoggerConfig(
    gcs_bucket_name="my-agent-logs-bucket",
    connection_id="projects/my-project/locations/us/connections/my-connection",
    log_multi_modal_content=True,
    max_content_length=500 * 1024,  # Offload content > 500KB to GCS
)
```

### Adjust Batching

Control how many events are buffered before writing to BigQuery:

```python
config = BigQueryLoggerConfig(
    batch_size=50,  # Write every 50 events (default varies)
)
```

---

## Building a Dashboard

Raw SQL is powerful, but dashboards make the data accessible to your whole team. Agent Starter Pack includes a pre-built Looker Studio template that connects directly to your `agent_events_v2` table.

**What you can visualize:**
- Agent usage trends over time
- Tool call frequency and error rates
- Token consumption by model and agent
- Latency distributions (time-to-first-token, total response time)
- Session deep-dives for debugging

To set it up:
1. Open the Looker Studio template (linked in your generated project's `docs/`)
2. Connect it to your BigQuery dataset
3. Start exploring

You can also build custom dashboards in Grafana, Metabase, or any BI tool that connects to BigQuery.

---

## The Bigger Picture: Observability as a First-Class Concern

Agent Starter Pack already gives you Cloud Trace telemetry out of the box. BigQuery Agent Analytics complements this by providing a **structured, queryable event store** designed for offline analysis:

| | Cloud Trace | BigQuery Agent Analytics |
|---|---|---|
| **Purpose** | Real-time debugging | Offline analysis & BI |
| **Data format** | Spans and traces | Structured event rows |
| **Query language** | Trace Explorer UI | SQL |
| **Best for** | "Why is this request slow?" | "How is my agent performing this week?" |
| **Always on?** | Yes | Opt-in (`--bq-analytics`) |

Together, they give you full-stack observability: Trace for real-time debugging, BigQuery for business intelligence.

---

## TL;DR

1. Add `--bq-analytics` when creating your Agent Starter Pack project
2. Every agent interaction is automatically streamed to BigQuery
3. Query with SQL, visualize in Looker Studio, analyze with BigQuery AI functions
4. Works with all ADK-based agents (adk, adk_a2a, agentic_rag, adk_live)
5. Zero-config locally — just set `GOOGLE_CLOUD_PROJECT` and `BQ_ANALYTICS_DATASET_ID`

Your agent shouldn't be a black box. With one flag, it isn't.

---

*For more on Agent Starter Pack, see the [GitHub repo](https://github.com/GoogleCloudPlatform/agent-starter-pack) and [developer guide](https://goo.gle/asp-dev). For BigQuery Agent Analytics details, check the [ADK documentation](https://google.github.io/adk-docs/) and the [Google Codelab](https://codelabs.developers.google.com/adk-bigquery-agent-analytics-plugin).*
