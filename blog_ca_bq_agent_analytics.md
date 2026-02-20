# BigQuery Agent Analytics + Conversational Analytics: A Closed-Loop for AI Agent Operations

*How to instrument, log, and analyze your AI agent's behavior using nothing but BigQuery — and then query those logs with natural language.*

---

If you're building AI agents with Google's [Agent Development Kit (ADK)](https://google.github.io/adk-docs/), you've probably asked yourself: *"What is my agent actually doing? Where is it slow? What are users asking? Is it leaking PII?"*

These are Agent Ops questions — and traditionally, answering them means setting up separate logging pipelines, shipping data to observability platforms, and writing dashboard after dashboard of hand-crafted SQL.

What if I told you that BigQuery can handle the *entire* loop — from capturing every agent event to analyzing them with plain English questions?

In this post, I'll walk you through a **closed-loop integration** that connects two powerful BigQuery capabilities:

1. **[BigQuery Agent Analytics Plugin](https://google.github.io/adk-docs/integrations/bigquery-agent-analytics/)** — automatically logs every agent event (user messages, LLM calls, tool executions) into a BigQuery table
2. **[Conversational Analytics (CA)](https://cloud.google.com/bigquery/docs/conversational-analytics)** — lets you query that same table using natural language, with the CA Data Agent writing SQL, returning results, and generating insights for you

The result is a workflow that looks like this:

```
Agent runs → BQ AA Plugin auto-logs events → BQ Table → CA Data Agent → NL Insights
```

No external observability stack. No manual dashboard building. Just BigQuery.

## What we're building

We'll use a real agent — `my_bq_agent` — built with Google ADK's `BigQueryToolset` and Gemini. This agent helps users explore NYC Citi Bike data in BigQuery. Nothing fancy, but it generates a rich set of real events: user messages, LLM requests and responses, tool calls, errors, and completion signals.

The **BigQuery Agent Analytics Plugin** captures all of this automatically. And then we point **Conversational Analytics** at the resulting event log table to unlock natural language analysis.

Here's the full companion notebook if you want to follow along: [NY_City_Bike_Agent_Logging.ipynb](https://github.com/haiyuan-eng-google/demo_BQ_agent_analytics_plugin_notebook/blob/main/NY_City_Bike_Agent_Logging.ipynb)

## Step 1: Instrument your agent with the BQ AA Plugin

The setup is refreshingly simple — a single plugin line:

```python
from google.adk.plugins.bigquery_agent_analytics_plugin import BigQueryAgentAnalyticsPlugin

bq_plugin = BigQueryAgentAnalyticsPlugin(
    project_id=PROJECT_ID,
    dataset_id=DATASET_ID,
    table_id=TABLE_ID,
)

app = App(
    name="my_bq_agent",
    root_agent=root_agent,
    plugins=[bq_plugin],  # That's it!
)
```

That's all the instrumentation you need. The plugin automatically creates the `agent_events` table if it doesn't exist, and streams every event from every invocation. No manual logging calls. No code scattered through your agent.

The resulting table captures a rich schema: `timestamp`, `event_type`, `session_id`, `invocation_id`, `user_id`, `content` (JSON), `latency_ms` (JSON), `status`, `error_message`, and more.

## Step 2: Run the agent to generate real event data

To make this self-contained, we run the agent with sample queries from multiple simulated users:

```python
from google.adk.runners import InMemoryRunner

runner = InMemoryRunner(app=app)

user_queries = [
    ("user1", [
        "What datasets and tables are available for NYC Citi Bike data?",
        "Show me the average trip duration in minutes",
        "Which are the top 5 most popular start stations?",
    ]),
    ("user2", [
        "What is the average trip duration by user type?",
        "Show me the number of trips per month for the last available year",
    ]),
    # ...more users with different query patterns
]
```

After running these, we wait 30 seconds for BigQuery streaming writes to flush, then verify:

```
| event_type             | count |
|:-----------------------|------:|
| LLM_RESPONSE           |    96 |
| TOOL_COMPLETED         |    74 |
| LLM_REQUEST            |    63 |
| TOOL_STARTING          |    58 |
| AGENT_COMPLETED        |    50 |
| USER_MESSAGE_RECEIVED  |    10 |
| INVOCATION_STARTING    |    10 |
```

We now have real agent telemetry data in BigQuery. Time to analyze it.

## Step 3: Create a Conversational Analytics Data Agent

Here's where it gets interesting. Instead of writing all our analysis queries by hand, we create a **CA Data Agent** that understands our event log table and can answer questions in natural language.

The key is giving the CA agent enough context — schema descriptions, glossary terms, and verified queries:

```python
from google.cloud import geminidataanalytics

# Point CA at our BQ AA Plugin's event log table
bq_table_ref = geminidataanalytics.BigQueryTableReference(
    project_id=PROJECT_ID,
    dataset_id=DATASET_ID,
    table_id=TABLE_ID,
    schema=geminidataanalytics.Schema(
        description="Agent event logs auto-captured by the BigQuery Agent Analytics Plugin.",
        fields=[
            geminidataanalytics.Field(
                name="event_type",
                description="Type of agent event: USER_MESSAGE_RECEIVED, LLM_REQUEST, "
                "LLM_RESPONSE, TOOL_STARTING, TOOL_COMPLETED, TOOL_ERROR, "
                "INVOCATION_STARTING, AGENT_COMPLETED"
            ),
            geminidataanalytics.Field(
                name="latency_ms",
                description="JSON field containing total_ms and time_to_first_token_ms"
            ),
            # ...more field descriptions
        ],
    ),
)
```

We also define **glossary terms** so the CA agent understands domain-specific vocabulary like "invocation" and "session":

```python
glossary_terms = [
    geminidataanalytics.GlossaryTerm(
        display_name="event_type",
        description="The type of agent event. Values include: USER_MESSAGE_RECEIVED, "
        "LLM_REQUEST, LLM_RESPONSE, TOOL_STARTING, TOOL_COMPLETED..."
    ),
    geminidataanalytics.GlossaryTerm(
        display_name="session",
        description="A single conversation between a user and the agent, "
        "identified by session_id"
    ),
]
```

And **verified queries** — pre-validated SQL that guides the CA agent for common analysis patterns:

```python
example_queries = [
    geminidataanalytics.ExampleQuery(
        natural_language_question="Show usage monitoring — daily active users, "
        "sessions, invocations, and average latency",
        sql_query=f"SELECT DATE(timestamp) AS usage_date, ..."
    ),
    # ...more verified queries for errors, performance, anomalies, etc.
]
```

Finally, we create the Data Agent and a conversation:

```python
ca_agent = data_agent_client.create_data_agent_sync(request=create_request)
ca_conversation = data_chat_client.create_conversation(request=conv_request)
```

Our natural language interface to agent operations is now live.

## Step 4: Analyze — Manual SQL vs. Conversational Analytics, side by side

This is the fun part. Throughout the notebook, we pair every manual SQL query with a CA equivalent — asking the same question in plain English. Let's look at a few examples.

### Usage monitoring

**Manual SQL:**

```sql
SELECT
    DATE(timestamp) AS usage_date,
    COUNT(DISTINCT user_id) AS unique_active_users,
    COUNT(DISTINCT session_id) AS total_sessions,
    COUNTIF(event_type = 'INVOCATION_STARTING') AS total_invocations,
    ROUND(AVG(SAFE_CAST(JSON_VALUE(latency_ms, '$.total_ms') AS INT64)), 2)
        AS avg_completion_latency_ms
FROM `project.dataset.agent_events`
GROUP BY usage_date
ORDER BY usage_date DESC
```

**Conversational Analytics:**

```python
responses = ca_ask("Show usage monitoring — daily active users, sessions, "
                   "invocations, and average latency")
display_ca_response(responses)
```

Both return the same data. But the CA response also includes automatically generated insights: *"Feb 9 had the highest activity with 6 unique users and 26 invocations, but also the highest average latency at ~4 seconds."*

### Performance analysis

Ask CA: *"Analyze performance by event type — show average, max, and p99 latency"*

The CA agent generates the SQL, runs it, and explains the results:

> **LLM Bottlenecks:** LLM_RESPONSE has an average latency of ~2.5 seconds, which is often the primary driver of perceived wait times for users.
>
> **Long-Tail Latency:** While the average tool execution (TOOL_COMPLETED) is relatively fast at ~1.6 seconds, it has a massive maximum latency spike of ~23.3 seconds.
>
> **Overall Cycle:** The average total time for an agent to complete a request (AGENT_COMPLETED) is approximately 7.3 seconds.

Try extracting that kind of narrative from raw query results alone!

### High-latency questions

Ask CA: *"Which user questions took the longest to complete?"*

The CA agent joins the `USER_MESSAGE_RECEIVED` events with `AGENT_COMPLETED` events on `invocation_id`, calculates latency, and explains what it finds:

> **Complexity of Trends:** Queries asking for "trends" consistently appear as high-latency events. This is likely due to the agent having to perform multiple data retrievals and aggregation steps before responding.
>
> **Optimization Opportunity:** The agent spends significant time on "discovery" questions. Optimizing metadata caching for these specific intents could reduce initial response latency by several seconds.

## Step 5: Combine CA with BigQuery ML AI functions

BigQuery's built-in AI functions — `AI.DETECT_ANOMALIES`, `AI.CLASSIFY`, `AI.GENERATE` — add another dimension to agent operations analysis.

### Anomaly detection

We use `AI.DETECT_ANOMALIES` to flag unusual latency patterns in agent completions. The SQL uses a 70/30 historical/target split, and the results show whether observed latencies fall within expected bounds.

Then we ask the CA agent the same question: *"Detect latency anomalies in agent completion events."* It produces an equivalent analysis and explains that despite a 28-second spike, the statistical model considers this normal given the workload's high variance.

### User intent classification

Using `AI.CLASSIFY` directly in SQL, we categorize every user message into intent buckets: Trend Analysis, Data Exploration, Location Service, Security, or Other.

```sql
AI.CLASSIFY(
    raw_message,
    categories => ['Trend Analysis', 'Data Exploration',
                    'Location Service', 'Security', 'Other'],
    connection_id => 'project.connection',
    endpoint => 'gemini-2.5-flash'
) AS ai_intent
```

This reveals that **Data Exploration** is the most frequent intent (29 occurrences) with moderate latency (~6.7s), while **Trend Analysis** is less common but significantly slower (~17.1s average).

### PII detection in agent responses

Perhaps the most operationally important analysis: scanning every LLM response for personally identifiable information using `AI.GENERATE`:

```sql
AI.GENERATE(
    prompt => CONCAT(
        'Extract any PII from the following text and return it as JSON...',
        JSON_VALUE(content, '$.response')
    ),
    connection_id => 'project.connection',
    endpoint => 'gemini-2.5-flash'
).result AS extracted_pii_json
```

In our test run, no PII was found — which is exactly what you want to see from a well-behaved agent.

## Why these two features are better together

Each capability is useful on its own. But together, they create something greater than the sum of their parts:

| Capability | BQ AA Plugin alone | CA alone | Both together |
|:---|:---|:---|:---|
| **Event capture** | Automatic, zero-code | N/A | Automatic, zero-code |
| **Ad-hoc analysis** | Manual SQL required | Natural language | NL for exploration, SQL for precision |
| **Insights** | You interpret the data | Auto-generated | Auto-generated from your own agent's data |
| **Onboarding** | Need SQL skills | Just ask questions | Anyone on your team can analyze agent behavior |
| **AI-powered analysis** | Via BQML functions in SQL | Built into CA responses | Both: BQML for custom analysis + CA for NL access |

The combination unlocks a workflow where:

- **Engineers** use manual SQL for precise, custom queries — like PII detection with `AI.GENERATE` or anomaly detection with `AI.DETECT_ANOMALIES`
- **Product managers** ask CA questions like *"What was the busiest hour for my agent?"* or *"Compare average latency between different users"*
- **Security teams** query in natural language: *"Find users asking security-related questions"*
- **Everyone** gets auto-generated insights without writing a single line of SQL

## The full closed loop

Here's what we demonstrated end to end:

1. **Agent runs** — `my_bq_agent` processes user requests using `BigQueryToolset` and Gemini
2. **BQ AA Plugin auto-logs** — Every event (user message, LLM call, tool execution) is automatically streamed to BigQuery
3. **Manual SQL analysis** — Traditional queries for full control over performance, error, and security analysis
4. **CA Data Agent** — The same table is instantly queryable via natural language, with automatic insights
5. **AI-powered analysis** — BigQuery ML functions (`AI.CLASSIFY`, `AI.GENERATE`, `AI.DETECT_ANOMALIES`) enrich both approaches

All within BigQuery. No external observability platform. No separate data pipeline. No dashboard-building sprint.

## Getting started

Want to try this yourself? Here's what you need:

1. A Google Cloud project with BigQuery, Vertex AI, and Conversational Analytics APIs enabled
2. A [BigQuery Cloud Resource Connection](https://cloud.google.com/bigquery/docs/create-cloud-resource-connection) for calling Gemini from SQL
3. The companion notebook: [NY_City_Bike_Agent_Logging.ipynb](https://github.com/haiyuan-eng-google/demo_BQ_agent_analytics_plugin_notebook/blob/main/NY_City_Bike_Agent_Logging.ipynb)

The notebook runs top-to-bottom in Colab, Vertex AI Workbench, or BigQuery Studio. It's fully self-contained — it creates the agent, runs it, generates events, sets up the CA Data Agent, and walks through every analysis pattern described in this post.

The Conversational Analytics API is **free during Preview**, so there's no additional cost to try the NL querying side.

## What's next

This is just the beginning. Some natural extensions:

- **Automated alerting**: Use CA's streaming API to set up NL-triggered alerts (*"Notify me if error rate exceeds 5% in any 10-minute window"*)
- **Multi-agent comparison**: Point the CA agent at event tables from multiple agents to compare performance across your fleet
- **Embedding-based session analysis**: Use `ML.GENERATE_EMBEDDING` and `VECTOR_SEARCH` to find semantically similar user sessions (already demonstrated in the companion [ShopBot notebook](https://github.com/haiyuan-eng-google/demo_BQ_agent_analytics_plugin_notebook/blob/main/Demo_Plan_BigQuery_for_Agent_Ops_Unified_Platform_Public.ipynb))
- **Custom dashboards**: Feed CA-generated Vega-Lite chart specs into Looker or your preferred visualization layer

The core insight is simple: if your agent's telemetry lives in BigQuery, you get the entire Google Cloud AI and analytics stack for free. And with Conversational Analytics, you don't even need SQL to use it.

---

## References

- **BigQuery Agent Analytics Plugin** — [Official documentation](https://google.github.io/adk-docs/integrations/bigquery-agent-analytics/)
- **Conversational Analytics** — [Official documentation](https://cloud.google.com/bigquery/docs/conversational-analytics)
- **Companion notebook** — [NY_City_Bike_Agent_Logging.ipynb on GitHub](https://github.com/haiyuan-eng-google/demo_BQ_agent_analytics_plugin_notebook/blob/main/NY_City_Bike_Agent_Logging.ipynb)
- **Agent Development Kit (ADK)** — [ADK documentation](https://google.github.io/adk-docs/)
- **BigQuery ML AI functions** — [AI.GENERATE](https://cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-generate), [AI.CLASSIFY](https://cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-classify), [AI.DETECT_ANOMALIES](https://cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-detect-anomalies)
