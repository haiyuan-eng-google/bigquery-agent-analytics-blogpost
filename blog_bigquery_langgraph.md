# Building Observable AI Agents: Real-Time Analytics for LangGraph with BigQuery

*How to instrument your LangGraph agents with production-grade observability using the AsyncBigQueryCallbackHandler*

---

When you deploy an AI agent into production, the first question isn't *"does it work?"* — it's *"how do I know when it stops working?"*

LLM-powered agents are probabilistic systems. They make decisions, call tools, and chain together operations in ways that are difficult to predict. Without observability, debugging a multi-step agent failure is like reading a novel with half the pages torn out.

In this post, I'll walk you through building an async LangGraph agent with full observability powered by Google BigQuery. By the end, you'll have:

- A working multi-tool agent with every event logged to BigQuery in real time
- SQL queries to analyze latency, token usage, tool errors, and cost
- BigQuery `AI.GENERATE()` for automated root cause analysis
- Production-ready configuration patterns

Let's build it.

---

## Why BigQuery?

You might wonder: why not LangSmith, Langfuse, or another observability platform?

Those are great tools. But BigQuery gives you something unique:

- **Serverless scale** — millions of events, no infrastructure to manage
- **SQL analytics** — query agent behavior with tools your data team already knows
- **Real-time streaming** — events land via the [Storage Write API](https://cloud.google.com/bigquery/docs/write-api) within seconds
- **AI on your logs** — use `AI.GENERATE()` to analyze failures with Gemini directly in BigQuery
- **Unified platform** — agent logs live alongside your business data

The `AsyncBigQueryCallbackHandler` uses the same `agent_events_v2` schema as the [ADK BigQuery Agent Analytics plugin](https://google.github.io/adk-docs/integrations/bigquery-agent-analytics/), giving you a standardized event format from day one.

---

## Prerequisites

1. A **Google Cloud project** with BigQuery API enabled
2. A **BigQuery dataset** (the handler auto-creates the table)
3. **Authentication**: `gcloud auth application-default login`

```bash
pip install "langchain-google-community[bigquery]" langchain-google-genai langgraph
```

---

## Step 1: Define tools

We'll build a travel assistant with three tools. Clear docstrings help the LLM choose the right tool:

```python
from datetime import datetime
from langchain_core.tools import tool


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    now = datetime.now()
    return f"Current time: {now.strftime('%I:%M:%S %p')} on {now.strftime('%B %d, %Y')}"


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    weather_data = {
        "tokyo": {"temp": 24, "condition": "Sunny", "humidity": 55},
        "london": {"temp": 14, "condition": "Overcast", "humidity": 85},
    }
    city_lower = city.lower().strip()
    if city_lower in weather_data:
        data = weather_data[city_lower]
        return f"Weather in {city.title()}: {data['temp']}°C, {data['condition']}"
    return f"Weather data for '{city}' not available."


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount from one currency to another."""
    rates = {"USD": 1.0, "EUR": 1.08, "GBP": 1.27, "JPY": 0.0067}
    from_curr, to_curr = from_currency.upper(), to_currency.upper()
    if from_curr not in rates or to_curr not in rates:
        return "Unknown currency"
    result = amount * rates[from_curr] / rates[to_curr]
    return f"{amount:,.2f} {from_curr} = {result:,.2f} {to_curr}"
```

---

## Step 2: Configure the async callback handler

The `AsyncBigQueryCallbackHandler` intercepts every event in the LangGraph execution lifecycle and streams it to BigQuery:

```python
from langchain_google_community.callbacks.bigquery_callback import (
    AsyncBigQueryCallbackHandler,
    BigQueryLoggerConfig,
)

config = BigQueryLoggerConfig(
    batch_size=1,               # Write immediately (dev); use 50+ in production
    batch_flush_interval=0.5,
)

handler = AsyncBigQueryCallbackHandler(
    project_id="your-project-id",
    dataset_id="agent_analytics",
    table_id="agent_events_v2",
    config=config,
    graph_name="travel_assistant",  # Enables LangGraph-specific event tracking
)
```

Key points:
- **`graph_name`** activates LangGraph integration — without it, you miss `NODE_STARTING`, `NODE_COMPLETED`, and graph-level tracking.
- The handler is `asyncio`-native and safe for concurrent requests — each invocation gets its own trace via `ContextVar`.
- The table is auto-created with date partitioning and clustering by `event_type`, `agent`, and `user_id`.

---

## Step 3: Build the async agent

Wire everything into an async LangGraph agent with a ReAct pattern:

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


tools = [get_current_time, get_weather, convert_currency]
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", project="your-project-id"
).bind_tools(tools)


async def agent_node(state: AgentState) -> dict:
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if isinstance(last, AIMessage) and last.tool_calls else END


workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")
agent = workflow.compile()
```

---

## Step 4: Run with graph context tracking

Wrap the invocation in `graph_context` to capture the full execution lifecycle:

```python
import asyncio

metadata = {
    "session_id": "session-001",
    "user_id": "user-123",
    "agent": "travel_assistant",
}

async with handler.graph_context("travel_assistant", metadata=metadata):
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="What's the weather in Tokyo?")]},
        config={"callbacks": [handler], "metadata": metadata},
    )

print(result["messages"][-1].content)
await handler.shutdown()  # Always shut down to flush remaining events
```

The `graph_context` emits `GRAPH_START` on entry and `GRAPH_END` (or `GRAPH_ERROR`) on exit, with total latency measured. You can also run queries concurrently — each gets its own trace:

```python
tasks = [
    run_query("Weather in London?", "session-001"),
    run_query("Convert 500 EUR to GBP", "session-002"),
]
responses = await asyncio.gather(*tasks)
```

---

## What gets logged?

After running the agent, BigQuery contains a full execution trace. Here's real output from our travel assistant querying weather, currency, flights, and time simultaneously:

| Timestamp (UTC) | Event Type | Node | Tool | Latency | Est. Tokens |
|-----------|-----------|------|------|---------|-------------|
| 21:43:38 | `GRAPH_START` | — | — | — | 25 |
| 21:43:38 | `NODE_STARTING` | agent | — | — | 148 |
| 21:43:38 | `LLM_REQUEST` | — | — | — | 120 |
| 21:43:48 | `LLM_RESPONSE` | — | — | 10,505ms | 7 |
| 21:43:48 | `NODE_COMPLETED` | agent | — | 10,508ms | 1,013 |
| 21:43:48 | `NODE_STARTING` | tools | — | — | 1,146 |
| 21:43:48 | `TOOL_STARTING` | — | get_current_time | — | 24 |
| 21:43:48 | `TOOL_STARTING` | — | get_weather | — | 30 |
| 21:43:48 | `TOOL_STARTING` | — | convert_currency | — | 55 |
| 21:43:48 | `TOOL_STARTING` | — | get_flight_info | — | 49 |
| 21:43:48 | `NODE_COMPLETED` | tools | — | 4ms | 397 |
| 21:43:48 | `NODE_STARTING` | agent | — | — | 1,612 |
| 21:43:48 | `LLM_REQUEST` | — | — | — | 1,028 |
| 21:43:49 | `LLM_RESPONSE` | — | — | 906ms | 167 |
| 21:43:49 | `NODE_COMPLETED` | agent | — | 908ms | 353 |
| 21:43:49 | `GRAPH_END` | — | — | **13,059ms** | 25 |

*Real data from a live run against Gemini 2.5 Flash on February 10, 2026. Token estimates use the `CEIL(LENGTH(TO_JSON_STRING(content)) / 4)` approximation.*

The trace tells a clear story: the first LLM call (tool selection) took 10.5s and consumed ~120 input tokens — the latency bottleneck. All four tools executed in parallel in 4ms. The second LLM call (synthesis) consumed ~1,028 input tokens (including tool results) but completed in only 906ms, producing ~167 output tokens. Total graph execution: 13.1s.

Notice how token usage grows through the graph: the first `LLM_REQUEST` sends 120 tokens (just the user query), but the second sends 1,028 tokens (user query + all tool results). This is typical of ReAct agents — context accumulates with each cycle. Monitoring this growth helps you catch runaway token consumption before it hits your budget.

Every event includes `trace_id` for correlation, `span_id`/`parent_span_id` for OpenTelemetry-compatible tracing, `latency_ms`, `content` (for token estimation), and `attributes` (node name, tool name, graph name).

Across a day of testing (11 sessions, 24 LLM calls, 23 tool invocations), the handler logged 141 events with an average graph latency of **9,987ms** (p95: 45,359ms) and an estimated token usage of **~10,716 input tokens** at **$0.0032** for the `travel_assistant` agent.

---

## Analyze your agent with SQL

With structured events in BigQuery, you can answer questions that would be impossible with traditional logging.

### Reconstruct a full execution trace

```sql
SELECT timestamp, event_type,
  JSON_VALUE(attributes, '$.tool_name') AS tool,
  JSON_VALUE(latency_ms, '$.total_ms') AS latency_ms, status
FROM `your-project.agent_analytics.agent_events_v2`
WHERE trace_id = 'your-trace-id'
ORDER BY timestamp;
```

### Token usage and cost estimation

Estimate per-agent costs directly from logged content:

```sql
WITH token_estimates AS (
    SELECT agent, event_type,
        CEIL(LENGTH(TO_JSON_STRING(content)) / 4) AS estimated_tokens
    FROM `your-project.agent_analytics.agent_events_v2`
    WHERE DATE(timestamp) = CURRENT_DATE()
      AND event_type IN ('LLM_REQUEST', 'LLM_RESPONSE')
)
SELECT agent,
    SUM(IF(event_type='LLM_REQUEST', estimated_tokens, 0)) AS input_tokens,
    SUM(IF(event_type='LLM_RESPONSE', estimated_tokens, 0)) AS output_tokens,
    -- Gemini 2.5 Flash: $0.30/1M input, $2.50/1M output
    ROUND(SUM(IF(event_type='LLM_REQUEST', estimated_tokens, 0)) * 0.0000003 +
          SUM(IF(event_type='LLM_RESPONSE', estimated_tokens, 0)) * 0.0000025, 4) AS cost_usd
FROM token_estimates
GROUP BY agent ORDER BY cost_usd DESC;
```

### Tool usage and error analysis

Find which tools fail most — high error rates may indicate tool description problems or upstream API issues:

```sql
SELECT
  JSON_VALUE(attributes, '$.tool_name') AS tool_name,
  COUNT(*) AS total_calls,
  COUNTIF(event_type = 'TOOL_COMPLETED') AS successes,
  COUNTIF(event_type = 'TOOL_ERROR') AS failures,
  ROUND(COUNTIF(event_type = 'TOOL_ERROR') * 100.0 / NULLIF(COUNT(*), 0), 2) AS error_rate_pct,
  ROUND(AVG(IF(event_type = 'TOOL_COMPLETED',
    CAST(JSON_VALUE(latency_ms, '$.total_ms') AS FLOAT64), NULL)), 0) AS avg_latency_ms
FROM `your-project.agent_analytics.agent_events_v2`
WHERE event_type IN ('TOOL_COMPLETED', 'TOOL_ERROR')
  AND DATE(timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
GROUP BY tool_name ORDER BY total_calls DESC;
```

### Latency percentiles

```sql
SELECT event_type, agent,
  ROUND(AVG(CAST(JSON_VALUE(latency_ms, '$.total_ms') AS FLOAT64)), 0) AS avg_ms,
  ROUND(APPROX_QUANTILES(CAST(JSON_VALUE(latency_ms, '$.total_ms') AS FLOAT64), 100)[OFFSET(95)], 0) AS p95_ms
FROM `your-project.agent_analytics.agent_events_v2`
WHERE event_type IN ('LLM_RESPONSE', 'TOOL_COMPLETED', 'GRAPH_END')
  AND JSON_VALUE(latency_ms, '$.total_ms') IS NOT NULL AND DATE(timestamp) = CURRENT_DATE()
GROUP BY event_type, agent ORDER BY avg_ms DESC;
```

---

## BigQuery AI for root cause analysis

Here's where BigQuery goes beyond traditional analytics. Use `AI.GENERATE()` to have Gemini analyze failures directly on your logs:

```sql
SELECT session_id,
  AI.GENERATE(
    ('Analyze this agent execution log. Identify the root cause of failure and suggest fixes:\n\n',
     full_trace),
    connection_id => 'your-project.us.bqml_connection',
    endpoint => 'gemini-2.5-flash'
  ).result AS root_cause_analysis
FROM (
    SELECT session_id,
      STRING_AGG(
        CONCAT(CAST(timestamp AS STRING), ' | ', event_type, ' | ',
          COALESCE(JSON_VALUE(attributes, '$.tool_name'), ''), ' | ',
          COALESCE(status, ''), ' | ', SUBSTR(TO_JSON_STRING(content), 1, 500)),
        '\n' ORDER BY timestamp
      ) AS full_trace
    FROM `your-project.agent_analytics.agent_events_v2`
    WHERE DATE(timestamp) = CURRENT_DATE()
      AND session_id IN (
        SELECT DISTINCT session_id FROM `your-project.agent_analytics.agent_events_v2`
        WHERE status = 'ERROR' AND DATE(timestamp) = CURRENT_DATE())
    GROUP BY session_id
) LIMIT 5;
```

This turns raw logs into actionable insights: *"The agent failed because `get_weather` was called with 'LA' — the tool only recognizes full city names like 'Los Angeles'. Update the tool description to clarify expected input format."*

You can also use [BigQuery Conversational Analytics](https://cloud.google.com/bigquery/docs/conversational-analytics) to ask questions in natural language — *"Show me error rates by agent this week"* — no SQL required.

---

## Production configuration

**Development** — see everything immediately:

```python
config = BigQueryLoggerConfig(batch_size=1, batch_flush_interval=0.5)
```

**Production** — optimize for throughput and cost:

```python
config = BigQueryLoggerConfig(
    batch_size=50,
    batch_flush_interval=5.0,
    queue_max_size=50000,
    shutdown_timeout=30.0,
    event_allowlist=[  # Only capture what matters
        "LLM_RESPONSE", "LLM_ERROR",
        "TOOL_COMPLETED", "TOOL_ERROR",
        "GRAPH_END", "GRAPH_ERROR",
    ],
)
```

**Multimodal agents** — offload large content to GCS:

```python
config = BigQueryLoggerConfig(
    gcs_bucket_name="my-agent-logs",
    connection_id="us.my-bq-connection",
    max_content_length=500 * 1024,
    log_multi_modal_content=True,
)
```

---

## What's next?

Once events are flowing to BigQuery:

- **[Looker Studio dashboards](https://lookerstudio.google.com/c/reporting/f1c5b513-3095-44f8-90a2-54953d41b125/page/8YdhF)** — connect the pre-built template to your table
- **[Analytics notebook](https://github.com/langchain-ai/langchain-google/blob/main/libs/community/examples/bigquery_callback/langgraph_agent_analytics.ipynb)** — six-phase analysis framework (latency, errors, sessions, users, time-series) that runs in BigQuery Studio, Colab, or Vertex AI Workbench
- **[FastAPI monitoring dashboard](https://github.com/langchain-ai/langchain-google/tree/main/libs/community/examples/bigquery_callback/webapp)** — real-time event streaming with 25+ REST endpoints
- **[ADK BigQuery Codelab](https://codelabs.developers.google.com/adk-bigquery-agent-analytics-plugin)** — hands-on tutorial with a multi-agent retail assistant
- **Cloud Monitoring alerts** on error rate spikes or latency degradation

The BigQuery Callback Handler transforms your LangGraph agents from black boxes into fully observable systems. Every decision, every tool call, every token — captured, queryable, and ready for analysis.

Start logging.

---

*The `AsyncBigQueryCallbackHandler` is available in [langchain-google-community](https://github.com/langchain-ai/langchain-google/tree/main/libs/community). For the complete API reference, see the [official documentation](https://docs.langchain.com/oss/python/integrations/callbacks/google_bigquery). Full example suite: [bigquery_callback examples](https://github.com/langchain-ai/langchain-google/tree/main/libs/community/examples/bigquery_callback).*
