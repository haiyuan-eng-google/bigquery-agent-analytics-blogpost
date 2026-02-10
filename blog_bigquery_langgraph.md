# Building Observable AI Agents: Real-Time Analytics for LangGraph with BigQuery

*How to instrument your LangGraph agents with production-grade observability using the AsyncBigQueryCallbackHandler*

---

When you deploy an AI agent into production, the first question isn't *"does it work?"* — it's *"how do I know when it stops working?"*

LLM-powered agents are probabilistic systems. They make decisions, call tools, and chain together operations in ways that are difficult to predict. Without observability, debugging a multi-step agent failure is like reading a novel with half the pages torn out. You can't improve what you can't measure.

In this post, I'll walk you through building an async LangGraph agent from scratch with full observability powered by Google BigQuery. By the end, you'll have:

- A working multi-tool LangGraph agent powered by Gemini
- Every LLM call, tool execution, and graph transition logged to BigQuery in real time
- SQL queries to analyze latency, token usage, tool errors, and cost
- BigQuery AI capabilities like `AI.GENERATE()` for automated root cause analysis
- Production-ready configuration patterns you can drop into your own project

Let's build it.

---

## The observability gap in LangGraph agents

If you've built LangGraph agents before, you've probably experienced this: your agent works perfectly in development, then fails silently in production. A user reports a wrong answer, but you have no way to reconstruct what happened.

Traditional logging captures inputs and outputs. But LangGraph execution is a *graph* — branching decisions, parallel tool calls, conditional edges. You need to capture the full execution trace with timing data, not just the endpoints.

Consider what happens when your agent handles a single user query:

1. The graph starts and routes to the model node
2. The LLM decides which tools to call (and sometimes calls the wrong ones)
3. Multiple tools execute — possibly in parallel
4. Results flow back to the model node for synthesis
5. The LLM generates a final response

Each of these steps can fail, be slow, produce unexpected results, or consume an unpredictable number of tokens. Without structured observability, you're flying blind.

---

## Why BigQuery for LangGraph agent analytics?

You might wonder: why not use LangSmith, Langfuse, or another observability platform?

Those are great tools. But BigQuery gives LangGraph users something unique:

- **Serverless scale** — no infrastructure to manage, handles millions of events without provisioning
- **SQL analytics** — query your agent's behavior with tools your data team already knows
- **Native JSON support** — store structured event payloads and query them with `JSON_VALUE()` at read time
- **Real-time streaming** — events land via the [Storage Write API](https://cloud.google.com/bigquery/docs/write-api) within seconds
- **BigQuery ML integration** — use Gemini directly on your logs with `AI.GENERATE()` for automated root cause analysis and conversational analytics
- **Unified data platform** — agent logs live alongside your business data in the same warehouse, enabling cross-domain analytics

If you're already on Google Cloud, having your agent's telemetry in the same platform as your application data is a powerful combination. And the `AsyncBigQueryCallbackHandler` uses the same `agent_events_v2` schema as the [BigQuery Agent Analytics plugin](https://google.github.io/adk-docs/integrations/bigquery-agent-analytics/) for Google's Agent Development Kit (ADK), so you get a standardized, well-documented event format from day one.

---

## What you'll build

We'll create an **async travel assistant agent** that can:
1. Check the current time
2. Look up weather for a city
3. Convert currencies
4. Evaluate math expressions
5. Search for flights
6. Generate random numbers

Every operation — from the initial LLM call to each tool invocation — gets logged to BigQuery with full trace correlation, latency measurements, and token tracking.

Here's the architecture:

```
User Query
    │
    ▼
┌──────────────────────────────────────────┐
│        Async LangGraph Agent             │
│                                          │
│  ┌─────────┐   ┌───────┐   ┌─────────┐  │
│  │  Model  │──▶│ Tools │──▶│  Model  │  │
│  │  Node   │   │ Node  │   │  Node   │  │
│  └────┬────┘   └───┬───┘   └────┬────┘  │
│       │            │            │        │
│       ▼            ▼            ▼        │
│  ┌──────────────────────────────────┐    │
│  │ AsyncBigQueryCallbackHandler     │    │
│  │  batching · tracing · latency    │    │
│  └──────────────┬───────────────────┘    │
└─────────────────┼────────────────────────┘
                  │  Storage Write API
                  ▼
     ┌─────────────────────┐
     │    BigQuery Table    │
     │   agent_events_v2   │
     │                     │
     │  SQL Analytics      │
     │  AI.GENERATE()      │
     │  Looker Studio      │
     └─────────────────────┘
```

---

## Prerequisites

Before we start, make sure you have:

1. **A Google Cloud project** with the BigQuery API enabled
2. **A BigQuery dataset** — the handler creates the table automatically, but the dataset must exist
3. **Authentication** — run `gcloud auth application-default login` locally, or ensure your service account has:
   - `roles/bigquery.jobUser` (project level)
   - `roles/bigquery.dataEditor` (table level)

Keep IAM scoped tightly. The callback handler only needs write access to the events table — don't grant broader permissions than necessary.

Install the dependencies:

```bash
pip install "langchain-google-community[bigquery]" langchain-google-genai langgraph
```

---

## Step 1: Define the agent state and tools

Every LangGraph agent needs a state definition and tools. Let's start with the state and six tools for our travel assistant:

```python
import math
import random
from datetime import datetime
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State for the async travel assistant agent."""
    messages: Annotated[list[BaseMessage], add_messages]


@tool
def get_current_time() -> str:
    """Get the current date and time with timezone info."""
    now = datetime.now()
    return (
        f"Current Date/Time:\n"
        f"  Date: {now.strftime('%B %d, %Y')}\n"
        f"  Time: {now.strftime('%I:%M:%S %p')}\n"
        f"  Day: {now.strftime('%A')}"
    )


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # In production, call a real weather API
    weather_data = {
        "tokyo": {"temp": 24, "condition": "Sunny", "humidity": 55},
        "london": {"temp": 14, "condition": "Overcast", "humidity": 85},
        "paris": {"temp": 19, "condition": "Light Rain", "humidity": 78},
    }
    city_lower = city.lower().strip()
    if city_lower in weather_data:
        data = weather_data[city_lower]
        return (
            f"Weather in {city.title()}: {data['temp']}°C, "
            f"{data['condition']}, Humidity: {data['humidity']}%"
        )
    return f"Weather data for '{city}' not available."


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount from one currency to another."""
    rates_to_usd = {
        "USD": 1.0, "EUR": 1.08, "GBP": 1.27,
        "JPY": 0.0067, "CNY": 0.14, "AUD": 0.65,
    }
    from_curr = from_currency.upper().strip()
    to_curr = to_currency.upper().strip()
    if from_curr not in rates_to_usd or to_curr not in rates_to_usd:
        return f"Unknown currency: {from_curr} or {to_curr}"
    usd_amount = amount * rates_to_usd[from_curr]
    result = usd_amount / rates_to_usd[to_curr]
    return f"{amount:,.2f} {from_curr} = {result:,.2f} {to_curr}"


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    try:
        expr = expression.replace("^", "**")
        allowed_names = {
            "sqrt": math.sqrt, "log": math.log,
            "abs": abs, "round": round, "pi": math.pi,
        }
        result = eval(expr, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {e}"


@tool
def get_flight_info(origin: str, destination: str) -> str:
    """Look up available flights between two cities."""
    flights = {
        ("new york", "tokyo"): {"airline": "ANA", "duration": "14h 30m", "price": "$1,250"},
        ("london", "paris"): {"airline": "Air France", "duration": "1h 20m", "price": "$150"},
    }
    key = (origin.lower().strip(), destination.lower().strip())
    if key in flights:
        data = flights[key]
        return f"Flight: {data['airline']}, {data['duration']}, {data['price']}"
    return f"No direct flights found from {origin} to {destination}."


@tool
def generate_random_number(min_val: int, max_val: int) -> str:
    """Generate a random number within a specified range."""
    return f"Random number: {random.randint(min_val, max_val)}"
```

The `@tool` decorator turns regular functions into LangChain-compatible tools with automatic schema generation from type hints and docstrings. Notice how each docstring is a clear, single-sentence description — this is what the LLM reads when deciding which tool to call.

---

## Step 2: Configure the AsyncBigQueryCallbackHandler

This is where observability begins. The `AsyncBigQueryCallbackHandler` intercepts every event in the LangGraph execution lifecycle and streams it to BigQuery using non-blocking I/O.

```python
from langchain_google_community.callbacks.bigquery_callback import (
    AsyncBigQueryCallbackHandler,
    BigQueryLoggerConfig,
)

PROJECT_ID = "your-gcp-project-id"
DATASET_ID = "agent_analytics"

# Configure batching and performance
config = BigQueryLoggerConfig(
    batch_size=1,               # Write events immediately (good for dev)
    batch_flush_interval=0.5,   # Flush partial batches every 0.5s
)

# Initialize the async handler
handler = AsyncBigQueryCallbackHandler(
    project_id=PROJECT_ID,
    dataset_id=DATASET_ID,
    table_id="agent_events_v2",
    config=config,
    graph_name="travel_assistant",  # Enables LangGraph-specific tracking
)
```

A few things to note:

- **`AsyncBigQueryCallbackHandler`** uses `asyncio`-native batching and is safe for concurrent requests — each invocation gets its own trace ID and execution context via `ContextVar`.
- **`graph_name`** is the key parameter that activates LangGraph integration. Without it, you'll still get LLM and tool events, but you'll miss `NODE_STARTING`, `NODE_COMPLETED`, and graph-level tracking.
- **`batch_size=1`** writes every event immediately. Great for development, but for production you'll want `batch_size=50` or higher to reduce API calls.
- The handler automatically creates the `agent_events_v2` table if it doesn't exist, partitioned by date and clustered by `event_type`, `agent`, and `user_id`.

---

## Step 3: Build the async LangGraph agent

Now let's wire everything together into an async LangGraph agent:

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

tools = [
    get_current_time, get_weather, convert_currency,
    calculate, get_flight_info, generate_random_number,
]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    project=PROJECT_ID,
).bind_tools(tools)


async def agent_node(state: AgentState) -> dict:
    """Async agent node that decides what to do."""
    messages = state["messages"]
    response = await llm.ainvoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent", should_continue, {"tools": "tools", END: END}
)
workflow.add_edge("tools", "agent")

agent = workflow.compile()
```

The agent follows a ReAct pattern: the `agent` node calls the LLM which decides whether to invoke tools. If it does, the `tools` node executes them, and control returns to the `agent` node for synthesis. The `should_continue` function routes between them based on whether the LLM returned tool calls.

---

## Step 4: Run with async graph context tracking

Here's the critical part — wrapping the agent invocation in an async `graph_context` to capture the full execution lifecycle:

```python
import asyncio


async def run_query(query: str, session_id: str) -> str:
    """Run a single query with graph context tracking."""
    metadata = {
        "session_id": session_id,
        "user_id": "user-123",
        "agent": "travel_assistant",
    }

    async with handler.graph_context("travel_assistant", metadata=metadata):
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=query)]},
            config={
                "callbacks": [handler],
                "metadata": metadata,
            },
        )

    return result["messages"][-1].content
```

The async `graph_context` context manager does three important things:

1. **Emits a `GRAPH_START` event** when entering — marking the beginning of a graph execution
2. **Tracks total latency** — measuring wall-clock time for the entire graph run
3. **Emits `GRAPH_END` or `GRAPH_ERROR`** on exit — capturing success or failure with timing

Because the handler is async-native, you can run multiple queries concurrently with proper trace isolation:

```python
# Run three queries concurrently — each gets its own trace
tasks = [
    run_query("What's the weather in London?", "session-001"),
    run_query("Convert 500 EUR to GBP", "session-002"),
    run_query("Calculate sqrt(144) + pi", "session-003"),
]
responses = await asyncio.gather(*tasks)
```

Each concurrent invocation maintains its own execution context via `ContextVar`, so events from different queries never mix.

---

## Step 5: Put it all together

Here's the complete, runnable async script:

```python
import asyncio
import os

from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from langchain_google_community.callbacks.bigquery_callback import (
    AsyncBigQueryCallbackHandler,
    BigQueryLoggerConfig,
)

# ... (tools and AgentState from Step 1) ...


async def main():
    project_id = os.environ.get("GCP_PROJECT_ID", "your-project-id")

    # 1. Configure the async callback handler
    config = BigQueryLoggerConfig(batch_size=1, batch_flush_interval=0.5)
    handler = AsyncBigQueryCallbackHandler(
        project_id=project_id,
        dataset_id="agent_analytics",
        table_id="agent_events_v2",
        config=config,
        graph_name="travel_assistant",
    )

    # 2. Create the async agent
    agent = create_async_agent()

    # 3. Run with graph context
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

    print(f"Response: {result['messages'][-1].content}")

    # 4. Always shut down to flush remaining events
    await handler.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
```

Run it:

```bash
export GCP_PROJECT_ID="your-project-id"
python travel_agent.py
```

---

## What gets logged?

After running the agent, your BigQuery table will contain a full execution trace. Here's what a typical run looks like:

| Timestamp | Event Type | Node | Tool | Latency |
|-----------|-----------|------|------|---------|
| 19:57:23 | `GRAPH_START` | — | — | — |
| 19:57:23 | `NODE_STARTING` | agent | — | — |
| 19:57:23 | `LLM_REQUEST` | — | — | — |
| 19:57:28 | `LLM_RESPONSE` | — | — | 4,357ms |
| 19:57:28 | `NODE_COMPLETED` | agent | — | 4,366ms |
| 19:57:28 | `TOOL_STARTING` | — | get_weather | — |
| 19:57:28 | `TOOL_COMPLETED` | — | get_weather | <1ms |
| 19:57:28 | `TOOL_STARTING` | — | convert_currency | — |
| 19:57:28 | `TOOL_COMPLETED` | — | convert_currency | <1ms |
| 19:57:28 | `TOOL_STARTING` | — | get_current_time | — |
| 19:57:28 | `TOOL_COMPLETED` | — | get_current_time | <1ms |
| 19:57:28 | `NODE_STARTING` | agent | — | — |
| 19:57:28 | `LLM_REQUEST` | — | — | — |
| 19:57:33 | `LLM_RESPONSE` | — | — | 5,084ms |
| 19:57:33 | `NODE_COMPLETED` | agent | — | 5,089ms |
| 19:57:33 | `GRAPH_END` | — | — | **9,465ms** |

The data tells a clear story: two LLM calls dominate latency (4.4s + 5.1s), while tool execution is sub-millisecond. If we needed to optimize this agent, the LLM calls are the bottleneck — and we'd know exactly where to focus.

Every event includes:
- **`trace_id`** — correlates all events in a single execution
- **`span_id` / `parent_span_id`** — OpenTelemetry-compatible trace hierarchy
- **`latency_ms`** — total time and component breakdown in JSON
- **`content`** — full request/response payloads for token estimation
- **`attributes`** — LangGraph metadata (node name, execution order, graph name)

### LangGraph-specific event types

The callback handler automatically detects LangGraph execution and emits these events in addition to standard LangChain events:

| Event Type | When It's Emitted |
|---|---|
| `NODE_STARTING` | A LangGraph node begins execution |
| `NODE_COMPLETED` | A LangGraph node completes successfully |
| `NODE_ERROR` | A LangGraph node fails |
| `GRAPH_START` | Graph execution begins (via `graph_context`) |
| `GRAPH_END` | Graph execution completes |
| `GRAPH_ERROR` | Graph execution fails |

This is what sets the BigQuery Callback Handler apart from generic logging — it understands LangGraph's execution model and captures the graph structure, not just individual operations.

---

## Step 6: Analyze your agent with SQL

Now the fun part. With structured events in BigQuery, you can answer questions about your agent's behavior that would be nearly impossible with traditional logging.

### Reconstruct a full execution trace

```sql
SELECT
  timestamp,
  event_type,
  JSON_VALUE(attributes, '$.langgraph.node_name') AS node,
  JSON_VALUE(attributes, '$.tool_name') AS tool,
  JSON_VALUE(latency_ms, '$.total_ms') AS latency_ms,
  status
FROM
  `your-project.agent_analytics.agent_events_v2`
WHERE
  trace_id = 'your-trace-id'
ORDER BY
  timestamp ASC;
```

This gives you the complete step-by-step execution flow — invaluable for debugging why an agent took a wrong turn or called the wrong tool.

### Latency analysis with percentiles

```sql
SELECT
  event_type,
  agent,
  COUNT(*) AS count,
  ROUND(AVG(CAST(JSON_VALUE(latency_ms, '$.total_ms') AS FLOAT64)), 2) AS avg_ms,
  ROUND(APPROX_QUANTILES(
    CAST(JSON_VALUE(latency_ms, '$.total_ms') AS FLOAT64), 100
  )[OFFSET(50)], 2) AS p50_ms,
  ROUND(APPROX_QUANTILES(
    CAST(JSON_VALUE(latency_ms, '$.total_ms') AS FLOAT64), 100
  )[OFFSET(95)], 2) AS p95_ms,
  MAX(CAST(JSON_VALUE(latency_ms, '$.total_ms') AS INT64)) AS max_ms
FROM
  `your-project.agent_analytics.agent_events_v2`
WHERE
  event_type IN ('LLM_RESPONSE', 'TOOL_COMPLETED', 'GRAPH_END')
  AND JSON_VALUE(latency_ms, '$.total_ms') IS NOT NULL
  AND DATE(timestamp) = CURRENT_DATE()
GROUP BY
  event_type, agent
ORDER BY
  avg_ms DESC;
```

P50 and P95 latencies reveal the difference between typical performance and tail latency — critical for SLA monitoring.

### Token usage and cost estimation

You can estimate token consumption and cost directly from your logged content. The `content` JSON field stores the full request and response payloads:

```sql
WITH token_estimates AS (
    SELECT
        agent,
        event_type,
        session_id,
        -- Rough estimate: 1 token ≈ 4 characters
        CEIL(LENGTH(TO_JSON_STRING(content)) / 4) AS estimated_tokens
    FROM `your-project.agent_analytics.agent_events_v2`
    WHERE DATE(timestamp) = CURRENT_DATE()
      AND event_type IN ('LLM_REQUEST', 'LLM_RESPONSE')
)
SELECT
    agent,
    COUNT(DISTINCT session_id) AS sessions,
    SUM(CASE WHEN event_type = 'LLM_REQUEST'
        THEN estimated_tokens ELSE 0 END) AS input_tokens,
    SUM(CASE WHEN event_type = 'LLM_RESPONSE'
        THEN estimated_tokens ELSE 0 END) AS output_tokens,
    SUM(estimated_tokens) AS total_tokens,
    -- Gemini 2.5 Flash: $0.30/1M input, $2.50/1M output
    ROUND(
        SUM(CASE WHEN event_type = 'LLM_REQUEST'
            THEN estimated_tokens ELSE 0 END) * 0.0000003 +
        SUM(CASE WHEN event_type = 'LLM_RESPONSE'
            THEN estimated_tokens ELSE 0 END) * 0.0000025,
    4) AS estimated_cost_usd
FROM token_estimates
GROUP BY agent
ORDER BY total_tokens DESC;
```

This query answers: *"How much is each agent costing me per day?"* — critical for budget planning and identifying agents that are burning through tokens unnecessarily.

### Tool usage and error analysis

Understanding which tools your agent uses most — and which ones fail — is critical for optimization. High failure rates on a specific tool might indicate a tool description problem, a schema mismatch, or an upstream API issue:

```sql
SELECT
  JSON_VALUE(attributes, '$.tool_name') AS tool_name,
  agent,
  COUNT(*) AS total_calls,
  COUNTIF(event_type = 'TOOL_COMPLETED') AS successes,
  COUNTIF(event_type = 'TOOL_ERROR') AS failures,
  ROUND(
    COUNTIF(event_type = 'TOOL_ERROR') * 100.0 /
    NULLIF(COUNT(*), 0), 2
  ) AS error_rate_pct,
  ROUND(AVG(
    IF(event_type = 'TOOL_COMPLETED',
       CAST(JSON_VALUE(latency_ms, '$.total_ms') AS FLOAT64),
       NULL)
  ), 0) AS avg_latency_ms,
  MAX(CAST(JSON_VALUE(latency_ms, '$.total_ms') AS FLOAT64)) AS max_latency_ms
FROM
  `your-project.agent_analytics.agent_events_v2`
WHERE
  event_type IN ('TOOL_COMPLETED', 'TOOL_ERROR')
  AND DATE(timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
GROUP BY
  tool_name, agent
ORDER BY
  total_calls DESC;
```

### Error rate tracking over time

Track error trends to catch regressions early:

```sql
SELECT
  DATE(timestamp) AS day,
  agent,
  COUNTIF(status = 'OK') AS successes,
  COUNTIF(status = 'ERROR') AS errors,
  ROUND(
    COUNTIF(status = 'ERROR') * 100.0 / COUNT(*), 2
  ) AS error_rate_pct
FROM
  `your-project.agent_analytics.agent_events_v2`
WHERE
  event_type IN ('GRAPH_END', 'GRAPH_ERROR')
  AND DATE(timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
GROUP BY
  day, agent
ORDER BY
  day DESC;
```

### Session-level execution metrics

Aggregate events into session-level metrics to understand end-to-end agent performance:

```sql
WITH graph_sessions AS (
    SELECT
        session_id,
        agent,
        user_id,
        MIN(timestamp) AS start_time,
        COUNTIF(event_type = 'LLM_REQUEST') AS llm_calls,
        COUNTIF(event_type = 'TOOL_STARTING') AS tool_calls,
        MAX(CASE WHEN event_type = 'GRAPH_END'
            THEN CAST(JSON_VALUE(latency_ms, '$.total_ms') AS INT64)
            END) AS total_latency_ms,
        COUNTIF(status = 'ERROR') AS errors
    FROM `your-project.agent_analytics.agent_events_v2`
    WHERE DATE(timestamp) = CURRENT_DATE()
    GROUP BY session_id, agent, user_id
)
SELECT
    session_id,
    agent,
    user_id,
    llm_calls,
    tool_calls,
    total_latency_ms,
    errors,
    CASE WHEN errors > 0 THEN 'Failed' ELSE 'Success' END AS status
FROM graph_sessions
ORDER BY start_time DESC
LIMIT 20;
```

### User engagement analytics

Understand how users interact with your agents:

```sql
SELECT
    user_id,
    COUNT(DISTINCT session_id) AS total_sessions,
    COUNT(DISTINCT agent) AS agents_used,
    COUNTIF(event_type = 'LLM_REQUEST') AS total_queries,
    COUNTIF(event_type = 'TOOL_STARTING') AS tool_interactions,
    ROUND(AVG(CAST(
        JSON_VALUE(latency_ms, '$.total_ms') AS FLOAT64
    )), 0) AS avg_response_time_ms,
    COUNTIF(status = 'ERROR') AS errors_encountered
FROM `your-project.agent_analytics.agent_events_v2`
WHERE DATE(timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
  AND user_id IS NOT NULL
GROUP BY user_id
ORDER BY total_sessions DESC;
```

### Measure graph-level performance by agent

If you run multiple LangGraph agents in production, compare their performance side by side:

```sql
SELECT
  agent,
  COUNT(*) AS total_runs,
  ROUND(AVG(CAST(
    JSON_VALUE(latency_ms, '$.total_ms') AS FLOAT64
  )), 0) AS avg_graph_latency_ms,
  ROUND(APPROX_QUANTILES(CAST(
    JSON_VALUE(latency_ms, '$.total_ms') AS FLOAT64
  ), 100)[OFFSET(95)], 0) AS p95_graph_latency_ms,
  COUNTIF(status = 'ERROR') AS error_count,
  ROUND(
    COUNTIF(status = 'ERROR') * 100.0 / NULLIF(COUNT(*), 0), 2
  ) AS error_rate_pct
FROM
  `your-project.agent_analytics.agent_events_v2`
WHERE
  event_type IN ('GRAPH_END', 'GRAPH_ERROR')
  AND DATE(timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
GROUP BY
  agent
ORDER BY
  total_runs DESC;
```

---

## Leverage BigQuery AI for root cause analysis

Here's where BigQuery goes beyond traditional analytics. Instead of manually inspecting error logs, you can use BigQuery's built-in AI capabilities to automatically analyze failures.

### Automated root cause analysis with AI.GENERATE()

Use Gemini directly on your agent logs to explain failures — no separate tool required:

```sql
SELECT
  session_id,
  error_message,
  AI.GENERATE(
    (
      'Analyze this agent execution log and identify the root cause of the failure. '
      'Suggest specific fixes:\n\n',
      full_trace
    ),
    connection_id => 'your-project.us.bqml_connection',
    endpoint => 'gemini-2.5-flash'
  ).result AS root_cause_analysis
FROM (
    SELECT
        session_id,
        MAX(error_message) AS error_message,
        STRING_AGG(
            CONCAT(
                CAST(timestamp AS STRING), ' | ',
                event_type, ' | ',
                COALESCE(JSON_VALUE(attributes, '$.tool_name'), ''), ' | ',
                COALESCE(status, ''), ' | ',
                SUBSTR(TO_JSON_STRING(content), 1, 500)
            ),
            '\n' ORDER BY timestamp
        ) AS full_trace
    FROM `your-project.agent_analytics.agent_events_v2`
    WHERE DATE(timestamp) = CURRENT_DATE()
      AND session_id IN (
          SELECT DISTINCT session_id
          FROM `your-project.agent_analytics.agent_events_v2`
          WHERE status = 'ERROR' AND DATE(timestamp) = CURRENT_DATE()
      )
    GROUP BY session_id
)
LIMIT 5;
```

This turns raw error logs into actionable insights: *"The agent failed because the `get_weather` tool was called with an unsupported city name 'LA' — the tool only recognizes full city names like 'Los Angeles'. Update the tool description to clarify expected input format."*

### Conversational analytics

BigQuery's [Conversational Analytics](https://cloud.google.com/bigquery/docs/conversational-analytics) lets you ask questions about your agent logs in natural language directly from the BigQuery console:

> "Show me the error rates by agent for the past week"
> "Which tools are the slowest?"
> "Compare the latency of travel_assistant vs finance_assistant"

No SQL required — the platform generates and executes queries from your questions.

### Tool usage heatmap across agents

If you run multiple agents, visualize which tools each agent uses most:

```sql
SELECT
  agent,
  JSON_VALUE(attributes, '$.tool_name') AS tool_name,
  COUNT(*) AS call_count,
  COUNTIF(status = 'ERROR') AS error_count
FROM `your-project.agent_analytics.agent_events_v2`
WHERE event_type IN ('TOOL_COMPLETED', 'TOOL_ERROR')
  AND DATE(timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
  AND JSON_VALUE(attributes, '$.tool_name') IS NOT NULL
GROUP BY agent, tool_name
ORDER BY agent, call_count DESC;
```

Feed this into a Looker Studio heatmap for a quick visual of your tool landscape.

---

## Production configuration patterns

### Development: see everything immediately

```python
config = BigQueryLoggerConfig(
    batch_size=1,
    batch_flush_interval=0.5,
    max_content_length=10000,
)
```

### Production: optimize for throughput and cost

Not every event is worth logging. Be deliberate about what you capture to control costs and reduce noise:

```python
config = BigQueryLoggerConfig(
    batch_size=50,               # Batch 50 events per write
    batch_flush_interval=5.0,    # Flush at least every 5 seconds
    queue_max_size=50000,        # Large buffer for traffic spikes
    shutdown_timeout=30.0,       # Extra time to drain on shutdown
    event_allowlist=[            # Only capture what matters
        "LLM_RESPONSE",
        "LLM_ERROR",
        "TOOL_COMPLETED",
        "TOOL_ERROR",
        "GRAPH_END",
        "GRAPH_ERROR",
    ],
)
```

Or if you want most events but need to exclude the noisy ones:

```python
config = BigQueryLoggerConfig(
    event_denylist=["CHAIN_START", "CHAIN_END"],
)
```

### Multimodal agents: offload to GCS

If your agent processes images, PDFs, or generates long outputs, configure GCS offloading. Large content goes to Google Cloud Storage while structured references stay in BigQuery:

```python
config = BigQueryLoggerConfig(
    gcs_bucket_name="my-agent-logs",
    connection_id="us.my-bq-connection",
    max_content_length=500 * 1024,  # 500 KB inline, larger goes to GCS
    log_multi_modal_content=True,
)
```

You can then query the offloaded content directly from BigQuery:

```sql
SELECT
  timestamp,
  part.mime_type,
  STRING(OBJ.GET_ACCESS_URL(part.object_ref, 'r').access_urls.read_url) AS signed_url
FROM `your-project.agent_analytics.agent_events_v2`,
UNNEST(content_parts) AS part
WHERE part.storage_mode = 'GCS_REFERENCE'
ORDER BY timestamp DESC
LIMIT 10;
```

---

## Production table setup

While the handler auto-creates the table, for production you should create it explicitly with the recommended DDL:

```sql
CREATE TABLE `your-project.agent_analytics.agent_events_v2`
(
  timestamp TIMESTAMP NOT NULL,
  event_type STRING,
  agent STRING,
  session_id STRING,
  invocation_id STRING,
  user_id STRING,
  trace_id STRING,
  span_id STRING,
  parent_span_id STRING,
  content JSON,
  content_parts ARRAY<STRUCT<
    mime_type STRING,
    uri STRING,
    object_ref STRUCT<
      uri STRING, version STRING,
      authorizer STRING, details JSON
    >,
    text STRING,
    part_index INT64,
    part_attributes STRING,
    storage_mode STRING
  >>,
  attributes JSON,
  latency_ms JSON,
  status STRING,
  error_message STRING,
  is_truncated BOOLEAN
)
PARTITION BY DATE(timestamp)
CLUSTER BY event_type, agent, user_id;
```

Partition pruning by date and clustering by event type gives you fast filtered queries — critical when you have millions of events.

---

## Always call `shutdown()`

The handler uses an asyncio task to batch-write events. If your application exits without calling `await handler.shutdown()`, you may lose the final batch. In web applications, hook this into your shutdown lifecycle:

```python
# FastAPI example
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app):
    yield
    await handler.shutdown()

app = FastAPI(lifespan=lifespan)
```

---

## Interactive analytics notebook

For a deeper dive into your agent data, check out the [LangGraph Agent Analytics Jupyter notebook](https://github.com/langchain-ai/langchain-google/blob/main/libs/community/examples/bigquery_callback/langgraph_agent_analytics.ipynb) that provides a complete six-phase analysis framework:

1. **Real-time observability** — event streams, volume by type, agent activity
2. **Performance analytics** — tool usage heatmaps, latency distributions, cost estimation
3. **Error analysis & debugging** — error summaries, detailed error context, error rate tracking
4. **Session & conversation analysis** — timeline reconstruction, conversation flow extraction
5. **User analytics** — engagement metrics, agent preference matrices
6. **Time-series analysis** — hourly activity patterns, latency trends

The notebook runs directly in [BigQuery Studio](https://cloud.google.com/bigquery/docs/bigquery-studio-introduction), [Colab](https://colab.research.google.com/), or [Vertex AI Workbench](https://cloud.google.com/vertex-ai/docs/workbench/introduction) — no local setup required.

---

## Real-time monitoring dashboard

For production monitoring, the BigQuery callback examples include a [FastAPI-based analytics dashboard](https://github.com/langchain-ai/langchain-google/tree/main/libs/community/examples/bigquery_callback/webapp) with 25+ REST endpoints and real-time Server-Sent Events (SSE):

- Summary stats (total events, sessions, users, error rates)
- Event streaming with 2-second poll interval
- Tool usage analytics and heatmaps
- Latency tracking with trends
- Error tracking and recent error details
- Session timeline and conversation reconstruction
- User engagement metrics

---

## The build-evaluate-observe cycle

Observability isn't a one-time setup — it's the foundation of continuous improvement. Here's how it fits into the LangGraph agent development lifecycle:

1. **Build** your agent with tools and graph logic
2. **Observe** every execution in production via BigQuery — trace flows, latency, tool usage, token costs, errors
3. **Evaluate** by querying your logs — which tools fail most? Where is latency hiding? Are certain user queries triggering bad paths? Which agents cost the most?
4. **Optimize** based on data — improve tool descriptions, adjust system prompts, tune model parameters, add guardrails
5. **Observe again** — verify your changes actually improved things

The SQL queries you write to analyze production logs are the same queries that feed back into your evaluation pipeline. The AI-powered analysis (`AI.GENERATE()`) surfaces root causes automatically. BigQuery becomes both your monitoring system and your data source for continuous agent improvement.

---

## What's next?

Once you have events flowing to BigQuery, the possibilities expand:

- **Connect Looker Studio** for real-time dashboards — there's a [pre-built template](https://lookerstudio.google.com/c/reporting/f1c5b513-3095-44f8-90a2-54953d41b125/page/8YdhF) you can connect to your table
- **Use BigQuery Conversational Analytics** to ask questions about your logs in natural language — "show me error rates by agent this week"
- **Run the [FastAPI monitoring dashboard](https://github.com/langchain-ai/langchain-google/tree/main/libs/community/examples/bigquery_callback/webapp)** for real-time event streaming with Server-Sent Events
- **Explore the [analytics notebook](https://github.com/langchain-ai/langchain-google/blob/main/libs/community/examples/bigquery_callback/langgraph_agent_analytics.ipynb)** for a complete six-phase analysis framework with visualizations
- **Set up alerts** with Cloud Monitoring on error rate spikes
- **Try the [ADK BigQuery Agent Analytics Codelab](https://codelabs.developers.google.com/adk-bigquery-agent-analytics-plugin)** for a hands-on tutorial with a multi-agent retail assistant

The BigQuery Callback Handler transforms your LangGraph agents from black boxes into fully observable systems. Every decision, every tool call, every token, every millisecond of latency — captured, queryable, and ready for analysis.

Your agents are only as reliable as your ability to understand what they're doing. Start logging.

---

*The `AsyncBigQueryCallbackHandler` is available in [langchain-google-community](https://github.com/langchain-ai/langchain-google/tree/main/libs/community). For the complete API reference and more examples, see the [official documentation](https://docs.langchain.com/oss/python/integrations/callbacks/google_bigquery).*

### Further reading

- [BigQuery Agent Analytics Plugin](https://google.github.io/adk-docs/integrations/bigquery-agent-analytics/) — Official documentation for the shared `agent_events_v2` schema
- [ADK BigQuery Agent Analytics Codelab](https://codelabs.developers.google.com/adk-bigquery-agent-analytics-plugin) — Hands-on tutorial with a multi-agent retail assistant
- [BigQuery Storage Write API](https://cloud.google.com/bigquery/docs/write-api) — How the handler streams events to BigQuery
- [BigQuery AI.GENERATE()](https://cloud.google.com/bigquery/docs/ai-generate) — Use Gemini models directly on your BigQuery data
- [BigQuery Conversational Analytics](https://cloud.google.com/bigquery/docs/conversational-analytics) — Ask questions about your data in natural language
- [LangGraph Documentation](https://www.langchain.com/langgraph) — LangGraph framework for building agent workflows
- [BigQuery Callback Examples](https://github.com/langchain-ai/langchain-google/tree/main/libs/community/examples/bigquery_callback) — Full example suite including async, filtering, analytics notebook, and monitoring dashboard
