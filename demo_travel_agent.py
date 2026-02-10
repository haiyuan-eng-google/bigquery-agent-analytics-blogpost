"""Demo: Async Travel Assistant Agent with BigQuery Observability.

This script creates an async LangGraph agent with 6 tools and logs every event
to BigQuery via the AsyncBigQueryCallbackHandler. It demonstrates:
- Async callback handler with non-blocking event logging
- LangGraph agent with realistic tool calls
- Async graph context manager for GRAPH_START/GRAPH_END tracking
- Concurrent query execution with proper trace isolation
- Full latency, token, and error tracking

Prerequisites:
    1. gcloud auth application-default login
    2. Create dataset: bq mk --dataset YOUR_PROJECT_ID:agent_analytics
    3. pip install langchain-google-community[bigquery] langgraph langchain-google-genai
"""

import asyncio
import math
import os
import random
from datetime import datetime
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_google_community.callbacks.bigquery_callback import (
    AsyncBigQueryCallbackHandler,
    BigQueryLoggerConfig,
)

# Configuration
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "test-project-0728-467323")
DATASET_ID = os.environ.get("BQ_DATASET_ID", "agent_analytics")
TABLE_ID = "agent_events_v2"


class AgentState(TypedDict):
    """State for the async travel assistant agent."""

    messages: Annotated[list[BaseMessage], add_messages]


# --- Tools ---
@tool
def get_current_time() -> str:
    """Get the current date and time with timezone info.

    Returns:
        Current date, time, and day of week.
    """
    now = datetime.now()
    return (
        f"Current Date/Time:\n"
        f"  Date: {now.strftime('%B %d, %Y')}\n"
        f"  Time: {now.strftime('%I:%M:%S %p')}\n"
        f"  Day: {now.strftime('%A')}\n"
        f"  Week: {now.isocalendar()[1]} of {now.year}"
    )


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The city name (e.g., Tokyo, London, Paris).

    Returns:
        Current weather conditions including temperature, humidity, and wind.
    """
    weather_data = {
        "new york": {
            "temp": 22,
            "condition": "Clear",
            "humidity": 60,
            "wind": "15 km/h S",
        },
        "tokyo": {
            "temp": 24,
            "condition": "Sunny",
            "humidity": 55,
            "wind": "8 km/h E",
        },
        "london": {
            "temp": 14,
            "condition": "Overcast",
            "humidity": 85,
            "wind": "18 km/h SW",
        },
        "paris": {
            "temp": 19,
            "condition": "Light Rain",
            "humidity": 78,
            "wind": "10 km/h W",
        },
        "sydney": {
            "temp": 28,
            "condition": "Warm and Clear",
            "humidity": 45,
            "wind": "12 km/h NE",
        },
        "san francisco": {
            "temp": 18,
            "condition": "Partly Cloudy",
            "humidity": 72,
            "wind": "12 km/h NW",
        },
    }
    city_lower = city.lower().strip()
    if city_lower in weather_data:
        data = weather_data[city_lower]
        return (
            f"Weather in {city.title()}:\n"
            f"  Temperature: {data['temp']}°C ({data['temp'] * 9 // 5 + 32}°F)\n"
            f"  Condition: {data['condition']}\n"
            f"  Humidity: {data['humidity']}%\n"
            f"  Wind: {data['wind']}"
        )
    return (
        f"Weather data for '{city}' not available. "
        "Try: New York, Tokyo, London, Paris, Sydney, or San Francisco."
    )


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount from one currency to another.

    Args:
        amount: The amount to convert.
        from_currency: Source currency code (e.g., USD, EUR, JPY).
        to_currency: Target currency code (e.g., USD, EUR, JPY).

    Returns:
        Converted amount with exchange rate.
    """
    rates_to_usd = {
        "USD": 1.0,
        "EUR": 1.08,
        "GBP": 1.27,
        "JPY": 0.0067,
        "CNY": 0.14,
        "CAD": 0.74,
        "AUD": 0.65,
        "CHF": 1.12,
    }
    from_curr = from_currency.upper().strip()
    to_curr = to_currency.upper().strip()
    if from_curr not in rates_to_usd:
        return f"Unknown currency: {from_curr}"
    if to_curr not in rates_to_usd:
        return f"Unknown currency: {to_curr}"
    usd_amount = amount * rates_to_usd[from_curr]
    result = usd_amount / rates_to_usd[to_curr]
    rate = rates_to_usd[from_curr] / rates_to_usd[to_curr]
    return (
        f"{amount:,.2f} {from_curr} = {result:,.2f} {to_curr} "
        f"(rate: 1 {from_curr} = {rate:.4f} {to_curr})"
    )


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Args:
        expression: A mathematical expression (e.g., "15 * 23 + 7", "sqrt(144)").

    Returns:
        The result of the calculation.
    """
    try:
        expr = expression.replace("^", "**").replace("×", "*").replace("÷", "/")
        allowed_names = {
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "abs": abs,
            "round": round,
            "pi": math.pi,
            "e": math.e,
        }
        allowed_chars = set("0123456789+-*/().  sqrtcosintanlogexpabsroundpie")
        if not all(c in allowed_chars for c in expr.lower()):
            return "Error: Expression contains invalid characters"
        result = eval(expr, {"__builtins__": {}}, allowed_names)
        if isinstance(result, float):
            if result == int(result):
                return f"Result: {int(result)}"
            return f"Result: {result:.6f}".rstrip("0").rstrip(".")
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {e}"


@tool
def get_flight_info(origin: str, destination: str) -> str:
    """Look up available flights between two cities.

    Args:
        origin: Departure city.
        destination: Arrival city.

    Returns:
        Available flight information.
    """
    flights = {
        ("new york", "tokyo"): {
            "airline": "ANA",
            "duration": "14h 30m",
            "price": "$1,250",
        },
        ("new york", "london"): {
            "airline": "British Airways",
            "duration": "7h 15m",
            "price": "$680",
        },
        ("london", "paris"): {
            "airline": "Air France",
            "duration": "1h 20m",
            "price": "$150",
        },
        ("tokyo", "sydney"): {
            "airline": "Qantas",
            "duration": "9h 45m",
            "price": "$890",
        },
        ("san francisco", "tokyo"): {
            "airline": "JAL",
            "duration": "11h 10m",
            "price": "$1,100",
        },
    }
    key = (origin.lower().strip(), destination.lower().strip())
    if key in flights:
        data = flights[key]
        return (
            f"Flight from {origin.title()} to {destination.title()}:\n"
            f"  Airline: {data['airline']}\n"
            f"  Duration: {data['duration']}\n"
            f"  Price: {data['price']} (economy)"
        )
    return (
        f"No direct flights found from {origin} to {destination}. "
        "Try popular routes: NYC-Tokyo, NYC-London, London-Paris, Tokyo-Sydney."
    )


@tool
def generate_random_number(min_val: int, max_val: int) -> str:
    """Generate a random number within a specified range.

    Args:
        min_val: Minimum value (inclusive).
        max_val: Maximum value (inclusive).

    Returns:
        A random number within the range.
    """
    if min_val > max_val:
        return (
            f"Error: min_val ({min_val}) cannot be greater "
            f"than max_val ({max_val})"
        )
    result = random.randint(min_val, max_val)
    return f"Random number between {min_val} and {max_val}: {result}"


def create_async_agent() -> StateGraph:
    """Create the async LangGraph ReAct agent.

    Returns:
        Compiled LangGraph agent.
    """
    tools = [
        get_current_time,
        get_weather,
        convert_currency,
        calculate,
        get_flight_info,
        generate_random_number,
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

    return workflow.compile()


async def run_query(
    agent: StateGraph,
    handler: AsyncBigQueryCallbackHandler,
    query: str,
    session_id: str,
    user_id: str = "demo-user",
) -> str:
    """Run a single query asynchronously with graph context tracking.

    Args:
        agent: The compiled agent.
        handler: The async callback handler.
        query: The user query.
        session_id: Session identifier.
        user_id: User identifier.

    Returns:
        The agent's response.
    """
    metadata = {
        "session_id": session_id,
        "user_id": user_id,
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

    final_message = result["messages"][-1]
    return (
        final_message.content
        if isinstance(final_message, AIMessage)
        else str(final_message)
    )


async def main() -> None:
    """Run the async travel assistant demo."""
    print("=" * 60)
    print("Async Travel Assistant with BigQuery Observability")
    print("=" * 60)
    print(f"Project: {PROJECT_ID}")
    print(f"Dataset: {DATASET_ID}.{TABLE_ID}")

    # 1. Configure the async callback handler
    config = BigQueryLoggerConfig(
        batch_size=1,
        batch_flush_interval=0.5,
    )

    handler = AsyncBigQueryCallbackHandler(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        table_id=TABLE_ID,
        config=config,
        graph_name="travel_assistant",
    )

    # 2. Create the async agent
    agent = create_async_agent()

    # --- Test 1: Multi-tool travel query ---
    print("\n" + "=" * 60)
    print("Test 1: Multi-Tool Travel Query")
    print("=" * 60)

    response = await run_query(
        agent,
        handler,
        "I'm planning a trip to Tokyo. What time is it now? "
        "What's the weather like in Tokyo? "
        "How much is 1000 USD in JPY? "
        "Are there flights from San Francisco to Tokyo?",
        session_id="demo-session-001",
    )
    print(f"Response:\n{response}")

    # --- Test 2: Complex calculation ---
    print("\n" + "=" * 60)
    print("Test 2: Calculation Query")
    print("=" * 60)

    response = await run_query(
        agent,
        handler,
        "Calculate the compound interest on $10,000 at 5% for 10 years. "
        "Use the formula: principal * (1 + rate)^years",
        session_id="demo-session-002",
    )
    print(f"Response:\n{response}")

    # --- Test 3: Concurrent queries ---
    print("\n" + "=" * 60)
    print("Test 3: Concurrent Queries")
    print("=" * 60)

    queries = [
        ("What's the weather in London and Paris?", "demo-session-003a"),
        ("How much is 500 EUR in GBP?", "demo-session-003b"),
        (
            "Generate a random number between 1 and 100 and calculate its square root.",
            "demo-session-003c",
        ),
    ]

    print("Running 3 queries concurrently...")
    tasks = [
        run_query(agent, handler, query, session_id)
        for query, session_id in queries
    ]
    responses = await asyncio.gather(*tasks)

    for (query, _), response in zip(queries, responses):
        print(f"\nQuery: {query}")
        print(f"Response: {response[:200]}...")

    # 3. Shut down and flush remaining events
    print("\n" + "=" * 60)
    print("Shutting down handler...")
    await handler.shutdown()
    print("Events flushed to BigQuery successfully.")

    # 4. Print helpful queries
    print(f"""
Done! Query your events in BigQuery:

-- Event overview
SELECT
    timestamp,
    event_type,
    session_id,
    JSON_EXTRACT_SCALAR(attributes, '$.tool_name') as tool_name,
    JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms') as latency_ms,
    status
FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
WHERE DATE(timestamp) = CURRENT_DATE()
  AND agent = 'travel_assistant'
ORDER BY timestamp DESC
LIMIT 50;

-- Token usage & cost estimation
WITH token_estimates AS (
    SELECT
        event_type,
        CEIL(LENGTH(TO_JSON_STRING(content)) / 4) as estimated_tokens
    FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
    WHERE DATE(timestamp) = CURRENT_DATE()
      AND agent = 'travel_assistant'
      AND event_type IN ('LLM_REQUEST', 'LLM_RESPONSE')
)
SELECT
    SUM(CASE WHEN event_type = 'LLM_REQUEST' THEN estimated_tokens ELSE 0 END) as input_tokens,
    SUM(CASE WHEN event_type = 'LLM_RESPONSE' THEN estimated_tokens ELSE 0 END) as output_tokens,
    ROUND(SUM(CASE WHEN event_type = 'LLM_REQUEST' THEN estimated_tokens ELSE 0 END) * 0.0000003 +
          SUM(CASE WHEN event_type = 'LLM_RESPONSE' THEN estimated_tokens ELSE 0 END) * 0.0000025, 4) as estimated_cost_usd
FROM token_estimates;
    """)


if __name__ == "__main__":
    asyncio.run(main())
