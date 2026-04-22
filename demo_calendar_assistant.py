"""Calendar-Assistant demo agent for Medium blog post #1.

Three tools against an in-memory fake store, seeded with a deliberate
name ambiguity (three "Priya" contacts) so that the blog post's
featured trace reproduces: when the user asks to book "a 1:1 with
Priya," the search_contacts tool returns three matches and the
agent asks the user which Priya they meant rather than picking one.
Unambiguous prompts (e.g., "Priya Patel" or "Jordan Lee") exercise
the full search_contacts -> get_calendar_availability -> book_meeting
chain instead, producing the contrasting clean-path trace the post
references.

Run:
    PROJECT_ID=... DATASET_ID=... python demo_calendar_assistant.py

The script works with any valid google-genai auth (ADC + Vertex,
Vertex Express Mode, or Developer API), but the exact traces
captured in the blog post were produced with Vertex AI Express
Mode, which needs both of these environment variables in
addition to the two above:

    GOOGLE_CLOUD_API_KEY=<vertex-api-key>
    GOOGLE_GENAI_USE_VERTEXAI=true

Traces land in the configured BigQuery dataset via
BigQueryAgentAnalyticsPlugin (writes still use ADC, independent
of how the LLM calls authenticate).
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.plugins.bigquery_agent_analytics_plugin import (
    BigQueryAgentAnalyticsPlugin,
)
from google.adk.runners import InMemoryRunner
from google.genai import types


# --- Seeded fake store ---------------------------------------------------

_CONTACTS: dict[str, dict[str, Any]] = {
    "pp-412": {"name": "Priya Patel", "email": "priya.patel@example.com", "team": "Platform"},
    "ps-519": {"name": "Priya Shah", "email": "priya.shah@example.com", "team": "Design"},
    "pv-203": {"name": "Priya Venkat", "email": "priya.venkat@example.com", "team": "Research"},
    "am-777": {"name": "Alex Morgan", "email": "alex.morgan@example.com", "team": "Platform"},
    "jl-088": {"name": "Jordan Lee", "email": "jordan.lee@example.com", "team": "Design"},
}

_BUSY_SLOTS: dict[str, list[str]] = {
    "pp-412": ["2026-04-28T10:00:00Z", "2026-04-28T11:00:00Z"],
    "ps-519": [],
    "pv-203": ["2026-04-28T14:00:00Z"],
    "am-777": [],
    "jl-088": ["2026-04-28T09:00:00Z", "2026-04-28T14:00:00Z"],
}

_BOOKED: list[dict[str, Any]] = []


# --- Tools ----------------------------------------------------------------

def search_contacts(name: str) -> dict[str, Any]:
  """Finds contacts whose name contains the given substring.

  Args:
    name: A substring (case-insensitive) to match against contact names.

  Returns:
    A dict with match_count and matches (list of {contact_id, name, team}).
  """
  needle = name.lower()
  matches = [
      {"contact_id": cid, "name": c["name"], "team": c["team"]}
      for cid, c in _CONTACTS.items()
      if needle in c["name"].lower()
  ]
  return {"match_count": len(matches), "matches": matches}


def get_calendar_availability(contact_id: str, date: str) -> dict[str, Any]:
  """Returns the busy and free slots for a contact on a given date.

  Args:
    contact_id: The contact ID returned by search_contacts.
    date: Date in YYYY-MM-DD format.

  Returns:
    A dict with date, busy (list of ISO timestamps), and free_windows
    (list of {start, end} pairs during the workday).
  """
  if contact_id not in _CONTACTS:
    return {"error": f"unknown contact_id: {contact_id}"}
  busy = [
      ts for ts in _BUSY_SLOTS.get(contact_id, []) if ts.startswith(date)
  ]
  workday_start = f"{date}T09:00:00Z"
  workday_end = f"{date}T17:00:00Z"
  # Simplified free-window computation for demo purposes.
  free = []
  cursor = workday_start
  for b in sorted(busy):
    if cursor < b:
      free.append({"start": cursor, "end": b})
    end_of_busy = (
        datetime.fromisoformat(b.replace("Z", "+00:00")) + timedelta(hours=1)
    ).isoformat().replace("+00:00", "Z")
    cursor = max(cursor, end_of_busy)
  if cursor < workday_end:
    free.append({"start": cursor, "end": workday_end})
  return {"date": date, "busy": busy, "free_windows": free}


def book_meeting(
    contact_id: str, slot_start: str, duration_minutes: int = 30
) -> dict[str, Any]:
  """Books a meeting with the given contact at the given time.

  Args:
    contact_id: The contact ID to book with.
    slot_start: ISO timestamp for the meeting start.
    duration_minutes: Meeting length in minutes (default 30).

  Returns:
    A dict with booking status and details.
  """
  if contact_id not in _CONTACTS:
    return {"error": f"unknown contact_id: {contact_id}"}
  booking = {
      "booking_id": f"mtg-{len(_BOOKED) + 1:04d}",
      "contact_id": contact_id,
      "contact_name": _CONTACTS[contact_id]["name"],
      "start": slot_start,
      "duration_minutes": duration_minutes,
  }
  _BOOKED.append(booking)
  return {"status": "booked", **booking}


# --- Agent + App ---------------------------------------------------------

root_agent = LlmAgent(
    name="calendar_assistant",
    description="Books meetings on the user's calendar with specified contacts.",
    instruction=(
        "You are a calendar assistant. When the user asks to book a meeting, "
        "first use search_contacts to find the person. If search_contacts "
        "returns a single match, proceed with get_calendar_availability and "
        "then book_meeting. If it returns multiple matches, stop and ask the "
        "user which person they meant; do not guess. If it returns zero "
        "matches, tell the user you could not find the contact. Always "
        "confirm bookings by the contact's full name and the meeting time."
    ),
    model="gemini-3-flash-preview",
    tools=[search_contacts, get_calendar_availability, book_meeting],
)


def build_app() -> App:
  project_id = os.environ["PROJECT_ID"]
  dataset_id = os.environ["DATASET_ID"]
  return App(
      name="calendar_assistant_demo",
      root_agent=root_agent,
      plugins=[
          BigQueryAgentAnalyticsPlugin(
              project_id=project_id,
              dataset_id=dataset_id,
          )
      ],
  )


# --- Runner ---------------------------------------------------------------

async def run_one_session(prompt: str, user_id: str = "demo_user") -> str:
  """Runs a single user turn end-to-end and returns the session_id."""
  app = build_app()
  runner = InMemoryRunner(
      app=app,
      app_name="calendar_assistant_demo",
  )
  session = await runner.session_service.create_session(
      user_id=user_id,
      app_name="calendar_assistant_demo",
  )
  print(f"[session {session.id}] user: {prompt}")
  async for event in runner.run_async(
      user_id=user_id,
      session_id=session.id,
      new_message=types.Content(
          role="user", parts=[types.Part.from_text(text=prompt)]
      ),
  ):
    author = getattr(event, "author", "?")
    text = ""
    if event.content and event.content.parts:
      text = " ".join(p.text or "" for p in event.content.parts if hasattr(p, "text"))
    if text:
      print(f"[session {session.id}] {author}: {text[:200]}")
  # Flush plugin buffers before exit.
  for plugin in app.plugins:
    flush = getattr(plugin, "flush", None)
    if flush is not None:
      maybe_coro = flush()
      if asyncio.iscoroutine(maybe_coro):
        await maybe_coro
  return session.id


async def main():
  prompt = "Book me a 1:1 with Priya next Tuesday at 2pm for 30 minutes."
  sid = await run_one_session(prompt)
  print(f"\nSession written: {sid}")


if __name__ == "__main__":
  asyncio.run(main())
