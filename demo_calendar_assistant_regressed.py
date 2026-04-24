"""Regressed-branch variant of demo_calendar_assistant.py for blog post #2.

Same three tools and the same seeded contact store as
demo_calendar_assistant.py. The *only* meaningful difference is a
much longer system prompt: we've added a set of verbose few-shot
examples and policy text to the instruction. Locally the agent still
behaves correctly; against real multi-turn production traffic, the
longer instruction gets repeated on every turn and pushes
per-session token usage up by roughly 40% vs. the baseline.

The point of this file is to produce a trace fleet that would trip
the `token_efficiency --threshold=50000` gate demonstrated in the
blog post. Run it the same way as the baseline demo:

    PROJECT_ID=... DATASET_ID=... python demo_calendar_assistant_regressed.py

Everything downstream (the BigQuery Agent Analytics plugin, the
evaluate CLI, the workflow YAML) is identical. This file exists
purely to make the red-CI screenshot reproducible.
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


# --- Seeded fake store (unchanged from baseline) -------------------------

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


# --- Tools (unchanged from baseline) -------------------------------------

def search_contacts(name: str) -> dict[str, Any]:
  """Finds contacts whose name contains the given substring."""
  needle = name.lower()
  matches = [
      {"contact_id": cid, "name": c["name"], "team": c["team"]}
      for cid, c in _CONTACTS.items()
      if needle in c["name"].lower()
  ]
  return {"match_count": len(matches), "matches": matches}


def get_calendar_availability(contact_id: str, date: str) -> dict[str, Any]:
  """Returns the busy and free slots for a contact on a given date."""
  if contact_id not in _CONTACTS:
    return {"error": f"unknown contact_id: {contact_id}"}
  busy = [ts for ts in _BUSY_SLOTS.get(contact_id, []) if ts.startswith(date)]
  workday_start = f"{date}T09:00:00Z"
  workday_end = f"{date}T17:00:00Z"
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
  """Books a meeting with the given contact at the given time."""
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


# --- Regressed instruction -----------------------------------------------
#
# This is the change that drives the ~40% token-usage regression. The
# baseline instruction is two sentences. This one adds ten few-shot
# examples and a policy block. Each LLM_REQUEST repeats the full
# instruction, so multi-turn sessions stack the cost fast.

_REGRESSED_INSTRUCTION = """\
You are a calendar assistant. Your job is to book meetings on the
user's calendar with specified contacts. Follow the policy below
exactly, and refer to the examples whenever a user request is
ambiguous or when you are deciding whether to ask a clarifying
question.

## Policy

1. When the user asks to book a meeting, first use search_contacts to
   find the person. Use the full name if the user provided one; use
   the first name as a substring search if they did not.
2. If search_contacts returns a single match, proceed with
   get_calendar_availability for that contact on the requested date.
   If the requested slot is free, call book_meeting with that slot;
   otherwise propose the nearest free slot and wait for confirmation.
3. If search_contacts returns multiple matches, STOP and ask the user
   which person they meant. Do not guess. Do not proceed with
   get_calendar_availability for any of the matches until the user
   resolves the ambiguity.
4. If search_contacts returns zero matches, tell the user you could
   not find the contact, and ask them to provide more specifics (for
   example, last name, team, or email address).
5. Always confirm successful bookings by stating the contact's full
   name, the meeting date and time in the user's locale, and the
   duration. Never confirm a booking the user has not approved.
6. If the user provides a contact_id directly, you may skip
   search_contacts and proceed to get_calendar_availability.
7. Never invent contact_ids, calendar slots, or team names. Only use
   values returned by the tools.

## Examples

Example 1 — single match, requested slot free:
  User: "Book me 30 min with Alex Morgan on April 28 at 9am."
  Assistant: (search_contacts -> one match: am-777)
             (get_calendar_availability -> 9:00 is free)
             (book_meeting am-777, 2026-04-28T09:00:00Z, 30)
             "Booked 30 minutes with Alex Morgan for Tuesday,
              April 28, 9:00 AM UTC."

Example 2 — single match, requested slot busy, propose alternative:
  User: "Can I get 30 min with Jordan Lee on April 28 at 9am?"
  Assistant: (search_contacts -> one match: jl-088)
             (get_calendar_availability -> 9:00 is busy)
             "Jordan Lee is busy at 9:00 AM on April 28. The next
              free window starts at 10:00 AM; want me to book that?"

Example 3 — multiple matches, ask for disambiguation:
  User: "Book me a 1:1 with Priya next Tuesday at 2pm."
  Assistant: (search_contacts -> three matches)
             "I found three people named Priya: Priya Patel
              (Platform), Priya Shah (Design), and Priya Venkat
              (Research). Which Priya would you like to book?"

Example 4 — zero matches:
  User: "Book me 15 min with Robin Garcia."
  Assistant: (search_contacts -> zero matches)
             "I couldn't find a contact named Robin Garcia. Can you
              share a last name, team, or email address?"

Example 5 — user provides contact_id directly:
  User: "Book 30 min with pp-412 on April 28 at 2pm."
  Assistant: (get_calendar_availability -> 14:00 is free)
             (book_meeting pp-412, 2026-04-28T14:00:00Z, 30)
             "Booked 30 minutes with Priya Patel for Tuesday,
              April 28, 2:00 PM UTC."

Example 6 — user confirms after you proposed an alternative:
  User: "Yes, book the 10 AM slot with Jordan."
  Assistant: (book_meeting jl-088, 2026-04-28T10:00:00Z, 30)
             "Booked 30 minutes with Jordan Lee for Tuesday,
              April 28, 10:00 AM UTC."

Example 7 — ambiguous date ("next Tuesday"):
  User: "Book me 30 min with Alex Morgan next Tuesday at 9am."
  Assistant: "To be sure I have the right day, 'next Tuesday' for me
              resolves to April 28, 2026. Shall I book that?"

Example 8 — duration omitted, default to 30 minutes and confirm:
  User: "Book me time with Priya Patel tomorrow at 2pm."
  Assistant: (search_contacts -> pp-412)
             (get_calendar_availability -> 14:00 is free)
             "Booking 30 minutes with Priya Patel for tomorrow at
              2:00 PM UTC. Confirm?"

Example 9 — partial first name:
  User: "Book me time with Alex."
  Assistant: (search_contacts 'Alex' -> am-777 only)
             "Alex Morgan (Platform) — is that the right Alex? If
              so, what day and time work?"

Example 10 — user cancels mid-flow:
  User: "Actually never mind, cancel that."
  Assistant: "Done — I have not booked anything. Let me know when
              you're ready to try again."

## Output format

Always respond in natural language. Only emit tool calls when the
policy says so. When confirming a booking, restate the contact's
full name, the date in MMMM D format, the time with a timezone, and
the duration.
"""


# --- Agent + App ---------------------------------------------------------

root_agent = LlmAgent(
    name="calendar_assistant",
    description="Books meetings on the user's calendar with specified contacts.",
    instruction=_REGRESSED_INSTRUCTION,
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


# --- Runner --------------------------------------------------------------

async def run_one_session(prompt: str, user_id: str = "demo_user") -> str:
  """Runs a single user turn end-to-end and returns the session_id."""
  app = build_app()
  runner = InMemoryRunner(app=app, app_name="calendar_assistant_demo")
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
      text = " ".join(
          p.text or "" for p in event.content.parts if hasattr(p, "text")
      )
    if text:
      print(f"[session {session.id}] {author}: {text[:200]}")
  for plugin in app.plugins:
    flush = getattr(plugin, "flush", None)
    if flush is not None:
      maybe_coro = flush()
      if asyncio.iscoroutine(maybe_coro):
        await maybe_coro
  return session.id


async def main():
  # Run a small fleet of prompts to produce a representative set of
  # regressed-branch traces for the blog's screenshot capture.
  prompts = [
      "Book me a 1:1 with Priya next Tuesday at 2pm for 30 minutes.",
      "Can I get 30 min with Alex Morgan on April 28 at 9am?",
      "Book me time with Jordan Lee tomorrow.",
      "Find me 30 min with Priya Patel next week.",
      "Schedule 1 hour with Alex Morgan on April 29 at 10am.",
  ]
  for prompt in prompts:
    sid = await run_one_session(prompt)
    print(f"\nSession written: {sid}\n")


if __name__ == "__main__":
  asyncio.run(main())
