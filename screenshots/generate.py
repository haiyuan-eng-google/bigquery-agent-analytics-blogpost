"""Generate PNG screenshots for the Medium blog post.

Produces four images using real data captured from the
Calendar-Assistant demo run:

- 01_hook_raw_agent_events.png  — BQ console-style result grid,
  hero/hook image for section 1.
- 02_doctor_output.png          — Terminal-style rendering of
  `bq-agent-sdk doctor` output, for section 3.
- 03_trace_render.png           — Terminal-style rendering of
  the featured trace (session 84ef108d), for section 4.
- 04_information_schema.png     — BQ console-style result grid
  for the SDK cost-by-feature query, for section 6.

Run:
    python generate.py

All four PNGs land in the same directory. Upload directly to
Medium's image widget at each section's screenshot placeholder.
"""

from __future__ import annotations

import os
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ #
# Palettes                                                            #
# ------------------------------------------------------------------ #

TERM_BG = (30, 30, 30)            # VS Code dark-like background
TERM_TEXT = (212, 212, 212)       # soft off-white
TERM_GREEN = (106, 180, 114)      # success checkmarks
TERM_YELLOW = (221, 188, 103)
TERM_BLUE = (97, 175, 239)
TERM_COMMENT = (128, 128, 128)
TERM_RED = (224, 108, 117)

BQ_BG = (255, 255, 255)
BQ_HEADER_BG = (243, 244, 246)
BQ_BORDER = (229, 231, 235)
BQ_TEXT = (32, 33, 36)
BQ_SUBTEXT = (95, 99, 104)
BQ_CHROME_BG = (245, 246, 247)
BQ_ACCENT = (26, 115, 232)

# ------------------------------------------------------------------ #
# Fonts                                                               #
# ------------------------------------------------------------------ #


def _load_font(candidates: list[tuple[str, int]]) -> ImageFont.FreeTypeFont:
  for path, size in candidates:
    try:
      return ImageFont.truetype(path, size)
    except (OSError, IOError):
      continue
  return ImageFont.load_default()


MONO = _load_font([
    ("/System/Library/Fonts/Menlo.ttc", 16),
    ("/System/Library/Fonts/Monaco.ttf", 16),
    ("/System/Library/Fonts/Courier.ttc", 16),
])
MONO_BOLD = _load_font([
    ("/System/Library/Fonts/Menlo.ttc", 17),
    ("/System/Library/Fonts/Monaco.ttf", 17),
])
SANS = _load_font([
    ("/Library/Fonts/Arial.ttf", 16),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", 16),
    ("/System/Library/Fonts/Helvetica.ttc", 16),
])
SANS_SMALL = _load_font([
    ("/Library/Fonts/Arial.ttf", 13),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", 13),
])
SANS_BOLD = _load_font([
    ("/Library/Fonts/Arial Bold.ttf", 16),
    ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 16),
])

LINE_H = 22


# ------------------------------------------------------------------ #
# Terminal-style renderer                                             #
# ------------------------------------------------------------------ #


def render_terminal(
    filename: str,
    title: str,
    lines: list[tuple[str, tuple[int, int, int]] | str],
    width: int = 1100,
    padding: int = 24,
    header_height: int = 36,
) -> None:
  content_lines = len(lines)
  height = header_height + padding * 2 + content_lines * LINE_H + padding
  img = Image.new("RGB", (width, height), TERM_BG)
  draw = ImageDraw.Draw(img)

  # Window chrome: three dots + title
  draw.rectangle([0, 0, width, header_height], fill=(45, 45, 45))
  cx = 18
  for color in [(255, 95, 86), (255, 189, 46), (40, 201, 64)]:
    draw.ellipse([cx, 12, cx + 13, 25], fill=color)
    cx += 20
  draw.text(
      (width // 2 - 100, 9), title, fill=(180, 180, 180), font=SANS_SMALL
  )

  y = header_height + padding
  for item in lines:
    if isinstance(item, tuple):
      text, color = item
    else:
      text, color = item, TERM_TEXT
    draw.text((padding, y), text, fill=color, font=MONO)
    y += LINE_H

  img.save(os.path.join(HERE, filename), "PNG")
  print(f"wrote {filename}  ({width}x{height})")


# ------------------------------------------------------------------ #
# BQ-console-style table renderer                                     #
# ------------------------------------------------------------------ #


def render_bq_table(
    filename: str,
    query_snippet: str,
    headers: list[str],
    rows: list[list[str]],
    col_widths: list[int],
    width: int | None = None,
    caption: str | None = None,
) -> None:
  if width is None:
    width = sum(col_widths) + 40
  row_h = 34
  header_h = 40
  query_h = 90
  caption_h = 30 if caption else 0
  chrome_h = 40
  padding = 20
  height = (
      chrome_h + padding + query_h + padding + header_h + row_h * len(rows)
      + caption_h + padding
  )
  img = Image.new("RGB", (width, height), BQ_BG)
  draw = ImageDraw.Draw(img)

  # Browser-ish chrome
  draw.rectangle([0, 0, width, chrome_h], fill=BQ_CHROME_BG)
  cx = 18
  for color in [(255, 95, 86), (255, 189, 46), (40, 201, 64)]:
    draw.ellipse([cx, 14, cx + 13, 27], fill=color)
    cx += 20
  draw.rectangle(
      [cx + 20, 8, width - 18, 32],
      fill=(255, 255, 255),
      outline=(220, 221, 222),
  )
  draw.text(
      (cx + 32, 13),
      "console.cloud.google.com/bigquery",
      fill=BQ_SUBTEXT,
      font=SANS_SMALL,
  )

  # Query block
  qy = chrome_h + padding
  draw.rectangle(
      [padding, qy, width - padding, qy + query_h],
      fill=(248, 249, 250),
      outline=BQ_BORDER,
  )
  draw.text(
      (padding + 8, qy + 6),
      "Query editor",
      fill=BQ_ACCENT,
      font=SANS_BOLD,
  )
  for i, qline in enumerate(query_snippet.splitlines()[:3]):
    draw.text(
        (padding + 8, qy + 28 + i * 18),
        qline,
        fill=BQ_TEXT,
        font=MONO,
    )

  # Results table
  ty = qy + query_h + padding
  x = padding
  # Header row
  draw.rectangle(
      [padding, ty, width - padding, ty + header_h], fill=BQ_HEADER_BG
  )
  for i, h in enumerate(headers):
    draw.text((x + 12, ty + 12), h, fill=BQ_TEXT, font=SANS_BOLD)
    x += col_widths[i]
  draw.line(
      [(padding, ty + header_h), (width - padding, ty + header_h)],
      fill=BQ_BORDER,
      width=1,
  )
  # Data rows
  ry = ty + header_h
  for r, row in enumerate(rows):
    bg = BQ_BG if r % 2 == 0 else (250, 251, 252)
    draw.rectangle([padding, ry, width - padding, ry + row_h], fill=bg)
    x = padding
    for i, cell in enumerate(row):
      cell_text = cell
      # Truncate long content to first N chars.
      max_chars = max(8, col_widths[i] // 8)
      if len(cell_text) > max_chars:
        cell_text = cell_text[: max_chars - 1] + "…"
      draw.text((x + 12, ry + 9), cell_text, fill=BQ_TEXT, font=SANS_SMALL)
      x += col_widths[i]
    draw.line(
        [(padding, ry + row_h), (width - padding, ry + row_h)],
        fill=BQ_BORDER,
        width=1,
    )
    ry += row_h
  # Column dividers
  x = padding
  for w in col_widths[:-1]:
    x += w
    draw.line(
        [(x, ty), (x, ry)], fill=BQ_BORDER, width=1,
    )
  # Caption
  if caption:
    draw.text(
        (padding, ry + 10), caption, fill=BQ_SUBTEXT, font=SANS_SMALL,
    )

  img.save(os.path.join(HERE, filename), "PNG")
  print(f"wrote {filename}  ({width}x{height})")


# ------------------------------------------------------------------ #
# Shot 01 — messy agent_events                                        #
# ------------------------------------------------------------------ #


def shot_01_hook():
  query = (
      "SELECT timestamp, event_type, agent, content, status, error_message\n"
      "FROM `test-project-0728-467323.agent_analytics_demo.agent_events`\n"
      "WHERE session_id = '84ef108d-745c-451a-ae79-d0f97673268d' ORDER BY timestamp;"
  )
  headers = ["timestamp", "event_type", "agent", "content", "status"]
  rows = [
      [
          "2026-04-22 21:31:10", "USER_MESSAGE_RECEIVED", "calendar_assistant",
          "{\"text\":\"Book me a 1:1 with Priya next Tuesday at 2pm for 30 minutes.\",\"role\":\"user\"}",
          "OK",
      ],
      [
          "2026-04-22 21:31:10", "INVOCATION_STARTING", "calendar_assistant",
          "{}", "OK",
      ],
      [
          "2026-04-22 21:31:10", "AGENT_STARTING", "calendar_assistant",
          "{\"raw\":\"You are a calendar assistant. When the user asks to book a meeting, first use search_contacts...\"}",
          "OK",
      ],
      [
          "2026-04-22 21:31:11", "LLM_REQUEST", "calendar_assistant",
          "{\"model\":\"gemini-3-flash-preview\",\"parts\":[{\"text\":\"Book me a 1:1 with Priya next Tuesday at 2pm...\"}]}",
          "OK",
      ],
      [
          "2026-04-22 21:31:13", "LLM_RESPONSE", "calendar_assistant",
          "{\"response\":\"call: search_contacts\",\"function_call\":{\"name\":\"search_contacts\",\"args\":{\"name\":\"Priya\"}}}",
          "OK",
      ],
      [
          "2026-04-22 21:31:13", "TOOL_STARTING", "calendar_assistant",
          "{\"tool\":\"search_contacts\",\"args\":{\"name\":\"Priya\"}}",
          "OK",
      ],
      [
          "2026-04-22 21:31:13", "TOOL_COMPLETED", "calendar_assistant",
          "{\"tool\":\"search_contacts\",\"result\":{\"match_count\":3,\"matches\":[{\"contact_id\":\"pp-412\",\"name\":\"Priya Patel\"},{\"contact_id\":\"ps-519\",\"name\":\"Priya Shah\"},{\"contact_id\":\"pv-203\",\"name\":\"Priya Venkat\"}]}}",
          "OK",
      ],
      [
          "2026-04-22 21:31:13", "LLM_REQUEST", "calendar_assistant",
          "{\"model\":\"gemini-3-flash-preview\",\"parts\":[{\"text\":\"Book me a 1:1 with Priya next Tuesday...\"}]}",
          "OK",
      ],
      [
          "2026-04-22 21:31:15", "LLM_RESPONSE", "calendar_assistant",
          "{\"response\":\"text: 'I found three people named Priya: Priya Patel (Platform), Priya Shah (Design), and Priya Venkat (Research). Which...'\"}",
          "OK",
      ],
      [
          "2026-04-22 21:31:15", "AGENT_COMPLETED", "calendar_assistant",
          "{\"response\":null,\"total_ms\":5234}", "OK",
      ],
      [
          "2026-04-22 21:31:15", "INVOCATION_COMPLETED", "calendar_assistant",
          "{\"total_ms\":19778}", "OK",
      ],
  ]
  render_bq_table(
      filename="01_hook_raw_agent_events.png",
      query_snippet=query,
      headers=headers,
      rows=rows,
      col_widths=[200, 220, 200, 660, 80],
      caption=(
          "11 rows · One session of the Calendar-Assistant demo. "
          "Everything says status=OK; nothing tells you why the agent "
          "asked instead of booking."
      ),
  )


# ------------------------------------------------------------------ #
# Shot 02 — bq-agent-sdk doctor                                       #
# ------------------------------------------------------------------ #


def shot_02_doctor():
  lines: list[tuple[str, tuple[int, int, int]] | str] = [
      ("$ bq-agent-sdk doctor", TERM_BLUE),
      "",
      ("bigquery-agent-analytics doctor — pre-flight checks", TERM_COMMENT),
      "",
      ("  [✓] Application Default Credentials loaded", TERM_GREEN),
      "       principal: raincoatrun@gmail.com",
      ("  [✓] Project accessible: test-project-0728-467323", TERM_GREEN),
      ("  [✓] Dataset accessible: agent_analytics_demo (location: US)", TERM_GREEN),
      ("  [✓] Table agent_events present · 11 rows found in last 24h", TERM_GREEN),
      ("  [✓] Required columns match schema (event_type, span_id, parent_span_id, content, ...)", TERM_GREEN),
      ("  [✓] BigQuery permissions: bigquery.jobs.create ✓  bigquery.tables.getData ✓", TERM_GREEN),
      ("  [✓] Python SDK version: bigquery-agent-analytics 0.2.0", TERM_GREEN),
      "",
      ("All checks passed. You're ready.", TERM_YELLOW),
      "",
      ("$ _", TERM_BLUE),
  ]
  render_terminal(
      filename="02_doctor_output.png",
      title="bq-agent-sdk doctor  —  zsh  —  80×24",
      lines=lines,
  )


# ------------------------------------------------------------------ #
# Shot 03 — trace.render() terminal                                   #
# ------------------------------------------------------------------ #


def shot_03_render():
  lines: list[tuple[str, tuple[int, int, int]] | str] = [
      (">>> trace = client.get_session_trace(\"84ef108d-dadb-...-268d\")", TERM_BLUE),
      (">>> trace.render()", TERM_BLUE),
      "",
      "Trace: e-ae1fe18c-887e-4df3-a91f-25eec174e2dd | Session: 84ef108d-745c-451a-ae79-d0f97673268d | 5235ms",
      "======================================================================================================",
      "└─ [✓] USER_MESSAGE_RECEIVED [calendar_assistant] - Book me a 1:1 with Priya next Tuesday at 2pm for 30 min…",
      "└─ [✓] INVOCATION_STARTING [calendar_assistant]",
      "└─ [✓] INVOCATION_COMPLETED [calendar_assistant] (19778ms)",
      "   ├─ [✓] AGENT_STARTING [calendar_assistant] - You are a calendar assistant. When the user asks to book a…",
      "   └─ [✓] AGENT_COMPLETED [calendar_assistant] (5234ms)",
      "      ├─ [✓] LLM_REQUEST [calendar_assistant] (gemini-3-flash-preview) - Book me a 1:1 with Priya next Tues…",
      "      ├─ [✓] LLM_RESPONSE [calendar_assistant] (2972ms) - call: search_contacts",
      "      ├─ [✓] TOOL_STARTING [calendar_assistant] (search_contacts)",
      "      ├─ [✓] TOOL_COMPLETED [calendar_assistant] (search_contacts) (0ms)",
      "      ├─ [✓] LLM_REQUEST [calendar_assistant] (gemini-3-flash-preview) - Book me a 1:1 with Priya next Tues…",
      "      └─ [✓] LLM_RESPONSE [calendar_assistant] (2175ms) - I found three people named Priya: Priya Patel (Pl…",
      "",
      (">>> _", TERM_BLUE),
  ]
  # Color the [✓] marks green post-hoc via a tweaked renderer: easiest
  # is to pre-split the line and draw segments. Keep it simple by
  # rendering lines as-is; the tree shape still reads at a glance.
  render_terminal(
      filename="03_trace_render.png",
      title="Python  —  trace.render()  —  zsh  —  120×24",
      lines=lines,
      width=1300,
  )


# ------------------------------------------------------------------ #
# Shot 04 — INFORMATION_SCHEMA cost query                             #
# ------------------------------------------------------------------ #


def shot_04_information_schema():
  query = (
      "SELECT (SELECT value FROM UNNEST(labels) WHERE key='sdk_feature') AS sdk_feature,\n"
      "       COUNT(*) AS runs, ROUND(SUM(total_bytes_processed)/POW(1024,3),3) AS gb_processed,\n"
      "       ROUND(AVG(total_slot_ms),0) AS avg_slot_ms FROM `region-us`.INFORMATION_SCHEMA.JOBS_BY_PROJECT ..."
  )
  headers = ["sdk_feature", "runs", "gb_processed", "avg_slot_ms"]
  rows = [
      ["trace-read", "6", "0.029", "1742"],
  ]
  render_bq_table(
      filename="04_information_schema.png",
      query_snippet=query,
      headers=headers,
      rows=rows,
      col_widths=[200, 100, 180, 160],
      caption=(
          "SDK-labeled jobs, last 1 hour · 6 trace-read calls = every "
          "`get_session_trace` and `list_traces` invocation from "
          "sections 4 and 5. 29 MB processed, 1.7s average slot time."
      ),
  )


# ------------------------------------------------------------------ #
# Driver                                                              #
# ------------------------------------------------------------------ #


def shot_05_hero():
  """Closing/hero: 'rows → tree' concept image."""
  width, height = 1400, 720
  img = Image.new("RGB", (width, height), (247, 249, 252))
  draw = ImageDraw.Draw(img)

  # Left panel: stacked rows (representing the messy BQ table)
  lx, ly = 80, 160
  row_w, row_h = 440, 28
  gap = 6
  for i in range(12):
    color = (220, 224, 230) if i % 2 == 0 else (232, 236, 241)
    draw.rectangle(
        [lx, ly + i * (row_h + gap),
         lx + row_w, ly + i * (row_h + gap) + row_h],
        fill=color,
    )
    # Simulated JSON-y text inside each row
    draw.rectangle(
        [lx + 12, ly + i * (row_h + gap) + 9,
         lx + 140, ly + i * (row_h + gap) + 19],
        fill=(180, 188, 196),
    )
    draw.rectangle(
        [lx + 160, ly + i * (row_h + gap) + 9,
         lx + row_w - 20, ly + i * (row_h + gap) + 19],
        fill=(155, 165, 175),
    )

  # Label under the left panel
  title_left_font = _load_font([
      ("/Library/Fonts/Arial Bold.ttf", 22),
      ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22),
  ])
  caption_font = _load_font([
      ("/Library/Fonts/Arial.ttf", 14),
      ("/System/Library/Fonts/Supplemental/Arial.ttf", 14),
  ])
  draw.text(
      (lx, 100), "agent_events", fill=(60, 70, 85), font=title_left_font
  )
  draw.text(
      (lx, 128), "11 rows · JSON · timestamps",
      fill=(120, 130, 145), font=caption_font,
  )

  # Middle: big arrow
  ax = lx + row_w + 60
  ay = height // 2
  draw.polygon(
      [
          (ax, ay - 30), (ax + 90, ay - 30), (ax + 90, ay - 60),
          (ax + 160, ay), (ax + 90, ay + 60), (ax + 90, ay + 30),
          (ax, ay + 30),
      ],
      fill=(66, 133, 244),
  )
  # "one line" label above the arrow
  arrow_label_font = _load_font([
      ("/Library/Fonts/Arial Bold.ttf", 18),
      ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 18),
  ])
  draw.text(
      (ax + 10, ay - 100), "trace.render()",
      fill=(66, 133, 244), font=arrow_label_font,
  )
  draw.text(
      (ax + 30, ay + 78), "one line",
      fill=(120, 130, 145), font=caption_font,
  )

  # Right panel: DAG tree
  rx = ax + 200
  ry_top = 160
  # Root
  root_box = (rx + 60, ry_top, rx + 260, ry_top + 40)
  draw.rectangle(root_box, fill=(255, 255, 255), outline=(66, 133, 244), width=2)
  draw.text(
      (rx + 75, ry_top + 10),
      "USER_MESSAGE",
      fill=(60, 70, 85),
      font=caption_font,
  )

  # Children levels
  level2_y = ry_top + 100
  level3_y = level2_y + 90
  level4_y = level3_y + 90

  children_level2 = [
      ("LLM_REQUEST", 40),
      ("TOOL_CALL", 220),
  ]
  for label, dx in children_level2:
    box = (rx + dx, level2_y, rx + dx + 180, level2_y + 40)
    draw.rectangle(box, fill=(255, 255, 255), outline=(155, 165, 175), width=1)
    draw.text(
        (rx + dx + 15, level2_y + 10), label,
        fill=(60, 70, 85), font=caption_font,
    )
    # connector from root to child
    draw.line(
        [(rx + 160, ry_top + 40), (rx + dx + 90, level2_y)],
        fill=(170, 180, 195), width=2,
    )

  # level3: tool result (under TOOL_CALL)
  tool_start_x = rx + 220
  box = (tool_start_x, level3_y, tool_start_x + 220, level3_y + 40)
  draw.rectangle(box, fill=(255, 255, 255), outline=(64, 167, 93), width=2)
  draw.text(
      (tool_start_x + 10, level3_y + 10), "3 matches returned",
      fill=(34, 85, 50), font=caption_font,
  )
  draw.line(
      [(tool_start_x + 90, level2_y + 40), (tool_start_x + 110, level3_y)],
      fill=(170, 180, 195), width=2,
  )

  # level4: decision
  dec_x = tool_start_x + 20
  box = (dec_x, level4_y, dec_x + 260, level4_y + 44)
  draw.rectangle(box, fill=(255, 252, 235), outline=(238, 179, 49), width=2)
  draw.text(
      (dec_x + 10, level4_y + 12), "agent asks: which Priya?",
      fill=(120, 85, 10), font=caption_font,
  )
  draw.line(
      [(tool_start_x + 110, level3_y + 40), (dec_x + 130, level4_y)],
      fill=(170, 180, 195), width=2,
  )

  # Right-panel title
  draw.text(
      (rx + 60, 100), "conversation DAG",
      fill=(60, 70, 85), font=title_left_font,
  )
  draw.text(
      (rx + 60, 128), "parent→child · tools · decisions",
      fill=(120, 130, 145), font=caption_font,
  )

  # Main title at top
  main_title_font = _load_font([
      ("/Library/Fonts/Arial Bold.ttf", 32),
      ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 32),
  ])
  draw.text(
      (width // 2 - 260, 40),
      "Your agent_events table is a graph.",
      fill=(36, 46, 58),
      font=main_title_font,
  )

  img.save(os.path.join(HERE, "05_hero_rows_to_tree.png"), "PNG")
  print(f"wrote 05_hero_rows_to_tree.png  ({width}x{height})")


def main():
  shot_01_hook()
  shot_02_doctor()
  shot_03_render()
  shot_04_information_schema()
  shot_05_hero()


if __name__ == "__main__":
  main()
