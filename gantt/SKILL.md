---
name: gantt-chart
description: Generate a Gantt chart PNG from a JSON config. Handles task dependencies,
  duration uncertainty (min/max ranges), category colours, and vertical markers
  (Kick-off, Delivery). Run whenever a user needs a project timeline visualisation.
---

# Gantt Chart Generator

Generates a styled Gantt chart PNG from a JSON config file or inline JSON string.

## Invocation

```bash
python scripts/gantt.py --config example_configs/roadmap_2025.json
python scripts/gantt.py --json '{"title": "...", "tasks": [...]}'
```

Output is saved to the path specified in `"output"`. Default: `gantt.png`.

---

## Config Schema

```json
{
  "title":           "string — chart title (top-left, large)",
  "subtitle":        "string — shown below title; completion range appended automatically",
  "theme":           "dark | light  (default: dark)",
  "output":          "path/to/output.png",
  "figsize":         [width, height],
  "dpi":             150,
  "show_kickoff":    true,
  "show_delivery":   true,
  "category_colors": { "Category Name": "#hex" },
  "tasks": [ ... ]
}
```

---

## Task Fields

Every task requires a **name**, a **start source** (one or more), and a **duration source** (one).

### Name

```json
{ "name": "Task A" }
```

### Start source — pick one or combine (latest date wins)

| Field | Description |
|---|---|
| `"start": "YYYY-MM-DD"` | Explicit start date |
| `"depends_on": "Task A"` | Start after Task A ends. Also accepts a list: `["Task A", "Task B"]` |
| `"start_offset": {"from": "Task A", "days": 7}` | Start N days after Task A's *start* (parallel work) |

### Duration source — pick one

| Field | Description |
|---|---|
| `"duration": 14` | Fixed length in days |
| `"duration_range": [7, 14]` | Uncertain length: solid bar for min, faded+striped extension for max. Use `0` as min for optional tasks |
| `"end": "YYYY-MM-DD"` | Explicit end date |

### Finish constraint (optional)

```json
{ "finish_requires": "Task B" }
```

Task cannot finish before Task B finishes (finish-to-finish). Also accepts a list.

### Category (optional)

```json
{ "category": "Design & Build" }
```

Groups tasks by colour. The following category names have built-in default colours — use them for automatic consistent styling:

| Category | Colour |
|---|---|
| `Discovery` | `#31859D` (teal) |
| `Design & Build` | `#C59849` (gold) |
| `Client` | `#A53F2B` (chestnut) — waiting for sign-off, approval, feedback |
| `Validation` | `#27AE60` (green) — internal QA, verification |
| `Implementation` | `#27AE60` (green) |

Any other category name is assigned a palette colour in first-seen order. All defaults can be overridden via `category_colors` in the config.

---

## Subtitle auto-completion

The subtitle is automatically appended with the computed delivery range:

```
Project Timeline - Expected delivery between 29 Jun 2026 (7 weeks) and 27 Jul 2026 (11 weeks)
```

- **Earliest** date: computed using minimum durations throughout the full dependency chain.
- **Latest** date: computed using maximum durations throughout the full dependency chain.

---

## Examples

See `example_configs/` for ready-to-run configs:

| File | Description |
|---|---|
| `roadmap_2025.json` | 10-task phased product roadmap |
| `sprint_kickoff.json` | 5-task sprint spanning weeks (demonstrates auto date-axis scaling) |

---

## Visual conventions

- **Solid bar** — expected (min) duration
- **Faded + hatched extension** — uncertain additional time (max − min)
- **Kick-off line** — first task start; full-height dashed line, label at top in data coords
- **Delivery line** — last task end; line capped at label height (does not overshoot)
- Categories separated by subtle horizontal rules
- Legend top-right, frameless
