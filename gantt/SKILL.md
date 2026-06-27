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
python scripts/gantt.py --config ci_before.json --x-range 35   # pin the x-axis width
```

Output is saved to the path specified in `"output"`. Default: `gantt.png`.

`--x-range SPEC` pins the x-axis range, **overriding** any `x_range` in the config (CLI wins).
- Minutes mode: `--x-range 35` → `0–35 min`; `--x-range 5,40` → `5–40 min`.
- Days mode: `--x-range 90` → 90 days from the first start; `--x-range 2026-01-01,2026-06-01` → explicit dates.

---

## Config Schema

```json
{
  "title":           "string — chart title (top-left, large)",
  "subtitle":        "string — shown below title; completion range appended automatically",
  "theme":           "dark | light  (default: dark)",
  "time_unit":       "days | minutes  (default: days)",
  "output":          "path/to/output.png",
  "figsize":         [width, height],
  "dpi":             150,
  "x_range":         "fix the x-axis width (see below); overridden by --x-range",
  "sort_by_start":   "true (default) orders rows by start; false keeps task order",
  "show_kickoff":    true,
  "show_delivery":   true,
  "delivery_label_y": 0.4,
  "font_scale":      1.0,
  "label_font_scale": 1.0,
  "show_legend":     true,
  "category_colors": { "Category Name": "#hex" },
  "tasks": [ ... ]
}
```

### Fonts

All text scales from a single fixed hierarchy multiplied by `font_scale`:

```
title  >  subtitle  >  axis labels ("Task" / "Elapsed time")  >  axis ticks · task labels · markers · legend
  18         14                 12                                              10
```

- `font_scale` (default `1.0`) scales **every** text element, preserving the hierarchy.
- `label_font_scale` (default `1.0`) is an extra multiplier on the task (y-tick) labels only.
- `show_legend` (default `true`) — set `false` to hide the legend.

### `time_unit`

| Value | Starts / ends | Durations | Axis |
|---|---|---|---|
| `"days"` (default) | `"YYYY-MM-DD"` strings | working days (weekends skipped) | calendar dates |
| `"minutes"` | numbers (minutes from `0`) | minutes (no weekend logic) | elapsed time, e.g. `0 min … 35 min` |

Use `"minutes"` for short-lived pipelines (CI runs, batch jobs) and any before/after
runtime comparison. In minutes mode `start`, `end`, `duration`, and `duration_range`
are plain numbers, and `start_offset` takes a `minutes` key instead of `days`.

### `x_range` (fixed axis width)

Pins the x-axis instead of auto-fitting to the data. Essential when two charts must be
drawn to the **same scale** (see *Before/After comparison* below).

| `time_unit` | Form | Meaning |
|---|---|---|
| `minutes` | `35` | `0–35 min` |
| `minutes` | `[5, 40]` | `5–40 min` |
| `days` | `90` | 90 days from the earliest start |
| `days` | `["2026-01-01", "2026-06-01"]` | explicit date range |

The `--x-range` CLI flag overrides `x_range` in the config, so the same two configs can be
re-rendered at any shared width without editing them.

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
| `"start": "YYYY-MM-DD"` | Explicit start date (a number in minutes mode, e.g. `"start": 0`) |
| `"depends_on": "Task A"` | Start after Task A ends. Also accepts a list: `["Task A", "Task B"]` |
| `"start_offset": {"from": "Task A", "days": 7}` | Start N days after Task A's *start* (parallel work). In minutes mode use a `minutes` key |

### Duration source — pick one

| Field | Description |
|---|---|
| `"duration": 14` | Fixed length in working days (minutes in minutes mode) |
| `"duration_range": [7, 14]` | Uncertain length: solid bar for min, faded+striped extension for max. Use `0` as min for optional tasks |
| `"end": "YYYY-MM-DD"` | Explicit end date (a number in minutes mode) |

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

## Before/After comparison

To compare two timelines side by side — e.g. a CI pipeline before and after an
optimisation, where the same components run **sequentially before** but **in parallel
after** — render two separate charts that share scale and colour. Three rules make the
pair directly comparable:

1. **Same width, same scale.** Give both configs the same `figsize`, and pin the same
   `x_range` (or pass the same `--x-range` to both renders). Set it to the *slower* of the
   two runtimes, rounded up, so both charts use one ruler. A bar that is twice as long
   genuinely took twice as long.

   ```bash
   python scripts/gantt.py --config ci_before.json --x-range 35
   python scripts/gantt.py --config ci_after.json  --x-range 35
   ```

2. **Same colour per component — always pin `category_colors`.** Make each component its
   own `category`, and define an **identical** `category_colors` map in both configs.
   This is mandatory, not cosmetic: without it, colours fall back to the auto-palette,
   which is assigned in *first-seen order*. When work is reordered or parallelised, that
   order changes between the two charts, so the same component would get a different
   colour — defeating the comparison.

   ```json
   "category_colors": { "lint": "#92CDDD", "dbt": "#C59849", "dagster": "#A53F2B", "deploy": "#27AE60" }
   ```

3. **Express the structural difference through dependencies, not colour.** Keep the task
   names and categories identical across both files; only change the `depends_on` wiring.

   ```json
   // before — dbt then dagster run sequentially
   { "name": "dbt",     "category": "dbt",     "depends_on": "lint", "duration": 15 },
   { "name": "dagster", "category": "dagster", "depends_on": "dbt",  "duration": 11 }

   // after — dbt and dagster run in parallel, deploy waits for both
   { "name": "dbt",     "category": "dbt",     "depends_on": "lint", "duration": 5 },
   { "name": "dagster", "category": "dagster", "depends_on": "lint", "duration": 5 },
   { "name": "deploy",  "category": "deploy",  "depends_on": ["dbt", "dagster"], "duration": 4 }
   ```

The reader then sees the same colours in the same legend on both charts, on one shared
time ruler — so the only visual differences are the ones that actually changed: bar
lengths and how they stack.

---

## Examples

See `example_configs/` for ready-to-run configs:

| File | Description |
|---|---|
| `roadmap_2025.json` | 10-task phased product roadmap |
| `sprint_kickoff.json` | 5-task sprint spanning weeks (demonstrates auto date-axis scaling) |
| `ci_before.json` / `ci_after.json` | Minutes-mode before/after pair; render both with `--x-range 35` for a like-for-like comparison |

---

## Visual conventions

- **Solid bar** — expected (min) duration
- **Faded + hatched extension** — uncertain additional time (max − min)
- **Kick-off line** — first task start; full-height dashed line, label at top in data coords
- **Delivery line** — last task end; line capped at label height (does not overshoot). Label height set by `delivery_label_y` (0–1, default 0.4)
- Categories separated by subtle horizontal rules
- Legend top-right, frameless
