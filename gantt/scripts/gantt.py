#!/usr/bin/env python3
"""Gantt chart generator.

Usage:
    python gantt.py --config config.json
    python gantt.py --json '{"title": "...", "tasks": [...]}'
    python gantt.py --config config.json --x-range 35   # pin x-axis width

Two time units are supported via the config key "time_unit":
    "days"    (default) — task starts/ends are calendar dates (YYYY-MM-DD),
                          durations are working days (weekends skipped).
    "minutes"           — task starts/ends are numbers (minutes from 0),
                          durations are minutes (no weekend logic). Useful for
                          short-lived pipelines (e.g. CI runs) and before/after
                          comparisons where a fixed x-axis width matters.
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_DIR = Path(__file__).parent.parent
_BUNDLED_FONT = _DIR / "fonts" / "NotoSansMono_SemiCondensed-SemiBold.ttf"
_SYSTEM_FONT = Path("/usr/share/fonts/noto_sans_mono/NotoSansMono_SemiCondensed-SemiBold.ttf")

# Arbitrary epoch used only in "minutes" mode: minute N maps to EPOCH + N minutes,
# so the rest of the rendering pipeline can keep working in matplotlib date space.
EPOCH = datetime(2000, 1, 1)

PALETTE = ["#31859D", "#C59849", "#A53F2B", "#1E497D", "#92CDDD", "#BFBEBE", "#27AE60", "#9B59B6"]

DEFAULT_CATEGORY_COLORS = {
    "Discovery":      PALETTE[0],
    "Design & Build": PALETTE[1],
    "Client":         PALETTE[2],
    "Data Import":    PALETTE[4],
    "Validation":     PALETTE[6],
    "Implementation": PALETTE[6],
}

THEMES = {
    "dark": {
        "bg": "#0D1B2A",
        "text": "#fefae0",
        "grid": "#1b2631",
        "accent": "#C59849",
        "separator": "#2d4a6a",
    },
    "light": {
        "bg": "#FFFFFF",
        "text": "#1B1B1B",
        "grid": "#E0E0E0",
        "accent": "#C59849",
        "separator": "#CCCCCC",
    },
}


def _load_font() -> None:
    for path in (_BUNDLED_FONT, _SYSTEM_FONT):
        if path.exists():
            fm.fontManager.addfont(str(path))
            prop = fm.FontProperties(fname=str(path))
            plt.rcParams["font.family"] = prop.get_name()
            return


def _parse(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _fmt_date(dt: datetime) -> str:
    return dt.strftime("%-d %b %Y")


def _add_working_days(start: datetime, days: int) -> datetime:
    """Advance `start` by `days` working days, skipping Sat/Sun."""
    current = start
    remaining = days
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:  # Mon–Fri
            remaining -= 1
    return current


def _count_working_days(start: datetime, end: datetime) -> int:
    """Count Mon–Fri days in [start, end)."""
    count = 0
    day = start
    while day < end:
        if day.weekday() < 5:
            count += 1
        day += timedelta(days=1)
    return count


def _friendly_duration(min_days: int, max_days: int) -> str:
    def fmt(d: int) -> str:
        if d < 7:
            return f"{d} day{'s' if d != 1 else ''}"
        if d % 30 == 0 and d >= 30:
            m = d // 30
            return f"{m} month{'s' if m != 1 else ''}"
        w = d / 7
        if w == int(w):
            w = int(w)
            return f"{w} week{'s' if w != 1 else ''}"
        w = round(w)
        return f"~{w} week{'s' if w != 1 else ''}"

    if min_days == max_days:
        return fmt(min_days)
    if max_days < 7:
        return f"{min_days}–{max_days} days"
    if min_days % 30 == 0 and max_days % 30 == 0:
        return f"{min_days // 30}–{max_days // 30} months"
    min_w, max_w = min_days / 7, max_days / 7
    exact = min_w == int(min_w) and max_w == int(max_w)
    prefix = "" if exact else "~"
    return f"{prefix}{round(min_w)}–{round(max_w)} weeks"


def _fmt_minutes_value(m: float) -> str:
    """Render a minute count: '5 min', '1 min', '1.5 min'."""
    if abs(m - round(m)) < 1e-6:
        return f"{int(round(m))} min"
    return f"{m:.1f} min"


def _friendly_minutes(min_amt: float, max_amt: float) -> str:
    if abs(min_amt - max_amt) < 1e-6:
        return _fmt_minutes_value(min_amt)
    return f"{_fmt_minutes_value(min_amt)}–{_fmt_minutes_value(max_amt)}"


def _time_model(time_unit: str) -> dict:
    """Return unit-aware helpers so the resolver/renderer stay unit-agnostic.

    Keys:
      parse(value)            -> datetime    (interpret a start/end spec)
      add_duration(start, d)  -> datetime    (advance by a duration)
      offset_start(start, n)  -> datetime    (advance a start by a raw offset)
      amount(start, end)      -> number      (duration in native units, for labels)
      friendly(lo, hi)        -> str         (human-readable duration)
      offset_key              -> str         (start_offset field name)
    """
    if time_unit == "minutes":
        return {
            "parse":        lambda v: EPOCH + timedelta(minutes=float(v)),
            "add_duration": lambda start, d: start + timedelta(minutes=float(d)),
            "offset_start": lambda start, n: start + timedelta(minutes=float(n)),
            "amount":       lambda start, end: (end - start).total_seconds() / 60.0,
            "friendly":     _friendly_minutes,
            "offset_key":   "minutes",
        }
    return {
        "parse":        _parse,
        "add_duration": lambda start, d: _add_working_days(start, int(d)),
        "offset_start": lambda start, n: start + timedelta(days=int(n)),
        "amount":       lambda start, end: _count_working_days(start, end),
        "friendly":     lambda lo, hi: _friendly_duration(int(lo), int(hi)),
        "offset_key":   "days",
    }


def _resolve_dates(tasks: list[dict], tm: dict, use_min_durations: bool = False) -> list[dict]:
    """Resolve concrete start/min_end/max_end (as datetimes) for all tasks.

    Duration fields (pick one):
      "duration"               — fixed length (working days, or minutes)
      "duration_range": [min, max] — uncertain length; max used for dependency chaining,
                                     min rendered as solid bar, extra as faded+striped extension
      "end"                    — explicit end (date string, or minute number)

    Start fields (one or more; the latest wins):
      "start"                  — explicit start (date string, or minute number)
      "depends_on"             — start after the chained end of named task(s)
      "start_offset"           — start N units after another task's start
                                 e.g. {"from": "Audit", "days": 7}    (days mode)
                                 or   {"from": "Lint",  "minutes": 2} (minutes mode)

    Finish constraint:
      "finish_requires"        — task(s) that must finish before this task can finish

    use_min_durations: if True, duration_range tasks use their minimum for chaining
                       (used to compute the optimistic earliest delivery).
    """
    by_name = {t["name"]: t for t in tasks}
    # stores (start, min_end, max_end) as datetimes
    resolved: dict[str, tuple[datetime, datetime, datetime]] = {}

    def _resolve(name: str, visiting: set) -> tuple[datetime, datetime, datetime]:
        if name in resolved:
            return resolved[name]
        if name in visiting:
            raise ValueError(f"Circular dependency involving '{name}'")
        visiting.add(name)

        task = by_name[name]

        # Collect all start candidates; the latest one wins
        start_candidates: list[datetime] = []

        if "start" in task:
            start_candidates.append(tm["parse"](task["start"]))

        deps = task.get("depends_on", [])
        if isinstance(deps, str):
            deps = [deps]
        # [2] = chained end (max_end in normal mode, min_end-equivalent in optimistic mode)
        dep_ends = [_resolve(d, visiting)[2] for d in deps]
        if dep_ends:
            start_candidates.append(max(dep_ends))

        offset_cfg = task.get("start_offset")
        if offset_cfg:
            ref_start, _, _ = _resolve(offset_cfg["from"], visiting)
            amount = offset_cfg.get(tm["offset_key"], offset_cfg.get("days", offset_cfg.get("minutes", 0)))
            start_candidates.append(tm["offset_start"](ref_start, amount))

        if not start_candidates:
            raise ValueError(f"Task '{name}' needs a start, depends_on, or start_offset")
        start = max(start_candidates)

        # duration / duration_range / end
        if "duration_range" in task:
            min_dur, max_dur = task["duration_range"]
            min_end = tm["add_duration"](start, min_dur)
            # In optimistic mode both ends use min_dur so downstream tasks chain from it
            max_end = tm["add_duration"](start, min_dur if use_min_durations else max_dur)
        elif "duration" in task:
            min_end = max_end = tm["add_duration"](start, task["duration"])
        elif "end" in task:
            min_end = max_end = tm["parse"](task["end"])
        else:
            raise ValueError(f"Task '{name}' needs end, duration, or duration_range")

        # finish_requires → end can't be before required tasks' chained end
        finish_reqs = task.get("finish_requires", [])
        if isinstance(finish_reqs, str):
            finish_reqs = [finish_reqs]
        req_ends = [_resolve(r, visiting)[2] for r in finish_reqs]
        if req_ends:
            gate = max(req_ends)
            min_end = max(min_end, gate)
            max_end = max(max_end, gate)

        visiting.discard(name)
        resolved[name] = (start, min_end, max_end)
        return start, min_end, max_end

    result = []
    for task in tasks:
        start, min_end, max_end = _resolve(task["name"], set())
        result.append({
            **task,
            "_start":   start,
            "_min_end": min_end,
            "_end":     max_end,
        })
    return result


def _resolve_x_range(x_range, time_unit: str, base_start: datetime) -> tuple[float, float]:
    """Turn an x_range spec into explicit (x_min, x_max) in matplotlib date2num units.

    minutes mode:
        35            -> [0 min, 35 min]
        [5, 40]       -> [5 min, 40 min]
    days mode:
        90                          -> [base_start, base_start + 90 days]
        ["2026-01-01","2026-06-01"] -> those two dates
    """
    if time_unit == "minutes":
        if isinstance(x_range, (list, tuple)):
            lo, hi = float(x_range[0]), float(x_range[1])
        else:
            lo, hi = 0.0, float(x_range)
        return (
            float(mdates.date2num(EPOCH + timedelta(minutes=lo))),
            float(mdates.date2num(EPOCH + timedelta(minutes=hi))),
        )
    # days mode
    if isinstance(x_range, (list, tuple)):
        lo = _parse(x_range[0]) if isinstance(x_range[0], str) else base_start + timedelta(days=float(x_range[0]))
        hi = _parse(x_range[1]) if isinstance(x_range[1], str) else base_start + timedelta(days=float(x_range[1]))
    else:
        lo, hi = base_start, base_start + timedelta(days=float(x_range))
    return float(mdates.date2num(lo)), float(mdates.date2num(hi))


def _nice_minute_step(raw: float) -> float:
    for c in (0.5, 1, 2, 5, 10, 15, 20, 30, 60, 120, 240):
        if raw <= c:
            return c
    return 60 * (int(raw / 60) + 1)


def render(config: dict, x_range_override=None) -> None:
    _load_font()

    c = THEMES.get(config.get("theme", "dark"), THEMES["dark"])
    time_unit = config.get("time_unit", "days")
    if time_unit not in ("days", "minutes"):
        raise ValueError(f"time_unit must be 'days' or 'minutes', got {time_unit!r}")
    tm = _time_model(time_unit)

    # Font-size hierarchy, every element multiplied by font_scale so a single
    # knob scales the whole chart while preserving relative sizes:
    #   title > subtitle > axis labels > axis ticks / everything else.
    # label_font_scale gives the task (y-tick) labels an optional extra bump.
    fs = float(config.get("font_scale", 1.0))
    _BASE_FONT = {
        "title":      18,
        "subtitle":   14,
        "axis_label": 12,  # the "Task" / "Elapsed time" axis titles
        "tick":       10,  # axis tick labels (x ticks, task labels)
        "marker":     10,  # kick-off / delivery labels
        "legend":     10,
    }

    def _fs(key: str) -> float:
        return _BASE_FONT[key] * fs

    label_fs = _fs("tick") * float(config.get("label_font_scale", 1.0))

    tasks            = _resolve_dates(config["tasks"], tm)
    tasks_optimistic = _resolve_dates(config["tasks"], tm, use_min_durations=True)

    # Ordered categories
    seen: set = set()
    categories: list[str] = []
    for t in tasks:
        cat = t.get("category", "")
        if cat not in seen:
            categories.append(cat)
            seen.add(cat)

    # Category → color (user config > defaults > palette in order)
    user_colors: dict = config.get("category_colors", {})
    cat_colors: dict[str, str] = {}
    palette_i = 0
    for cat in categories:
        cat_colors[cat] = user_colors.get(cat) or DEFAULT_CATEGORY_COLORS.get(cat) or PALETTE[palette_i % len(PALETTE)]
        if cat not in user_colors and cat not in DEFAULT_CATEGORY_COLORS:
            palette_i += 1

    # Row order: by start date (default), or keep the config's task order when
    # sort_by_start is false. Reverse either way so the first row sits at the top.
    if config.get("sort_by_start", True):
        ordered = sorted(tasks, key=lambda t: t["_start"])
    else:
        ordered = tasks
    tasks_sorted = list(reversed(ordered))
    n = len(tasks_sorted)

    # Figure
    figsize = config.get("figsize", [16, max(4.0, n * 0.55 + 2.5)])
    dpi = config.get("dpi", 150)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(c["bg"])
    ax.set_facecolor(c["bg"])

    all_starts = [t["_start"] for t in tasks]
    all_ends   = [t["_end"]   for t in tasks]

    # X range: explicit override (CLI > config) wins; otherwise auto with padding.
    x_range = x_range_override if x_range_override is not None else config.get("x_range")
    if x_range is not None:
        x_lo, x_hi = _resolve_x_range(x_range, time_unit, min(all_starts))
    else:
        span = max(all_ends) - min(all_starts)
        if time_unit == "minutes":
            pad = timedelta(minutes=max(1.0, span.total_seconds() / 60.0 * 0.03))
        else:
            pad = timedelta(days=max(7, int(span.days * 0.03)))
        x_lo = float(mdates.date2num(min(all_starts) - pad))
        x_hi = float(mdates.date2num(max(all_ends) + pad))
    ax.set_xlim(x_lo, x_hi)

    # Task bars
    bar_height = 0.5
    for i, task in enumerate(tasks_sorted):
        start   = task["_start"]
        min_end = task["_min_end"]
        max_end = task["_end"]
        color   = cat_colors.get(task.get("category", ""), PALETTE[0])
        rgb     = mcolors.to_rgb(color)

        x0 = mdates.date2num(start)
        x1 = mdates.date2num(min_end)
        x2 = mdates.date2num(max_end)

        # Solid bar: start → min_end
        ax.barh(i, x1 - x0, left=x0, height=bar_height, color=color, alpha=0.88, zorder=3)

        # Uncertain extension: min_end → max_end (faded + diagonal stripes)
        if x2 - x1 > 0:
            ax.barh(i, x2 - x1, left=x1,
                    height=bar_height,
                    facecolor=(*rgb, 0.15),
                    edgecolor=(*rgb, 0.6),
                    hatch="//",
                    linewidth=0,
                    zorder=3)

    # Category separators
    if len(categories) > 1:
        prev_cat = tasks_sorted[0].get("category", "")
        for i in range(1, n):
            cur_cat = tasks_sorted[i].get("category", "")
            if cur_cat != prev_cat:
                ax.axhline(i - 0.5, color=c["separator"], linewidth=0.8, alpha=0.5, zorder=2)
            prev_cat = cur_cat

    # Vertical markers
    def _vmarker(x_num, label, color, alpha, linewidth=1.2, y_axes=None, cap_line=False):
        ymax_line = y_axes if (y_axes is not None and cap_line) else 1.0
        ax.axvline(x_num, color=color, linewidth=linewidth, linestyle="--",
                   alpha=alpha, ymin=0, ymax=ymax_line, zorder=4)
        if y_axes is not None:
            ax.text(x_num, y_axes, label, transform=ax.get_xaxis_transform(),
                    color=color, fontsize=_fs("marker"), fontweight="bold",
                    ha="center", va="bottom" if cap_line else "center", alpha=alpha, zorder=5)
        else:
            ax.text(x_num, n - 0.45, label, color=color, fontsize=_fs("marker"), fontweight="bold",
                    ha="center", va="bottom", alpha=alpha, zorder=5)

    def _marker_label(prefix_date: str, prefix_min: str, dt: datetime) -> str:
        if time_unit == "minutes":
            mins = (dt - EPOCH).total_seconds() / 60.0
            return f"{prefix_min}\n{_fmt_minutes_value(mins)}"
        return f"{prefix_date}\n{_fmt_date(dt)}"

    if config.get("show_kickoff", True):
        kickoff = min(all_starts)
        _vmarker(mdates.date2num(kickoff), _marker_label("Kick-off", "Start", kickoff),
                 c["text"], alpha=0.5)

    if config.get("show_delivery", True):
        delivery = max(all_ends)
        delivery_label_y = config.get("delivery_label_y", 0.4)
        _vmarker(mdates.date2num(delivery), _marker_label("Delivery", "End", delivery),
                 c["text"], alpha=0.5, y_axes=delivery_label_y, cap_line=True)

    # Y-axis (task names + duration)
    def _label(t: dict) -> str:
        lo = tm["amount"](t["_start"], t["_min_end"])
        hi = tm["amount"](t["_start"], t["_end"])
        return f"{t['name']}  ({tm['friendly'](lo, hi)})"

    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [_label(t) for t in tasks_sorted],
        fontsize=label_fs, color=c["text"], fontweight="semibold",
    )
    ax.tick_params(axis="y", length=0, pad=8)
    ax.set_ylim(-0.55, n - 0.45)

    # X-axis
    if time_unit == "minutes":
        base_num = float(mdates.date2num(EPOCH))
        lo_min = (x_lo - base_num) * 1440.0
        hi_min = (x_hi - base_num) * 1440.0
        step = _nice_minute_step((hi_min - lo_min) / 7.0)
        first = step * (int(lo_min / step) + (0 if lo_min % step == 0 else 1))
        ticks, m = [], first
        while m <= hi_min + 1e-9:
            ticks.append(base_num + m / 1440.0)
            m += step
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _pos: _fmt_minutes_value((x - base_num) * 1440.0))
        )
        ax.set_xlabel("Elapsed time", color=c["text"], fontsize=_fs("axis_label"), fontweight="semibold", labelpad=8)
    else:
        locator = mdates.AutoDateLocator(minticks=4, maxticks=15)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, show_offset=False))
        ax.set_xlabel("Date", color=c["text"], fontsize=_fs("axis_label"), fontweight="semibold", labelpad=8)
    ax.tick_params(axis="x", labelcolor=c["text"], labelsize=_fs("tick"), length=0, pad=6)
    ax.set_ylabel("Task", color=c["text"], fontsize=_fs("axis_label"), fontweight="semibold", labelpad=10)

    # Grid (vertical only, dashed)
    ax.xaxis.grid(True, color=c["grid"], linestyle="--", linewidth=0.6, zorder=0)
    ax.yaxis.grid(False)

    # Remove all spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Legend (only when categories are named and more than one)
    named_cats = [cat for cat in categories if cat]
    if config.get("show_legend", True) and len(named_cats) > 1:
        patches = [mpatches.Patch(color=cat_colors[cat], label=cat) for cat in named_cats]
        ax.legend(
            handles=patches,
            loc="upper right",
            frameon=False,
            labelcolor=c["text"],
            fontsize=_fs("legend"),
            handlelength=1.2,
            handleheight=0.8,
        )

    # Title & subtitle (top-left, above axes)
    title = config.get("title", "")
    project_start  = min(all_starts)
    min_completion = max(t["_end"] for t in tasks_optimistic)
    max_completion = max(all_ends)

    if time_unit == "minutes":
        min_total = (min_completion - project_start).total_seconds() / 60.0
        max_total = (max_completion - project_start).total_seconds() / 60.0
        if abs(min_total - max_total) < 1e-6:
            completion_str = f"Total runtime {_fmt_minutes_value(max_total)}"
        else:
            completion_str = (
                f"Total runtime between {_fmt_minutes_value(min_total)}"
                f" and {_fmt_minutes_value(max_total)}"
            )
    else:
        min_total = (min_completion - project_start).days
        max_total = (max_completion - project_start).days
        min_dur_str = _friendly_duration(min_total, min_total)
        max_dur_str = _friendly_duration(max_total, max_total)
        if min_completion == max_completion:
            completion_str = f"Expected delivery {_fmt_date(max_completion)} ({max_dur_str})"
        else:
            completion_str = (
                f"Expected delivery between {_fmt_date(min_completion)} ({min_dur_str})"
                f" and {_fmt_date(max_completion)} ({max_dur_str})"
            )

    subtitle_base = config.get("subtitle", "")
    subtitle = f"{subtitle_base} - {completion_str}" if subtitle_base else completion_str

    if title and subtitle:
        ax.text(0, 1.18, title, transform=ax.transAxes, fontsize=_fs("title"),
                fontweight="semibold", color=c["text"], va="bottom")
        ax.text(0, 1.10, subtitle, transform=ax.transAxes, fontsize=_fs("subtitle"),
                color=c["text"], alpha=0.65, va="bottom")
    elif title:
        ax.text(0, 1.06, title, transform=ax.transAxes, fontsize=_fs("title"),
                fontweight="semibold", color=c["text"], va="bottom")

    # Save
    output = config.get("output", "gantt.png")
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=c["bg"])
    plt.close(fig)
    print(f"Saved: {output}")


def _parse_cli_x_range(raw: str):
    """Parse --x-range. Numbers stay numeric (minutes mode); dates stay strings.

    "35"                    -> 35.0
    "5,40"                  -> [5.0, 40.0]
    "2026-01-01,2026-06-01" -> ["2026-01-01", "2026-06-01"]
    """
    parts = [p.strip() for p in raw.split(",")]

    def conv(p):
        try:
            return float(p)
        except ValueError:
            return p

    vals = [conv(p) for p in parts]
    return vals[0] if len(vals) == 1 else vals


def main() -> None:
    parser = argparse.ArgumentParser(description="Gantt chart generator")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", metavar="FILE", help="Path to JSON config file")
    group.add_argument("--json", metavar="JSON", help="Inline JSON config string")
    parser.add_argument(
        "--x-range", dest="x_range", metavar="SPEC",
        help="Pin the x-axis range, overriding config. Minutes mode: '35' or '5,40'. "
             "Days mode: a day count or 'YYYY-MM-DD,YYYY-MM-DD'.",
    )
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    else:
        config = json.loads(args.json)

    x_range_override = _parse_cli_x_range(args.x_range) if args.x_range else None
    render(config, x_range_override=x_range_override)


if __name__ == "__main__":
    main()
