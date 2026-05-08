#!/usr/bin/env python3
"""Gantt chart generator.

Usage:
    python gantt.py --config config.json
    python gantt.py --json '{"title": "...", "tasks": [...]}'
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

_DIR = Path(__file__).parent.parent
_BUNDLED_FONT = _DIR / "fonts" / "NotoSansMono_SemiCondensed-SemiBold.ttf"
_SYSTEM_FONT = Path("/usr/share/fonts/noto_sans_mono/NotoSansMono_SemiCondensed-SemiBold.ttf")

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
    if min_days % 30 == 0 and max_days % 30 == 0:
        return f"{min_days // 30}–{max_days // 30} months"
    min_w, max_w = min_days / 7, max_days / 7
    exact = min_w == int(min_w) and max_w == int(max_w)
    prefix = "" if exact else "~"
    return f"{prefix}{round(min_w)}–{round(max_w)} weeks"


def _resolve_dates(tasks: list[dict], use_min_durations: bool = False) -> list[dict]:
    """Resolve concrete start/min_end/max_end for all tasks.

    Duration fields (pick one):
      "duration"               — fixed length in days
      "duration_range": [min, max] — uncertain length; max used for dependency chaining,
                                     min rendered as solid bar, extra as faded+striped extension
      "end"                    — explicit end date (treated as both min and max)

    Start fields (one or more; the latest date wins):
      "start"                  — explicit start date
      "depends_on"             — start after the chained end of named task(s)
      "start_offset"           — start N days after another task's start date
                                 e.g. {"from": "Audit", "days": 7}

    Finish constraint:
      "finish_requires"        — task(s) that must finish before this task can finish
                                 (finish-to-finish); applied after normal end computation

    use_min_durations: if True, duration_range tasks use their minimum for chaining
                       (used to compute the optimistic earliest delivery date).
    """
    by_name = {t["name"]: t for t in tasks}
    # stores (start, min_end, max_end)
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
            start_candidates.append(_parse(task["start"]))

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
            start_candidates.append(ref_start + timedelta(days=int(offset_cfg["days"])))

        if not start_candidates:
            raise ValueError(f"Task '{name}' needs a start date, depends_on, or start_offset")
        start = max(start_candidates)

        # duration / duration_range / end
        if "duration_range" in task:
            min_dur, max_dur = task["duration_range"]
            min_end = start + timedelta(days=int(min_dur))
            # In optimistic mode both ends use min_dur so downstream tasks chain from it
            max_end = start + timedelta(days=int(min_dur if use_min_durations else max_dur))
        elif "duration" in task:
            min_end = max_end = start + timedelta(days=int(task["duration"]))
        elif "end" in task:
            min_end = max_end = _parse(task["end"])
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
            "start":    start.strftime("%Y-%m-%d"),
            "_min_end": min_end.strftime("%Y-%m-%d"),
            "end":      max_end.strftime("%Y-%m-%d"),
        })
    return result


def render(config: dict) -> None:
    _load_font()

    c = THEMES.get(config.get("theme", "dark"), THEMES["dark"])
    tasks           = _resolve_dates(config["tasks"])
    tasks_optimistic = _resolve_dates(config["tasks"], use_min_durations=True)

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

    # Sort by start date; reverse so top task has highest y
    tasks_sorted = list(reversed(
        sorted(tasks, key=lambda t: _parse(t["start"]))
    ))
    n = len(tasks_sorted)

    # Figure
    figsize = config.get("figsize", [16, max(4.0, n * 0.55 + 2.5)])
    dpi = config.get("dpi", 150)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(c["bg"])
    ax.set_facecolor(c["bg"])

    # X range with padding
    all_starts   = [_parse(t["start"])    for t in tasks]
    all_min_ends = [_parse(t["_min_end"]) for t in tasks]
    all_ends     = [_parse(t["end"])      for t in tasks]
    span_days = (max(all_ends) - min(all_starts)).days
    pad = timedelta(days=max(7, int(span_days * 0.03)))
    ax.set_xlim(
        mdates.date2num(min(all_starts) - pad),
        mdates.date2num(max(all_ends) + pad),
    )

    # Task bars
    bar_height = 0.5
    for i, task in enumerate(tasks_sorted):
        start   = _parse(task["start"])
        min_end = _parse(task["_min_end"])
        max_end = _parse(task["end"])
        color   = cat_colors.get(task.get("category", ""), PALETTE[0])
        rgb     = mcolors.to_rgb(color)

        # Solid bar: start → min_end
        ax.barh(i, (min_end - start).days, left=mdates.date2num(start),
                height=bar_height, color=color, alpha=0.88, zorder=3)

        # Uncertain extension: min_end → max_end (faded + diagonal stripes)
        extra = (max_end - min_end).days
        if extra > 0:
            ax.barh(i, extra, left=mdates.date2num(min_end),
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
                    color=color, fontsize=9, fontweight="bold",
                    ha="center", va="bottom" if cap_line else "center", alpha=alpha, zorder=5)
        else:
            ax.text(x_num, n - 0.45, label, color=color, fontsize=9, fontweight="bold",
                    ha="center", va="bottom", alpha=alpha, zorder=5)

    if config.get("show_kickoff", True):
        kickoff = min(all_starts)
        _vmarker(mdates.date2num(kickoff), f"Kick-off\n{_fmt_date(kickoff)}", c["text"], alpha=0.5)

    if config.get("show_delivery", True):
        delivery = max(all_ends)
        _vmarker(mdates.date2num(delivery), f"Delivery\n{_fmt_date(delivery)}", c["text"], alpha=0.5,
                 y_axes=0.4, cap_line=True)

    # Y-axis (task names + duration)
    def _label(t: dict) -> str:
        min_days = (_parse(t["_min_end"]) - _parse(t["start"])).days
        max_days = (_parse(t["end"])      - _parse(t["start"])).days
        return f"{t['name']}  ({_friendly_duration(min_days, max_days)})"

    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [_label(t) for t in tasks_sorted],
        fontsize=9, color=c["text"], fontweight="semibold",
    )
    ax.tick_params(axis="y", length=0, pad=8)
    ax.set_ylim(-0.55, n - 0.45)

    # X-axis (dates)
    locator = mdates.AutoDateLocator(minticks=4, maxticks=15)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, show_offset=False))
    ax.tick_params(axis="x", labelcolor=c["text"], labelsize=9, length=0, pad=6)
    ax.set_xlabel("Date", color=c["text"], fontsize=9, fontweight="semibold", labelpad=8)
    ax.set_ylabel("Task", color=c["text"], fontsize=9, fontweight="semibold", labelpad=10)

    # Grid (vertical only, dashed)
    ax.xaxis.grid(True, color=c["grid"], linestyle="--", linewidth=0.6, zorder=0)
    ax.yaxis.grid(False)

    # Remove all spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Legend (only when categories are named and more than one)
    named_cats = [cat for cat in categories if cat]
    if len(named_cats) > 1:
        patches = [mpatches.Patch(color=cat_colors[cat], label=cat) for cat in named_cats]
        ax.legend(
            handles=patches,
            loc="upper right",
            frameon=False,
            labelcolor=c["text"],
            fontsize=9,
            handlelength=1.2,
            handleheight=0.8,
        )

    # Title & subtitle (top-left, above axes)
    title = config.get("title", "")
    # Build subtitle: user text + computed completion range
    project_start  = min(all_starts)
    min_completion = max(_parse(t["end"]) for t in tasks_optimistic)
    max_completion = max(all_ends)
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
        ax.text(0, 1.18, title, transform=ax.transAxes, fontsize=14,
                fontweight="semibold", color=c["text"], va="bottom")
        ax.text(0, 1.10, subtitle, transform=ax.transAxes, fontsize=10,
                color=c["text"], alpha=0.65, va="bottom")
    elif title:
        ax.text(0, 1.06, title, transform=ax.transAxes, fontsize=14,
                fontweight="semibold", color=c["text"], va="bottom")

    # Save
    output = config.get("output", "gantt.png")
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=c["bg"])
    plt.close(fig)
    print(f"Saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gantt chart generator")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", metavar="FILE", help="Path to JSON config file")
    group.add_argument("--json", metavar="JSON", help="Inline JSON config string")
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    else:
        config = json.loads(args.json)

    render(config)


if __name__ == "__main__":
    main()
