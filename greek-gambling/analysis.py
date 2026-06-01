import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Gambling in Greece — Market Analysis

    ## Terminology

    | Term | Greek | Definition |
    |------|-------|------------|
    | **TGR / Turnover** | Κύκλος Εργασιών | Total amount staked by players. Every bet placed counts, regardless of outcome. This is the gross flow of money through the market — not revenue. |
    | **GGR** | Ακαθάριστα Έσοδα / Μικτά Κέρδη | Gross Gaming Revenue. TGR minus player winnings. This is what operators actually keep, and what the state taxes. Equivalent to "revenue" in other industries. |
    | **OPAP** | ΟΠΑΠ Α.Ε. | The dominant land-based operator — runs sports betting (PAME STOIXHMA), numbers games (KINO, LOTTO, TZOKER), and VLTs through ~4,000 retail shops. |
    | **VLT** | Παιχνιομηχανήματα VLT | Video Lottery Terminals. Electronic slot-like machines in OPAP shops. Introduced late 2017, grew rapidly. |
    | **Online** | Διαδίκτυο | Internet gambling — sports betting, casino games, poker — licensed and operated digitally. Includes OPAP's online arm and third-party operators. |
    | **Casinos** | Καζίνα | Land-based casinos. Greece has ~9 licensed casinos (Parnitha, Thessaloniki, Loutraki, Rhodes, etc.). |
    | **State Lotteries** | Ελληνικά / Κρατικά Λαχεία | Scratch cards (SKRATCH) and draw lotteries (Λαϊκό, Εθνικό, Πρωτοχρονιάτικο). |
    | **ODIE** | ΟΔΙΕ / Ιπποδρομίες Α.Ε. | Horse racing betting. Operated by OPAP's subsidiary from 2015; suspended Jan 2024. |
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np

    return mo, mticker, pd, plt


@app.cell(hide_code=True)
def _(mo, pd):
    DATA_PATH = "/home/gpoulis/projects/portfolio/greek-gambling/data/financial_data.csv"

    df = pd.read_csv(DATA_PATH)
    df["value_eur_millions"] = pd.to_numeric(df["value_eur_millions"], errors="coerce")

    mo.ui.table(df)
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Turnover by Gambling Category

    Each line shows the total amount wagered (TGR — *Κύκλος Εργασιών*) per year, broken down by category.

    The **Online** line is shown faded and dashed before 2020. During 2012–2019, online gambling operated under a transitional licensing regime (Art. 50, Law 4002/2011) in which operators self-declared revenues to the tax authority without formal auditing by the regulator (ΕΕΕΠ). Figures for this period are derived or estimated from report text and growth rates — not sourced from audited tables. From 2020 onward, all figures come from fully audited regulatory data. All other categories (OPAP land-based, Casinos, State Lotteries, Horse Racing) were formally reported throughout.

    *TGR data is not available for 2024 — the 2024 report switched to GGR-only reporting.*
    """)
    return


@app.cell(hide_code=True)
def _(df, mticker, plt):
    import matplotlib.transforms as _transforms
    from matplotlib import font_manager as _fm

    TRUST_YEAR = 2020
    FONT_PATH = "/home/gpoulis/projects/portfolio/greek-gambling/fonts/NotoSansMono_SemiCondensed-SemiBold.ttf"
    BG, TEXT, GRID, SPINE = "#0D1B2A", "#fefae0", "#1b2631", "#2d4a6a"

    CAT_COLORS = {
        "OPAP":            "#31859D",
        "Online":          "#C59849",
        "Casinos":         "#A53F2B",
        "State_Lotteries": "#96BBBB",
        "ODIE":            "#1E497D",
    }
    LABEL_MAP = {
        "OPAP":            "OPAP (land-based)",
        "Online":          "Online",
        "Casinos":         "Casinos",
        "State_Lotteries": "State Lotteries",
        "ODIE":            "Horse Racing (ODIE)",
    }

    _fm.fontManager.addfont(FONT_PATH)
    plt.rcParams["font.family"] = _fm.FontProperties(fname=FONT_PATH).get_name()

    tgr = df[(df["metric"] == "Turnover") & (df["category"] != "Total")].copy()
    pivot = tgr.pivot_table(index="year", columns="category", values="value_eur_millions")

    # Determine top 3 by final value — these get inline labels instead of legend entries
    end_vals = {
        cat: (pivot[cat].dropna().index[-1], pivot[cat].dropna().values[-1])
        for cat in CAT_COLORS if cat in pivot.columns and not pivot[cat].dropna().empty
    }
    top3 = set(sorted(end_vals, key=lambda c: end_vals[c][1], reverse=True)[:3])

    fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG, dpi=150)
    ax.set_facecolor(BG)

    for cat, color in CAT_COLORS.items():
        if cat not in pivot.columns:
            continue
        series = pivot[cat].dropna()
        plot_kw = {} if cat in top3 else {"label": LABEL_MAP.get(cat, cat)}

        if cat == "Online":
            pre  = series[series.index < TRUST_YEAR]
            post = series[series.index >= TRUST_YEAR]
            if not pre.empty:
                ax.plot(pre.index, pre.values, color=color, alpha=0.25,
                        linewidth=1.8, linestyle="--", zorder=2)
            if not pre.empty and not post.empty:
                ax.plot([pre.index[-1], post.index[0]],
                        [pre.values[-1], post.values[0]],
                        color=color, alpha=0.25, linewidth=1.8, linestyle="--", zorder=2)
            if not post.empty:
                ax.plot(post.index, post.values, color=color, alpha=1.0,
                        linewidth=2.5, zorder=3, **plot_kw)
                ax.scatter([post.index[-1]], [post.values[-1]], color=color, s=35, zorder=4)
        else:
            ax.plot(series.index, series.values, color=color, alpha=1.0,
                    linewidth=2.5, zorder=3, **plot_kw)
            ax.scatter([series.index[-1]], [series.values[-1]], color=color, s=35, zorder=4)

    # Inline labels for top 3, just to the right of each line's last point
    for cat in top3:
        year, val = end_vals[cat]
        ax.text(year + 0.35, val, LABEL_MAP[cat],
                color=CAT_COLORS[cat], va="center", ha="left", fontsize=12)

    ax.axvline(TRUST_YEAR - 0.5, color=TEXT, alpha=0.3, linewidth=1, linestyle=":")

    _trans = _transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(TRUST_YEAR - 0.65, 0.97, "← estimated", transform=_trans,
            color=TEXT, alpha=0.4, fontsize=8, ha="right", va="top")
    ax.text(TRUST_YEAR - 0.35, 0.97, "audited →", transform=_trans,
            color=TEXT, alpha=0.85, fontsize=8, ha="left", va="top")

    ax.set_xlabel("Year", color=TEXT, fontsize=10)
    ax.set_ylabel("Turnover (€M)", color=TEXT, fontsize=10)
    ax.set_title("Gambling Turnover by Category — Greece 2012–2023",
                 loc='left', color=TEXT, fontsize=16, pad=44)
    ax.text(0, 1.04, "Online wagering surged in recent years, and the trend remains strong",
            transform=ax.transAxes, color=TEXT, alpha=0.6, fontsize=14, ha='left', va='bottom')
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_xticks(pivot.index)
    ax.set_xlim(right=2026)  # extra space for inline labels

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(SPINE)
    ax.xaxis.set_tick_params(color=SPINE)
    ax.tick_params(left=False)
    ax.yaxis.grid(True, color=GRID, linestyle="--", linewidth=0.5)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    ax.legend(frameon=False, labelcolor=TEXT, fontsize=9, loc="upper left")
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    fig
    return


@app.cell(hide_code=True)
def _(df, mticker, plt):
    def _ggr_chart():
        import matplotlib.transforms as _tr
        from matplotlib import font_manager as _fm

        _TRUST = 2020
        _FONT  = "/home/gpoulis/projects/portfolio/greek-gambling/fonts/NotoSansMono_SemiCondensed-SemiBold.ttf"
        _BG, _TEXT, _GRID, _SPINE = "#0D1B2A", "#fefae0", "#1b2631", "#2d4a6a"

        _COLORS = {
            "OPAP":            "#31859D",
            "Online":          "#C59849",
            "Casinos":         "#A53F2B",
            "State_Lotteries": "#96BBBB",
            "ODIE":            "#1E497D",
        }
        _LABELS = {
            "OPAP":            "OPAP (land-based)",
            "Online":          "Online",
            "Casinos":         "Casinos",
            "State_Lotteries": "State Lotteries",
            "ODIE":            "Horse Racing (ODIE)",
        }

        _fm.fontManager.addfont(_FONT)
        plt.rcParams["font.family"] = _fm.FontProperties(fname=_FONT).get_name()

        _ggr = df[(df["metric"] == "GGR") & (df["category"] != "Total")].copy()
        _piv = _ggr.pivot_table(index="year", columns="category", values="value_eur_millions")

        _ends = {
            c: (_piv[c].dropna().index[-1], _piv[c].dropna().values[-1])
            for c in _COLORS if c in _piv.columns and not _piv[c].dropna().empty
        }
        _top3 = set(sorted(_ends, key=lambda c: _ends[c][1], reverse=True)[:3])

        _fig, _ax = plt.subplots(figsize=(13, 6), facecolor=_BG, dpi=150)
        _ax.set_facecolor(_BG)

        for _cat, _color in _COLORS.items():
            if _cat not in _piv.columns:
                continue
            _s = _piv[_cat].dropna()
            _kw = {} if _cat in _top3 else {"label": _LABELS.get(_cat, _cat)}

            if _cat == "Online":
                _pre  = _s[_s.index < _TRUST]
                _post = _s[_s.index >= _TRUST]
                if not _pre.empty:
                    _ax.plot(_pre.index, _pre.values, color=_color, alpha=0.25,
                             linewidth=1.8, linestyle="--", zorder=2)
                if not _pre.empty and not _post.empty:
                    _ax.plot([_pre.index[-1], _post.index[0]],
                             [_pre.values[-1], _post.values[0]],
                             color=_color, alpha=0.25, linewidth=1.8, linestyle="--", zorder=2)
                if not _post.empty:
                    _ax.plot(_post.index, _post.values, color=_color, alpha=1.0,
                             linewidth=2.5, zorder=3, **_kw)
                    _ax.scatter([_post.index[-1]], [_post.values[-1]], color=_color, s=35, zorder=4)
            else:
                _ax.plot(_s.index, _s.values, color=_color, alpha=1.0,
                         linewidth=2.5, zorder=3, **_kw)
                _ax.scatter([_s.index[-1]], [_s.values[-1]], color=_color, s=35, zorder=4)

        for _cat in _top3:
            _yr, _v = _ends[_cat]
            _ax.text(_yr + 0.35, _v, _LABELS[_cat],
                     color=_COLORS[_cat], va="center", ha="left", fontsize=12)

        _ax.axvline(_TRUST - 0.5, color=_TEXT, alpha=0.3, linewidth=1, linestyle=":")
        _t = _tr.blended_transform_factory(_ax.transData, _ax.transAxes)
        _ax.text(_TRUST - 0.65, 0.97, "← estimated", transform=_t,
                 color=_TEXT, alpha=0.4, fontsize=8, ha="right", va="top")
        _ax.text(_TRUST - 0.35, 0.97, "audited →", transform=_t,
                 color=_TEXT, alpha=0.85, fontsize=8, ha="left", va="top")

        _ax.set_xlabel("Year", color=_TEXT, fontsize=10)
        _ax.set_ylabel("GGR (€M)", color=_TEXT, fontsize=10)
        _ax.set_title("Gross Gaming Revenue (GGR) by Category — Greece 2012–2024",
                      loc='left', color=_TEXT, fontsize=16, pad=44)
        _ax.text(0, 1.04, "OPAP leads on revenue despite lower turnover but Online is catching up fast",
                 transform=_ax.transAxes, color=_TEXT, alpha=0.6, fontsize=14, ha='left', va='bottom')
        _ax.tick_params(colors=_TEXT, labelsize=9)
        _ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        _ax.set_xticks(_piv.index)
        _ax.set_xlim(right=_piv.index.max() + 3)

        for _sp in _ax.spines.values():
            _sp.set_visible(False)
        _ax.spines["bottom"].set_visible(True)
        _ax.spines["bottom"].set_color(_SPINE)
        _ax.xaxis.set_tick_params(color=_SPINE)
        _ax.tick_params(left=False)
        _ax.yaxis.grid(True, color=_GRID, linestyle="--", linewidth=0.5)
        _ax.xaxis.grid(False)
        _ax.set_axisbelow(True)

        _ax.legend(frameon=False, labelcolor=_TEXT, fontsize=9, loc="upper left")
        _fig.tight_layout(rect=[0, 0, 1, 0.84])
        return _fig

    _ggr_chart()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Hold Percentage

    **Hold %** = GGR ÷ TGR × 100

    This is the standard industry metric for how much of every euro wagered the operator retains as revenue. Its complement is the **Return to Player (RTP)** rate — what players get back on average. A hold of 5% means players receive €0.95 back for every €1 staked.

    Hold % is structurally very different across gambling types:
    - **Lotteries** are high-hold by design — a large share funds the state prize pool and government revenue
    - **Casinos** sit in the middle — table games and slots have fixed mathematical edges
    - **OPAP** blends sports betting margins with numbers games (KINO, LOTTO)
    - **Online** is low-hold because digital operators compete on RTP to attract players

    Only years where both TGR and GGR are available are shown (2024 is excluded — the 2024 report dropped TGR figures). The Online line is faded before 2020 for the same data-quality reason as above.
    """)
    return


@app.cell(hide_code=True)
def _(df, mticker, plt):
    def _hold_chart():
        import matplotlib.transforms as _tr
        from matplotlib import font_manager as _fm

        _TRUST = 2020
        _FONT  = "/home/gpoulis/projects/portfolio/greek-gambling/fonts/NotoSansMono_SemiCondensed-SemiBold.ttf"
        _BG, _TEXT, _GRID, _SPINE = "#0D1B2A", "#fefae0", "#1b2631", "#2d4a6a"

        _COLORS = {
            "OPAP":            "#31859D",
            "Online":          "#C59849",
            "Casinos":         "#A53F2B",
            "State_Lotteries": "#96BBBB",
            "ODIE":            "#1E497D",
        }
        _LABELS = {
            "OPAP":            "OPAP (land-based)",
            "Online":          "Online",
            "Casinos":         "Casinos",
            "State_Lotteries": "State Lotteries",
            "ODIE":            "Horse Racing (ODIE)",
        }

        _fm.fontManager.addfont(_FONT)
        plt.rcParams["font.family"] = _fm.FontProperties(fname=_FONT).get_name()

        _tgr = df[df["metric"] == "Turnover"].pivot_table(
            index="year", columns="category", values="value_eur_millions")
        _ggr = df[df["metric"] == "GGR"].pivot_table(
            index="year", columns="category", values="value_eur_millions")

        _hold = (_ggr / _tgr * 100).drop(columns=["Total"], errors="ignore")

        _ends = {
            c: (_hold[c].dropna().index[-1], _hold[c].dropna().values[-1])
            for c in _COLORS if c in _hold.columns and not _hold[c].dropna().empty
        }
        _top3 = set(sorted(_ends, key=lambda c: _ends[c][1], reverse=True)[:3])

        _fig, _ax = plt.subplots(figsize=(15, 6), facecolor=_BG, dpi=150)
        _fig.patch.set_facecolor(_BG)
        _ax.set_facecolor(_BG)

        for _cat, _color in _COLORS.items():
            if _cat not in _hold.columns:
                continue
            _s = _hold[_cat].dropna()
            _kw = {}

            if _cat == "Online":
                _pre  = _s[_s.index < _TRUST]
                _post = _s[_s.index >= _TRUST]
                if not _pre.empty:
                    _ax.plot(_pre.index, _pre.values, color=_color, alpha=0.25,
                             linewidth=1.8, linestyle="--", zorder=2)
                if not _pre.empty and not _post.empty:
                    _ax.plot([_pre.index[-1], _post.index[0]],
                             [_pre.values[-1], _post.values[0]],
                             color=_color, alpha=0.25, linewidth=1.8, linestyle="--", zorder=2)
                if not _post.empty:
                    _ax.plot(_post.index, _post.values, color=_color, alpha=1.0,
                             linewidth=2.5, zorder=3, **_kw)
                    _ax.scatter([_post.index[-1]], [_post.values[-1]], color=_color, s=35, zorder=4)
            else:
                _ax.plot(_s.index, _s.values, color=_color, alpha=1.0,
                         linewidth=2.5, zorder=3, **_kw)
                _ax.scatter([_s.index[-1]], [_s.values[-1]], color=_color, s=35, zorder=4)

        for _cat, (_yr, _v) in _ends.items():
            _ax.text(_yr + 0.35, _v, f"{_LABELS[_cat]} ({_v:.1f}% in {_yr})",
                     color=_COLORS[_cat], va="center", ha="left", fontsize=12, clip_on=False)

        _ax.axvline(_TRUST - 0.5, color=_TEXT, alpha=0.3, linewidth=1, linestyle=":")
        _t = _tr.blended_transform_factory(_ax.transData, _ax.transAxes)
        _ax.text(_TRUST - 0.65, 0.97, "← estimated", transform=_t,
                 color=_TEXT, alpha=0.4, fontsize=8, ha="right", va="top")
        _ax.text(_TRUST - 0.35, 0.97, "audited →", transform=_t,
                 color=_TEXT, alpha=0.85, fontsize=8, ha="left", va="top")

        _ax.set_xlabel("Year", color=_TEXT, fontsize=10)
        _ax.set_ylabel("Hold % (GGR / TGR)", color=_TEXT, fontsize=10)
        _ax.set_title("Operator Hold Percentage by Category — Greece 2012–2023",
                      loc='left', color=_TEXT, fontsize=16, pad=44)
        _ax.text(0, 1.04, "Online's low hold (~4%–5%) explains why it trails on revenue while it dominates turnover",
                 transform=_ax.transAxes, color=_TEXT, alpha=0.6, fontsize=14, ha='left', va='bottom')

        _ax.tick_params(colors=_TEXT, labelsize=9)
        _ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
        _valid_years = _hold.dropna(how="all").index
        _ax.set_xticks(_valid_years)
        _ax.set_xlim(right=_valid_years.max() + 7)
        _ax.margins(y=0.12)

        for _sp in _ax.spines.values():
            _sp.set_visible(False)
        _ax.spines["bottom"].set_visible(True)
        _ax.spines["bottom"].set_color(_SPINE)
        _ax.xaxis.set_tick_params(color=_SPINE)
        _ax.tick_params(left=False)
        _ax.yaxis.grid(True, color=_GRID, linestyle="--", linewidth=0.5)
        _ax.xaxis.grid(False)
        _ax.set_axisbelow(True)

        _fig.subplots_adjust(left=0.06, right=0.97, top=0.82, bottom=0.10)
        return _fig

    _hold_chart()
    return


@app.cell(hide_code=True)
def _(df, mticker, plt):
    def _waterfall_chart():
        from matplotlib import font_manager as _fm
        from matplotlib.patches import FancyArrowPatch

        _FONT  = "/home/gpoulis/projects/portfolio/greek-gambling/fonts/NotoSansMono_SemiCondensed-SemiBold.ttf"
        _BG, _TEXT, _GRID, _SPINE = "#0D1B2A", "#fefae0", "#1b2631", "#2d4a6a"

        _COLORS = {
            "OPAP":            "#31859D",
            "Online":          "#C59849",
            "Casinos":         "#A53F2B",
            "State_Lotteries": "#96BBBB",
            "ODIE":            "#1E497D",
        }
        _LABELS = {
            "OPAP":            "OPAP",
            "Online":          "Online",
            "Casinos":         "Casinos",
            "State_Lotteries": "State Lotteries",
            "ODIE":            "Horse Racing",
        }
        _TOTAL_COLOR = "#fefae0"

        _fm.fontManager.addfont(_FONT)
        plt.rcParams["font.family"] = _fm.FontProperties(fname=_FONT).get_name()

        _d = (
            df[(df["year"] == 2024) & (df["metric"] == "GGR") & (df["category"] != "Total")]
            .set_index("category")["value_eur_millions"]
            .sort_values(ascending=False)
        )

        _cats   = list(_d.index) + ["Total"]
        _vals   = list(_d.values) + [_d.sum()]
        _starts = []
        _cumsum = 0
        for _i, (_c, _v) in enumerate(zip(_cats, _vals)):
            if _c == "Total":
                _starts.append(0)
            else:
                _starts.append(_cumsum)
                _cumsum += _v

        _xlabels = [_LABELS.get(c, c) for c in _cats]
        _colors  = [_COLORS.get(c, _TOTAL_COLOR) for c in _cats]
        _x       = list(range(len(_cats)))

        _fig, _ax = plt.subplots(figsize=(12, 6), facecolor=_BG, dpi=150)
        _ax.set_facecolor(_BG)

        _BAR_W = 0.55
        for _xi, (_start, _val, _color, _cat) in enumerate(zip(_starts, _vals, _colors, _cats)):
            _is_total = _cat == "Total"
            _ax.bar(_xi, _val, bottom=_start, color=_color,
                    width=_BAR_W, zorder=2,
                    alpha=1.0, edgecolor=_BG, linewidth=0.5,
                    linestyle="--" if _is_total else "-",
                    fill=not _is_total)
            if _is_total:
                _ax.bar(_xi, _val, bottom=_start, color=_color,
                        width=_BAR_W, zorder=2, alpha=0.12)

            # Contribution label inside / above bar
            _label_y = _start + _val / 2
            _ax.text(_xi, _label_y, f"€{_val:,.0f}M",
                     color=_BG if not _is_total else _TEXT,
                     fontsize=8.5, ha="center", va="center",
                     fontweight="bold", zorder=3)

            # Cumulative total above each bar (except Total bar)
            if not _is_total:
                _cum_val = _start + _val
                _cum_pct = _cum_val / _d.sum() * 100
                _ax.text(_xi, _cum_val + 25, f"€{_cum_val:,.0f}M  ({_cum_pct:.1f}%)",
                         color=_color, fontsize=10, ha="center", va="bottom")

        # Connector lines between bars
        for _i in range(len(_cats) - 2):
            _top = _starts[_i] + _vals[_i]
            _ax.plot([_i + _BAR_W / 2, _i + 1 - _BAR_W / 2],
                     [_top, _top], color=_SPINE, linewidth=1, linestyle="--", zorder=1)

        # Styling
        _ax.set_xticks(_x)
        _ax.set_xticklabels(_xlabels, color=_TEXT, fontsize=10)
        _ax.set_ylabel("GGR (€M)", color=_TEXT, fontsize=10)
        _ax.tick_params(axis="y", colors=_TEXT, labelsize=9, left=False)
        _ax.tick_params(axis="x", bottom=False)
        _ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        _ax.set_ylim(0, _d.sum() * 1.18)

        for _sp in _ax.spines.values():
            _sp.set_visible(False)
        _ax.spines["bottom"].set_visible(True)
        _ax.spines["bottom"].set_color(_SPINE)
        _ax.yaxis.grid(True, color=_GRID, linestyle="--", linewidth=0.5)
        _ax.xaxis.grid(False)
        _ax.set_axisbelow(True)

        _ax.set_title("Cumulative GGR by Category — Greece 2024",
                      loc='left', color=_TEXT, fontsize=16, pad=44)
        _ax.text(0, 1.04, "OPAP and Online alone account for 86.9% of total GGR — €2.5B of a €2.9B market",
                 transform=_ax.transAxes, color=_TEXT, alpha=0.6, fontsize=14, ha='left', va='bottom')
        _fig.tight_layout(rect=[0, 0, 1, 0.84])
        return _fig

    _waterfall_chart()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Summary

    - The Greek gambling market has undergone a major transformation over the
      past decade, with online gambling growing from a negligible share to dominating turnover by 2023.

    - OPAP remains the largest single operator by revenue, but online operators
      are rapidly closing the gap due to their dominant share of turnover, even with lower hold percentages.

    - In 2024, OPAP and Online combined account for €2.5B of GGR, representing
      86.9% of the total €2.9B market revenue.
    """)
    return


if __name__ == "__main__":
    app.run()
