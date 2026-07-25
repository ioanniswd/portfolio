"""
Natural-language painting search — the production funnel.

    user text ─► Ollama (structured JSON) ─► SearchIntent
              ─► resolve color pools (numpy KNN over the vocab)
              ─► compose a parameterized Cypher query
              ─► run against Neo4j ─► print matches

The LLM never writes Cypher. Its only job is to decompose a request into a
fixed set of constraints (colors / moods / lightness / contrast / exclusions).
Everything downstream is deterministic code, so a small local model is enough.

Usage:
    uv run python scripts/search.py "predominantly gold, with no red colours"
    uv run python scripts/search.py "warm and earthy" --limit 8
    uv run python scripts/search.py --show-intent --show-cypher "deep ocean water"
    uv run python scripts/search.py            # interactive REPL

Requires Ollama running locally with the model pulled:
    ollama pull llama3.1:8b
"""
import argparse
import difflib
import os
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import ollama
from dotenv import load_dotenv
from neo4j import GraphDatabase
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent))
from colors import build_vocabulary, rgb_to_lab, FAMILIES  # noqa: E402

_PROJECT_ROOT = Path(__file__).parent.parent

# ── Intent schema (what the LLM must fill) ────────────────────────────────────

Quantifier = Literal["predominantly", "mostly", "a_lot", "some", "main", "any"]


class ColorConstraint(BaseModel):
    """A color the painting should contain."""
    color_name: str | None = Field(
        None, description="Standard color name if it is a common/pigment color, else null")
    approx_rgb: list[int] | None = Field(
        None, description="Estimated [r,g,b] 0-255 for off-vocabulary/descriptive colors")
    match: Literal["exact", "adjacent", "family"] = "adjacent"
    quantifier: Quantifier = "a_lot"


class MoodConstraint(BaseModel):
    """A broad palette vibe expressed as one or more color families."""
    families: list[str]
    min_coverage: float = 0.4


class Exclude(BaseModel):
    """A color/family the painting should NOT meaningfully contain."""
    family: str | None = None
    color_name: str | None = None
    # "no red" rarely means zero red — treat trace amounts as absent.
    max_percentage: float = 0.05


class SearchIntent(BaseModel):
    colors: list[ColorConstraint] = []
    moods: list[MoodConstraint] = []
    exclude: list[Exclude] = []
    lightness: Literal["dark", "light", "any"] = "any"
    contrast: bool = False
    limit: int = 5


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""You translate a person's free-text request for paintings into a structured
search intent. You do NOT write database queries — you only fill the schema.

The collection is searchable by the colors present in each painting.

Color families (use these EXACT strings for `moods` and `exclude.family`):
{chr(10).join('  - ' + f for f in FAMILIES)}

Rules:
- `colors`: one entry per specific color the user names or implies. ALWAYS fill
  BOTH `color_name` and `approx_rgb`:
  * `color_name`: your best short name for the colour — a common colour word or
    artist pigment ("blue", "gold", "crimson", "burnt sienna", "sand", "teal",
    "sage", "navy", "salmon"). Even for a descriptive phrase, give the closest
    single colour word (e.g. "deep ocean water" → "ocean blue"; "luxury green"
    → "emerald"; "colour of sand" → "sand").
  * `approx_rgb`: your best [r,g,b] 0-255 estimate — used as a fallback if the
    named colour isn't in the palette. So always provide both.
  * `quantifier` — map quantity words:
      "predominantly X" -> predominantly   (rank<=2 and >30% of the painting)
      "mostly X"        -> mostly           (rank<=2)
      "a lot of X"      -> a_lot            (rank<=3)
      "some/touches X"  -> some             (minor presence)
      "X as the main color" -> main         (the single most common color)
      no quantity word  -> a_lot
  * `match`: "adjacent" (default) = a shade and its near neighbours; use it for a
    specific shade ("burgundy", "teal", "deep ocean water"). Use "family" for a
    broad colour category — a bare colour word like "blue", "green", "reds",
    "red tones". Use "exact" ONLY if the user says "exactly". (A plain "a burgundy
    painting" is match "adjacent", quantifier "a_lot" — not "main".)
- `moods`: broad vibes. "warm" -> Yellows & Golds, Oranges & Corals, Reds & Pinks,
  Browns & Earth Tones. "cool" -> Blues, Greens, Purples & Violets.
  "earthy" -> Browns & Earth Tones, Yellows & Golds. Combine families for a
  single vibe into ONE mood entry.
- `exclude`: colors/families the user does NOT want ("no red", "without blue").
  Use `family` for a color group, `color_name` for one specific color.
- `lightness`: "dark and dramatic / moody" -> dark. "light / airy / bright" -> light.
- `contrast`: true only if the user asks for high contrast / dark-and-bright.
- `limit`: a number the user asks for ("3 paintings"), else 5.

Examples:
  "I want a predominantly gold painting, with no red colours"
    -> {{"colors":[{{"color_name":"gold","match":"adjacent","quantifier":"predominantly"}}],
         "exclude":[{{"family":"Reds & Pinks"}}],"limit":5}}
  "predominantly white and gold"
    -> {{"colors":[{{"color_name":"white","quantifier":"predominantly"}},
                   {{"color_name":"gold","quantifier":"predominantly"}}]}}
  "find me paintings that feel warm and earthy"
    -> {{"moods":[{{"families":["Yellows & Golds","Oranges & Corals","Reds & Pinks","Browns & Earth Tones"]}}]}}
  "something with the color of deep ocean water"
    -> {{"colors":[{{"color_name":"ocean blue","approx_rgb":[16,52,86],"match":"adjacent"}}]}}
  "find paintings with the color of sand"
    -> {{"colors":[{{"color_name":"sand","approx_rgb":[194,178,128],"match":"adjacent"}}]}}
  "3 paintings with blue"
    -> {{"colors":[{{"color_name":"blue","match":"family","quantifier":"a_lot"}}],"limit":3}}
  "paintings with lots of green"
    -> {{"colors":[{{"color_name":"green","match":"family","quantifier":"a_lot"}}]}}
  "I want a burgundy painting"
    -> {{"colors":[{{"color_name":"burgundy","match":"adjacent","quantifier":"a_lot"}}]}}
"""


# ── LLM step ──────────────────────────────────────────────────────────────────

def translate(query: str, model: str, host: str) -> SearchIntent:
    """Ask the local model to decompose the request into a SearchIntent."""
    client = ollama.Client(host=host)
    resp = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        format=SearchIntent.model_json_schema(),  # grammar-constrained JSON
        options={"temperature": 0},
    )
    # ollama >=0.4 returns a ChatResponse object; older versions a dict.
    content = getattr(getattr(resp, "message", None), "content", None) or resp["message"]["content"]
    return SearchIntent.model_validate_json(content)


# ── Color-pool resolution (numpy KNN over the in-process vocabulary) ──────────

def _knn_names(lab: np.ndarray, vocab: dict, k: int) -> list[str]:
    """k nearest named colors to a LAB point, by Euclidean distance (matches the index)."""
    d = ((vocab["labs"] - lab) ** 2).sum(axis=1)
    idx = np.argsort(d)[:k]
    return [vocab["names"][i] for i in idx]


def resolve_color_term(term: str, vocab: dict) -> str | None:
    """Match a colour word to a vocabulary colour name: exact first, then fuzzy (fixes typos).

    e.g. "burgundy" -> "burgundy", "burgandy" -> "burgundy", "greeen" -> "green".
    Returns None if nothing is close enough (caller then falls back to approx_rgb).
    """
    t = term.strip().lower()
    names = vocab["names"]
    if t in set(names):
        return t
    matches = difflib.get_close_matches(t, names, n=1, cutoff=0.8)
    return matches[0] if matches else None


def _family_members(family: str, vocab: dict) -> list[str]:
    """All vocabulary colour names belonging to a given family."""
    return [n for n, f in zip(vocab["names"], vocab["families"]) if f == family]


def resolve_pool(c: ColorConstraint, vocab: dict, k: int = 8) -> list[str]:
    """Turn one colour constraint into a concrete list of vocabulary colour names.

    match="exact"    → a tight neighbourhood (a specific shade)
    match="adjacent" → perceptual neighbours (default; a shade + near ones)
    match="family"   → the entire colour family (a broad category like "blue")

    The colour word is matched against the vocab with typo tolerance; off-vocab
    descriptions fall back to their estimated RGB.
    """
    names = vocab["names"]
    name_to_idx = {n: i for i, n in enumerate(names)}

    resolved = resolve_color_term(c.color_name, vocab) if c.color_name else None
    if resolved:
        idx = name_to_idx[resolved]
        if c.match == "family":
            return _family_members(vocab["families"][idx], vocab)
        return _knn_names(vocab["labs"][idx], vocab, 3 if c.match == "exact" else k)

    if c.approx_rgb:
        lab = rgb_to_lab(np.array(c.approx_rgb, dtype=float))
        if c.match == "family":
            nearest = name_to_idx[_knn_names(lab, vocab, 1)[0]]
            return _family_members(vocab["families"][nearest], vocab)
        return _knn_names(lab, vocab, k)

    return []


# ── Cypher builder ────────────────────────────────────────────────────────────

_QUANTIFIER_COND = {
    # (pct_col, rank_col) -> boolean expression
    "predominantly": lambda p, r: f"{p} > 0.30 AND {r} <= 2",
    "mostly":        lambda p, r: f"{r} <= 2",
    "a_lot":         lambda p, r: f"{r} <= 3",
    "some":          lambda p, r: f"{p} > 0 AND {p} < 0.15 AND {r} <= 5",
    "main":          lambda p, r: f"{r} = 1",
    "any":           lambda p, r: f"{p} > 0",
}


def build_query(intent: SearchIntent, vocab: dict) -> tuple[str, dict]:
    """Compose one parameterized Cypher query from all constraints."""
    lines = ["MATCH (p:Painting)"]
    params: dict = {}
    carry = ["p"]          # columns kept alive across WITH boundaries
    score_terms: list[str] = []

    def with_line() -> str:
        return "WITH " + ", ".join(carry)

    # ── colors ────────────────────────────────────────────────────────────
    for i, c in enumerate(intent.colors):
        pool = resolve_pool(c, vocab)
        if not pool:
            continue
        pct, rank = f"c{i}_pct", f"c{i}_rank"
        params[f"c{i}_pool"] = pool
        lines += [
            "CALL (p) {",
            f"  MATCH (p)-[h:HAS_COLOR]->(col:Color) WHERE col.name IN $c{i}_pool",
            "  RETURN coalesce(sum(h.percentage), 0.0) AS pct, coalesce(min(h.rank), 9999) AS best_rank }",
        ]
        carry += [f"pct AS {pct}", f"best_rank AS {rank}"]
        lines.append(with_line())
        carry = [x.split(" AS ")[-1] for x in carry]  # normalize to bare names
        lines.append(f"WHERE {_QUANTIFIER_COND[c.quantifier](pct, rank)}")
        score_terms.append(pct)

    # ── moods ─────────────────────────────────────────────────────────────
    for j, m in enumerate(intent.moods):
        fams = [f for f in m.families if f in FAMILIES]
        if not fams:
            continue
        col = f"m{j}_pct"
        params[f"m{j}_fams"] = fams
        params[f"m{j}_min"] = m.min_coverage
        lines += [
            "CALL (p) {",
            f"  MATCH (p)-[h:HAS_COLOR]->(col:Color) WHERE col.family IN $m{j}_fams",
            "  RETURN coalesce(sum(h.percentage), 0.0) AS pct }",
        ]
        carry.append(f"pct AS {col}")
        lines.append(with_line())
        carry = [x.split(" AS ")[-1] for x in carry]
        lines.append(f"WHERE {col} > $m{j}_min")
        score_terms.append(col)

    # ── lightness ─────────────────────────────────────────────────────────
    if intent.lightness != "any":
        cond = "col.l < 30" if intent.lightness == "dark" else "col.l > 75"
        col = "light_pct"
        lines += [
            "CALL (p) {",
            f"  MATCH (p)-[h:HAS_COLOR]->(col:Color) WHERE {cond}",
            "  RETURN coalesce(sum(h.percentage), 0.0) AS pct }",
        ]
        carry.append(f"pct AS {col}")
        lines.append(with_line())
        carry = [x.split(" AS ")[-1] for x in carry]
        lines.append(f"WHERE {col} > 0.35")
        score_terms.append(col)

    # ── exclusions ────────────────────────────────────────────────────────
    ex_conds = []
    for k, ex in enumerate(intent.exclude):
        if ex.family and ex.family in FAMILIES:
            params[f"ex{k}_fam"] = ex.family
            params[f"ex{k}_max"] = ex.max_percentage
            ex_conds.append(
                f"NOT EXISTS {{ MATCH (p)-[hx:HAS_COLOR]->(x:Color) "
                f"WHERE x.family = $ex{k}_fam AND hx.percentage > $ex{k}_max }}")
        elif ex.color_name:
            params[f"ex{k}_pool"] = resolve_pool(
                ColorConstraint(color_name=ex.color_name, match="adjacent"), vocab)
            params[f"ex{k}_max"] = ex.max_percentage
            ex_conds.append(
                f"NOT EXISTS {{ MATCH (p)-[hx:HAS_COLOR]->(x:Color) "
                f"WHERE x.name IN $ex{k}_pool AND hx.percentage > $ex{k}_max }}")
    if ex_conds:
        lines.append(with_line())
        lines.append("WHERE " + "\n  AND ".join(ex_conds))

    # ── contrast ──────────────────────────────────────────────────────────
    if intent.contrast:
        lines.append(with_line())
        lines.append(
            "WHERE EXISTS { MATCH (p)-[:HAS_COLOR]->(d:Color) WHERE d.l < 25 }\n"
            "  AND EXISTS { MATCH (p)-[hb:HAS_COLOR]->(b:Color) WHERE b.l > 70 AND hb.percentage > 0.05 }")

    # ── return / order ────────────────────────────────────────────────────
    score = " + ".join(score_terms) if score_terms else "0.0"
    lines += [
        f"RETURN p.name AS painting, p.path AS path, round(({score}) * 100, 1) AS score",
        "ORDER BY score DESC",
        "LIMIT $limit",
    ]
    params["limit"] = intent.limit
    return "\n".join(lines), params


# ── Execution ─────────────────────────────────────────────────────────────────

def run_cypher(cypher: str, params: dict) -> list[dict]:
    uri = os.environ["NEO4J_URI"]
    user = os.environ.get("NEO4J_USER", "neo4j")
    pwd = os.environ.get("NEO4J_PASSWORD", "")
    auth = (user, pwd) if pwd else None
    driver = GraphDatabase.driver(uri, auth=auth)
    try:
        with driver.session() as session:
            return session.run(cypher, **params).data()
    finally:
        driver.close()


# ── Progressive relaxation ────────────────────────────────────────────────────
# When a query returns nothing, loosen it one notch and retry, rather than giving
# up. Never fabricates matches — it only widens the ask and reports what it did.

# Strict → loose ladder for "how dominant" a colour must be.
_RELAX_LADDER = ["main", "predominantly", "mostly", "a_lot"]


def _relax_once(intent: SearchIntent) -> SearchIntent | None:
    """Return a slightly looser copy of the intent, or None if maximally relaxed.

    Order: loosen colour quantifiers → drop exclusions → lower mood thresholds.
    """
    nxt = intent.model_copy(deep=True)

    loosened = False
    for c in nxt.colors:
        if c.quantifier in _RELAX_LADDER:
            i = _RELAX_LADDER.index(c.quantifier)
            if i < len(_RELAX_LADDER) - 1:
                c.quantifier = _RELAX_LADDER[i + 1]
                loosened = True
    if loosened:
        return nxt

    widened = False
    for c in nxt.colors:
        if c.match == "adjacent":     # broaden a shade to its whole colour family
            c.match = "family"
            widened = True
    if widened:
        return nxt

    if nxt.exclude:
        nxt.exclude = []
        return nxt

    lowered = False
    for m in nxt.moods:
        if m.min_coverage > 0.2:
            m.min_coverage = round(m.min_coverage - 0.15, 2)
            lowered = True
    if lowered:
        return nxt

    return None


def _describe_relaxation(original: SearchIntent, final: SearchIntent) -> list[str]:
    """Human-readable net difference between the asked-for intent and what ran."""
    notes: list[str] = []
    for a, b in zip(original.colors, final.colors):
        label = a.color_name or "colour"
        if a.quantifier != b.quantifier:
            notes.append(f"loosened '{label}' ({a.quantifier} → {b.quantifier})")
        if a.match != b.match:
            notes.append(f"broadened '{label}' to its whole colour family")
    if original.exclude and not final.exclude:
        dropped = ", ".join(str(e.family or e.color_name) for e in original.exclude)
        notes.append(f"dropped the exclusion ({dropped})")
    for a, b in zip(original.moods, final.moods):
        if a.min_coverage != b.min_coverage:
            notes.append(f"lowered mood coverage ({a.min_coverage} → {b.min_coverage})")
    return notes


def search(query: str, model: str, host: str, show_intent: bool = False,
           show_cypher: bool = False, relax: bool = True) -> tuple[list[dict], list[str]]:
    """Translate, run, and (optionally) progressively relax until results appear.

    Returns (rows, relaxation_notes). notes is empty when the original query matched.
    """
    vocab = build_vocabulary()
    original = translate(query, model, host)
    if show_intent:
        print("\nIntent:", original.model_dump_json(indent=2, exclude_defaults=True), "\n")

    target = max(1, original.limit)   # keep widening until we can fill the requested count
    attempt, rows, cypher = original, [], ""
    for _ in range(8):  # bounded: ladder (≤3) + exclusions + moods
        cypher, params = build_query(attempt, vocab)
        rows = run_cypher(cypher, params)
        if len(rows) >= target or not relax:
            break
        nxt = _relax_once(attempt)
        if nxt is None:
            break
        attempt = nxt

    if show_cypher:
        print("Cypher (executed):\n" + cypher + "\n")
    return rows, _describe_relaxation(original, attempt)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _print_results(query: str, rows: list[dict], notes: list[str]) -> None:
    if notes:
        print("\nNo exact matches — relaxed the search: " + "; ".join(notes))
    if not rows:
        msg = "Still no matches even after relaxing." if notes else "No matches."
        print(msg + " Try a broader colour or mood.")
        return
    header = "closest after relaxing" if notes else f"{len(rows)} result(s)"
    print(f'\n"{query}" — {header}:\n')
    for r in rows:
        print(f"  {r['painting']:<28} {r['score']:>5}%   {r['path']}")
    print()


def main() -> None:
    load_dotenv(_PROJECT_ROOT / ".env")
    ap = argparse.ArgumentParser(description="Natural-language painting search")
    ap.add_argument("query", nargs="*", help="Free-text request (omit for interactive mode)")
    ap.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "llama3.1:8b"))
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    ap.add_argument("--limit", type=int, default=None, help="Override result count")
    ap.add_argument("--show-intent", action="store_true", help="Print the parsed intent")
    ap.add_argument("--show-cypher", action="store_true", help="Print the generated Cypher")
    ap.add_argument("--no-relax", action="store_true",
                    help="Disable progressive relaxation on empty results")
    args = ap.parse_args()

    def one(q: str) -> None:
        rows, notes = search(q, args.model, args.host, args.show_intent,
                             args.show_cypher, relax=not args.no_relax)
        _print_results(q, rows, notes)

    if args.query:
        q = " ".join(args.query)
        if args.limit:
            q += f" ({args.limit} paintings)"
        one(q)
        return

    print("Painting search — type a request, or 'q' to quit.")
    while True:
        try:
            q = input("\n› ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"q", "quit", "exit"}:
            break
        if q:
            one(q)


if __name__ == "__main__":
    main()
