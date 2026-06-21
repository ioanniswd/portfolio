---
name: painting-search
description: Search a Neo4j painting collection by color using natural language. Translates
  free-text color requests into Cypher queries with exact matching or perceptual
  adjacency via CIE LAB vector search. Use whenever a user asks for paintings by
  color, mood, or palette.
---

# Painting Search Skill

Translate free-text color requests into Cypher queries against the painting-search Neo4j graph and return matching paintings with color breakdowns.

---

## Graph Schema

```
(:Painting {name, path, width, height})
    -[:HAS_COLOR {percentage: float, rank: int}]->
(:Color {name, hex, r, g, b, l, a_lab, b_lab, lab_vector: [L, a, b], family})
    -[:IN_FAMILY]->
(:ColorFamily {name})
```

Vector index `color_lab_index` is defined on `Color.lab_vector` (dimensions: 3, cosine similarity). Use it to find perceptually adjacent colors without pre-computed relationships.

---

## Color Families

| Family | Typical colors |
|---|---|
| Whites & Creams | white, ivory, cream, titanium white, zinc white |
| Yellows & Golds | gold, yellow, naples yellow, cadmium yellow, yellow ochre |
| Oranges & Corals | orange, coral, peach, burnt sienna, raw sienna |
| Reds & Pinks | red, crimson, rose, pink, alizarin crimson, cadmium red |
| Browns & Earth Tones | brown, raw umber, burnt umber, van dyke brown, sienna, tan |
| Greens | green, viridian, olive, sage, forest green, terre verte |
| Blues | blue, navy, cerulean blue, prussian blue, ultramarine, petrol |
| Purples & Violets | purple, violet, lavender, indigo, mauve |
| Grays & Neutrals | gray, silver, beige, taupe |
| Blacks & Darks | black, ivory black, lamp black, charcoal |

---

## Translating Natural Language

### Quantity words

| User says | Cypher filter |
|---|---|
| "predominantly X" | `h.rank <= 2 AND h.percentage > 0.30` |
| "mostly X" | `h.rank <= 2` |
| "a lot of X" | `h.rank <= 3` |
| "some X" / "touches of X" | `h.rank <= 5 AND h.percentage < 0.15` |
| "X as the main color" | `h.rank = 1` |

### Adjacency words

| User says | Strategy |
|---|---|
| "blue or something like it" | vector KNN on blue's lab_vector, k=10 |
| "blue and related colors" | vector KNN, k=15 |
| "exactly blue" | exact `{name: "blue"}` match |
| "warm colors" | family filter: Yellows, Oranges, Reds, Browns |
| "cool colors" | family filter: Blues, Greens, Purples |
| "earthy" | family filter: Browns & Earth Tones, Yellows & Golds |
| "dark and dramatic" | `c.l < 30` LAB lightness threshold |
| "light / airy" | `c.l > 75` LAB lightness threshold |

### Color name resolution

If the user names a color not in the vocabulary (e.g., "salmon", "eggshell", "petrol"):

1. Estimate its approximate RGB from knowledge (e.g., salmon ≈ `[250, 128, 114]`)
2. Convert to CIE LAB using the formula below
3. Use query pattern 4 (free-form LAB vector search) with `$lab_vector`

**RGB → CIE LAB conversion** (D65 illuminant):
```
norm   = rgb / 255
linear = (norm > 0.04045) ? ((norm + 0.055) / 1.055)^2.4 : norm / 12.92
xyz    = [[0.4124564, 0.3575761, 0.1804375],
          [0.2126729, 0.7151522, 0.0721750],
          [0.0193339, 0.1191920, 0.9503041]] @ linear
xyz_n  = xyz / [0.95047, 1.00000, 1.08883]
f      = (xyz_n > 0.008856) ? xyz_n^(1/3) : (903.3*xyz_n + 16)/116
L = 116*f[1] - 16
a = 500*(f[0] - f[1])
b = 200*(f[1] - f[2])
```

---

## Query Templates

### Template 1 — Family-based multi-color

Aggregate each family's total coverage per painting separately, then filter for paintings that have enough of both. Avoids cartesian products that inflate percentages.

```cypher
MATCH (p:Painting)-[h:HAS_COLOR]->(c:Color)-[:IN_FAMILY]->(f:ColorFamily)
WHERE f.name IN [$family1, $family2]
WITH p, f.name AS family, sum(h.percentage) AS family_pct
WITH p,
     sum(CASE WHEN family = $family1 THEN family_pct ELSE 0 END) AS pct1,
     sum(CASE WHEN family = $family2 THEN family_pct ELSE 0 END) AS pct2
WHERE pct1 > $threshold AND pct2 > $threshold
RETURN p.name AS painting, p.path,
       round(pct1 * 100, 1) AS color1_pct,
       round(pct2 * 100, 1) AS color2_pct
ORDER BY (pct1 + pct2) DESC
LIMIT 20
```

Use `$threshold = 0.15` for "mostly", `0.10` for broader results.

### Template 2 — Single color + perceptual adjacency

```cypher
MATCH (target:Color {name: $color_name})
CALL db.index.vector.queryNodes('color_lab_index', $k, target.lab_vector)
YIELD node AS similar_color, score
MATCH (p:Painting)-[h:HAS_COLOR]->(similar_color)
WITH p, sum(h.percentage) AS family_pct, collect(similar_color.name) AS matched
RETURN p.name AS painting,
       round(family_pct * 100, 1) AS coverage_pct,
       matched
ORDER BY family_pct DESC
LIMIT 20
```

Use `$k = 10` for "and similar", `$k = 20` for broader adjacency.

### Template 3 — Free-form LAB vector (off-vocabulary colors)

Use when the user names a color not in the vocabulary (e.g. "sand", "luxury green", "salmon"). Estimate RGB from knowledge, convert to LAB, query the index directly. Use `$k = 20` as default; increase if fewer than 5 paintings are returned.

```cypher
CALL db.index.vector.queryNodes('color_lab_index', 20, $lab_vector)
YIELD node AS similar_color, score
MATCH (p:Painting)-[h:HAS_COLOR]->(similar_color)
WITH p, sum(h.percentage) AS pct, collect(similar_color.name) AS matched
RETURN p.name AS painting, p.path,
       round(pct * 100, 1) AS coverage_pct,
       matched
ORDER BY pct DESC
LIMIT 20
```

Always use Template 3 (or Template 2) for proximity/adjacency questions — never fall back to exact name matching for these.

### Template 4 — Mood / family

```cypher
MATCH (p:Painting)-[h:HAS_COLOR]->(c:Color)-[:IN_FAMILY]->(f:ColorFamily)
WHERE f.name IN $families
WITH p, sum(h.percentage) AS mood_pct
WHERE mood_pct > $threshold
RETURN p.name AS painting, round(mood_pct * 100, 1) AS mood_pct
ORDER BY mood_pct DESC
LIMIT 20
```

### Template 5 — Exclusion

```cypher
MATCH (p:Painting)-[h:HAS_COLOR]->(c:Color {name: $want})
WHERE h.rank <= $rank
  AND NOT EXISTS {
    MATCH (p)-[:HAS_COLOR]->(x:Color)
    WHERE x.family = $exclude_family
  }
RETURN p.name AS painting, round(h.percentage * 100, 1) AS pct
ORDER BY pct DESC
LIMIT 20
```

---

## Execution

Always run the query and display results — never just print the Cypher. Use the neo4j Python driver with credentials from `.env`:

```python
from dotenv import load_dotenv
from neo4j import GraphDatabase
import os

load_dotenv("painting-search/.env")
uri  = os.environ["NEO4J_URI"]
user = os.environ["NEO4J_USER"]
pwd  = os.environ["NEO4J_PASSWORD"]
auth = (user, pwd) if pwd else None

driver = GraphDatabase.driver(uri, auth=auth)
with driver.session() as session:
    results = session.run(query, **params).data()
driver.close()
```

After running the query, read each returned `p.path` as an image file so it renders visually inline. Paths are relative to the `painting-search/` directory.

---

## Response Format

For each result, display the image and a one-line caption with name and color coverage. Keep the list tight — show at most the number the user asked for, defaulting to 5.

Example for "find me 3 paintings with blue":

> **Piet Mondrian 8** — 88.7% blue  
> [image]
>
> **Marc Chagall 223** — 88.4% blue  
> [image]
>
> **Marc Chagall 214** — 88.3% blue  
> [image]

If no results are found, say so and suggest a broader search (family instead of exact name, or higher k for KNN).

If the user's color maps to multiple close vocabulary terms (e.g., "gold" vs "yellow ochre" vs "naples yellow"), run adjacency search (Template 2) with k=8 rather than exact match to avoid missing paintings.

---

## Setup

Populate the graph before using this skill:

```bash
cd painting-search/
uv run python scripts/pipeline.py --images-dir images/
```

See `README.md` for full setup instructions.
