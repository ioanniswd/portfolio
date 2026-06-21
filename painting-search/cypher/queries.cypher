// ── painting-search: example Cypher queries ───────────────────────────────────
//
// All queries assume the Neo4j graph has been populated by pipeline.py.
// The vector index 'color_lab_index' on Color.lab_vector enables proximity
// search in CIE LAB space without pre-computed ADJACENT_TO relationships.
//
// Run these in Neo4j Browser or via the neo4j Python driver.
// ─────────────────────────────────────────────────────────────────────────────


// ── 1. Exact match: paintings with white AND gold in top 3 colors ─────────────
//
// Use this when the user names specific colors and wants both present.
// Adjust rank threshold and/or add a percentage floor as needed.

MATCH (p:Painting)-[h1:HAS_COLOR]->(c1:Color {name: "white"}),
      (p)-[h2:HAS_COLOR]->(c2:Color {name: "gold"})
WHERE h1.rank <= 3 AND h2.rank <= 3
RETURN
  p.name                              AS painting,
  round(h1.percentage * 100, 1)      AS white_pct,
  round(h2.percentage * 100, 1)      AS gold_pct,
  round((h1.percentage + h2.percentage) * 100, 1) AS combined_pct
ORDER BY combined_pct DESC;


// ── 2. Dominant color only (rank = 1) ────────────────────────────────────────
//
// The single most prevalent color in the painting.

MATCH (p:Painting)-[h:HAS_COLOR {rank: 1}]->(c:Color)
WHERE c.name IN ["white", "titanium white", "zinc white", "ivory"]
RETURN p.name AS painting, c.name AS dominant_color, round(h.percentage * 100, 1) AS pct
ORDER BY pct DESC;


// ── 3. Perceptual adjacency via vector KNN ────────────────────────────────────
//
// Finds the 10 most perceptually similar colors to a named color in CIE LAB
// space, then aggregates paintings that contain any of those colors.
// This is how "blue" returns petrol, navy, sky blue, cerulean, etc.

MATCH (target:Color {name: "blue"})
CALL db.index.vector.queryNodes('color_lab_index', 10, target.lab_vector)
YIELD node AS similar_color, score
MATCH (p:Painting)-[h:HAS_COLOR]->(similar_color)
WITH p, sum(h.percentage) AS blue_family_pct, collect(similar_color.name) AS matched_colors
RETURN
  p.name                                    AS painting,
  round(blue_family_pct * 100, 1)          AS blue_family_pct,
  matched_colors
ORDER BY blue_family_pct DESC
LIMIT 20;


// ── 4. Free-form color: search by LAB vector ──────────────────────────────────
//
// When the user describes a color that may not match a node name exactly,
// compute its LAB vector externally (or via the SKILL) and pass it as a
// parameter. Claude computes this vector and substitutes it for $lab_vector.
//
// Example: user says "salmon" → Claude converts to LAB ≈ [60, 35, 20]

CALL db.index.vector.queryNodes('color_lab_index', 5, $lab_vector)
YIELD node AS color, score
MATCH (p:Painting)-[h:HAS_COLOR]->(color)
RETURN
  p.name        AS painting,
  color.name    AS matched_color,
  round(h.percentage * 100, 1) AS pct,
  round(score, 3) AS similarity
ORDER BY h.percentage DESC
LIMIT 20;


// ── 5. Color family / mood search ────────────────────────────────────────────
//
// Aggregate by color family for broad mood queries:
// "warm", "cool", "earthy", "dark and dramatic", etc.

MATCH (p:Painting)-[h:HAS_COLOR]->(c:Color)-[:IN_FAMILY]->(f:ColorFamily)
WHERE f.name IN ["Yellows & Golds", "Oranges & Corals", "Browns & Earth Tones", "Reds & Pinks"]
WITH p, sum(h.percentage) AS warm_pct
WHERE warm_pct > 0.4
RETURN p.name AS painting, round(warm_pct * 100, 1) AS warm_pct
ORDER BY warm_pct DESC;


// ── 6. Cool / blue-green mood ─────────────────────────────────────────────────

MATCH (p:Painting)-[h:HAS_COLOR]->(c:Color)-[:IN_FAMILY]->(f:ColorFamily)
WHERE f.name IN ["Blues", "Greens", "Purples & Violets"]
WITH p, sum(h.percentage) AS cool_pct
WHERE cool_pct > 0.5
RETURN p.name AS painting, round(cool_pct * 100, 1) AS cool_pct
ORDER BY cool_pct DESC;


// ── 7. Exclude a color ────────────────────────────────────────────────────────
//
// Gold-dominant paintings that contain no red at all.

MATCH (p:Painting)-[h:HAS_COLOR]->(c:Color)
WHERE c.name = "gold" AND h.rank <= 2
  AND NOT EXISTS {
    MATCH (p)-[:HAS_COLOR]->(red:Color)
    WHERE red.family = "Reds & Pinks"
  }
RETURN p.name AS painting, round(h.percentage * 100, 1) AS gold_pct
ORDER BY gold_pct DESC;


// ── 8. High contrast: dark + bright ──────────────────────────────────────────
//
// Paintings that have both very dark colors AND a bright accent.

MATCH (p:Painting)-[:HAS_COLOR]->(dark:Color),
      (p)-[h_bright:HAS_COLOR]->(bright:Color)
WHERE dark.family IN ["Blacks & Darks", "Grays & Neutrals"] AND dark.l < 25
  AND bright.l > 70 AND h_bright.percentage > 0.05
RETURN DISTINCT p.name AS painting, dark.name AS dark_color, bright.name AS bright_color
LIMIT 20;


// ── 9. Inspect a specific painting ───────────────────────────────────────────

MATCH (p:Painting {name: $painting_name})-[h:HAS_COLOR]->(c:Color)-[:IN_FAMILY]->(f:ColorFamily)
RETURN
  h.rank        AS rank,
  c.name        AS color,
  c.hex         AS hex,
  f.name        AS family,
  round(h.percentage * 100, 1) AS pct
ORDER BY h.rank;


// ── 10. List all color families and their size ───────────────────────────────

MATCH (c:Color)-[:IN_FAMILY]->(f:ColorFamily)
RETURN f.name AS family, count(c) AS n_colors
ORDER BY n_colors DESC;
