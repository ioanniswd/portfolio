# Painting Search

Search a collection of paintings by color. Dominant colors are extracted from each image using K-Means clustering on randomly sampled pixels, then stored in Neo4j as a color knowledge graph. Queries support both exact color matches and perceptual adjacency — asking for "blue" returns paintings with navy, petrol, cerulean, and sky blue as well, using Neo4j's native vector index on CIE LAB color space.

The color vocabulary covers ~960 named colors: the full [xkcd color survey](https://xkcd.com/color/rgb/) plus classical art pigment names (burnt sienna, ultramarine, viridian, etc.).

## Data

Download the [Artistic Painting Dataset](https://www.kaggle.com/datasets/programmer3/artistic-painting-dataset) from Kaggle and extract it into `images/`:

```bash
kaggle datasets download -d programmer3/artistic-painting-dataset -p images/ --unzip
```

The dataset extracts into `images/images/`, organized by artist (20 artists, ~4,100 paintings). The script scans subdirectories recursively, so `--images-dir images/images/` is all you need.

## Setup

**1. Start Neo4j**

The vector index this project uses requires Neo4j 5.11+. The quickest way to run it locally is Docker:

```bash
docker run \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=none \
  -v $HOME/neo4j/data:/data \
  neo4j:latest
```

The `-v` flag mounts `~/neo4j/data` on your machine to Neo4j's data directory inside the container. All nodes, relationships, and vector indexes are persisted there — removing the container does not lose your data.

The browser UI is available at `http://localhost:7474`. To stop and restart later:

```bash
docker stop neo4j
docker start neo4j
```

**2. Install dependencies**

```bash
uv sync
```

**3. Create a `.env` file** from the example and fill in your Neo4j credentials:

```bash
cp .env.example .env
```

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=
OLLAMA_MODEL=llama3.1:8b
OLLAMA_HOST=http://localhost:11434
```

**4. Install [Ollama](https://ollama.com)** (only needed for natural-language search) and pull the model:

```bash
ollama pull llama3.1:8b
```

From then on `make up` starts both Neo4j and Ollama, and `make search Q="…"` runs a query — see [Natural-language search](#natural-language-search).

## Usage

**Step 1 — Benchmark** to find the right `sample_size` for your machine before processing the full dataset. The benchmark runs a few images across a range of pixel sample counts and prints timing alongside the extracted colors, so you can see where the results stabilize:

```bash
uv run python scripts/pipeline.py --images-dir images/images/ --benchmark
```

By default the benchmark uses the first N images (alphabetically). Pass `--image-select random` to sample randomly instead:

```bash
uv run python scripts/pipeline.py --images-dir images/images/ --benchmark --image-select random
```

Once you have a good `sample_size`, update it in `config.toml`.

**Step 2 — Extract colors and populate Neo4j:**

```bash
uv run python scripts/pipeline.py --images-dir images/images/
```

To process only a subset — useful for testing end-to-end before committing to the full dataset:

```bash
# First 10 images
uv run python scripts/pipeline.py --images-dir images/images/ --limit 10

# Random 20%
uv run python scripts/pipeline.py --images-dir images/images/ --limit 20% --image-select random
```

Individual config values can be overridden per-run with CLI flags:

| Flag | Config key | Description |
|---|---|---|
| `--images-dir` | — | Folder containing painting images |
| `--sample-size` | `sample_size` | Random pixels sampled per image |
| `--n-colors` | `n_colors` | Dominant colors extracted per painting |
| `--image-select` | — | Image selection strategy: `first` (default) or `random` |
| `--limit` | — | Process a subset: `10` for 10 images, `20%` for 20 percent |
| `--benchmark` | — | Run benchmark then exit |
| `--profile` | — | Print per-step timing breakdown for one image, then exit |
| `--profile-runs` | — | Repetitions to average in profile mode (default: 5) |

## Configuration

All tunable parameters live in [`config.toml`](config.toml), each with an inline description:

| Key | Default | Description |
|---|---|---|
| `sample_size` | `50000` | Pixels randomly sampled per image before clustering |
| `n_colors` | `8` | K-Means clusters (dominant colors) per painting |
| `image_extensions` | `jpg, jpeg, png` | File types scanned in `--images-dir` |
| `neo4j_batch_size` | `100` | Painting nodes written per Neo4j UNWIND batch |
| `color_batch_size` | `200` | Color nodes written per Neo4j UNWIND batch |
| `benchmark.n_images` | `3` | Images used during the benchmark |
| `benchmark.sample_sizes` | `10k … full` | Sample sizes tested by the benchmark (`0` = full pixel count) |

## Natural-language search

Two ways to search the collection in plain English:

### 1. Local LLM funnel — fully local, no paid API (recommended)

[`scripts/search.py`](scripts/search.py) is the full production funnel powered by a **local [Ollama](https://ollama.com) model** — no cloud, no API key, no per-query cost. The model decomposes your request into a structured `SearchIntent`; deterministic Python resolves colours in CIE LAB space and composes a parameterized Cypher query. **The LLM never writes Cypher** — it only fills a schema — so a small local model (default `llama3.1:8b`) is enough, and there's no injection surface.

Start the services and search via the [Makefile](Makefile):

```bash
make up                                    # start Neo4j + Ollama
make search Q="find paintings with the color of sand"
make search Q="paintings with both blue and green"
make search Q="a lot of blue, but no red"
make search Q="something warm and earthy" FLAGS="--show-intent --show-cypher"
make search                                # no Q → interactive REPL
make down                                  # stop everything
```

(Equivalently: `ollama pull llama3.1:8b`, then `uv run python scripts/search.py "<query>"`.)

What it handles:

- **Colour breadth** — a specific shade (`burgundy`, `deep ocean water`), a shade plus its neighbours, or a whole **family** (`blue`, `reds`) — chosen per request.
- **Typos & off-vocabulary colours** — `burgandy` → `burgundy`, `sand` → the real tan; matched against the ~960 named colours (fuzzy), with the model's RGB estimate as a fallback.
- **Multiple + negative constraints together** — "predominantly gold, no red" compiles include *and* exclude blocks into one query.
- **Moods, lightness, contrast** — "warm and earthy", "dark and dramatic".
- **Progressive relaxation** — if nothing matches, it widens the search one notch at a time (and reports how) instead of returning nothing.

Inspect the internals with `--show-intent` (the parsed JSON) and `--show-cypher` (the generated query). See the [`Makefile`](Makefile) for all service targets.

### 2. Claude Skill (interactive)

This project also includes a [`SKILL.md`](SKILL.md) for use with Claude Code or any agent that supports skill files. Load it and the agent translates your request into a Cypher query, runs it, and displays matching paintings with images inline.

```
Find me paintings that feel warm and earthy
Show me something with the color of deep ocean water
I want a painting that is predominantly white and gold
```

## Examples

All produced by the local Ollama funnel (`scripts/search.py`) against the graph.

**"Find paintings with the color of sand"** — a single, off-vocabulary colour

<img src="assets/examples/sand_gauguin_31.jpg" width="175"> <img src="assets/examples/sand_gauguin_282.jpg" width="175"> <img src="assets/examples/sand_kandinsky_18.jpg" width="175"> <img src="assets/examples/sand_vangogh_490.jpg" width="175">

*Gauguin 31 · Gauguin 282 · Kandinsky 18 · Van Gogh 490*

---

**"Paintings with both blue and green"** — two colours, both required

<img src="assets/examples/bluegreen_vangogh_448.jpg" width="175"> <img src="assets/examples/bluegreen_magritte_84.jpg" width="175"> <img src="assets/examples/bluegreen_cezanne_24.jpg" width="175"> <img src="assets/examples/bluegreen_vangogh_317.jpg" width="175">

*Van Gogh 448 · Magritte 84 · Cézanne 24 · Van Gogh 317*

---

**"A lot of blue, but no red"** — one colour included, a whole family excluded

<img src="assets/examples/bluenored_magritte_71.jpg" width="175"> <img src="assets/examples/bluenored_magritte_4.jpg" width="175"> <img src="assets/examples/bluenored_magritte_148.jpg" width="175"> <img src="assets/examples/bluenored_magritte_170.jpg" width="175">

*Magritte 71 · Magritte 4 · Magritte 148 · Magritte 170*

---

**"Something dark and moody"** — a mood, matched by overall lightness

<img src="assets/examples/dark_rembrandt_96.jpg" width="175"> <img src="assets/examples/dark_rembrandt_111.jpg" width="175"> <img src="assets/examples/dark_titian_240.jpg" width="175"> <img src="assets/examples/dark_bruegel_47.jpg" width="175">

*Rembrandt 96 · Rembrandt 111 · Titian 240 · Bruegel 47*

---

## Queries

See [`cypher/queries.cypher`](cypher/queries.cypher) for annotated examples including:

- Exact multi-color match ("white AND gold in top 3")
- Perceptual adjacency via vector KNN ("blue and similar")
- Mood/family search ("warm", "cool", "earthy")
- Dominant color, high contrast, and exclusion queries
