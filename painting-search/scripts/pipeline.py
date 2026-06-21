"""
Extract dominant colors from painting images and store them in Neo4j.

Configuration lives in config.toml at the project root. CLI flags override
individual config values when provided.

Usage:
    python scripts/pipeline.py --benchmark
    python scripts/pipeline.py --profile
    python scripts/pipeline.py --images-dir images/images/
    python scripts/pipeline.py --images-dir images/images/ --sample-size 100000
"""
import argparse
import os
import random
import time
import tomllib
from contextlib import contextmanager
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sklearn.cluster import KMeans
from tqdm import tqdm

from colors import build_vocabulary, rgb_to_lab
from extract import extract_single, process_chunk, to_pixel_array, sample_from_array, nearest_color_idx
from writer import setup_graph, write_paintings

_PROJECT_ROOT = Path(__file__).parent.parent


def load_config() -> dict:
    with open(_PROJECT_ROOT / "config.toml", "rb") as f:
        return tomllib.load(f)


def _pick_images(images_dir: Path, extensions: tuple[str, ...], n: int,
                 select: str) -> list[Path] | None:
    all_paths = sorted(p for ext in extensions for p in images_dir.rglob(ext))
    if not all_paths:
        print(f"No images found in {images_dir}")
        print("Download the dataset first:")
        print("  kaggle datasets download -d programmer3/artistic-painting-dataset -p images/ --unzip")
        print("Then run with: --images-dir images/images/")
        return None
    if select == "random":
        return random.sample(all_paths, min(n, len(all_paths)))
    return all_paths[:n]


# ── Benchmark ─────────────────────────────────────────────────────────────────

def run_benchmark(images_dir: Path, n_colors: int, vocab: dict, cfg_benchmark: dict,
                  extensions: tuple[str, ...], select: str) -> None:
    n_images    = cfg_benchmark["n_images"]
    sample_sizes = [s if s != 0 else 10 ** 9 for s in cfg_benchmark["sample_sizes"]]
    size_labels  = [f"{s:,}" if s < 10 ** 9 else "full res" for s in sample_sizes]

    paths = _pick_images(images_dir, extensions, n_images, select)
    if paths is None:
        return

    vocab_labs  = vocab["labs"]
    vocab_names = vocab["names"]

    print(f"\nBenchmarking on {len(paths)} image(s): {[p.name for p in paths]}\n")
    print(f"  {'Sample':>10}  {'Avg time':>10}  Top 3 colors (last image)")
    print("  " + "-" * 65)

    for size, label in zip(sample_sizes, size_labels):
        t0 = time.perf_counter()
        top_colors = []
        for path in paths:
            result = process_chunk([str(path)], size, n_colors, vocab_labs, vocab_names)
            if result:
                top_colors = [c["name"] for c in result[0]["colors"][:3]]
        avg = (time.perf_counter() - t0) / len(paths)
        print(f"  {label:>10}  {avg:>9.2f}s  {', '.join(top_colors)}")

    print()


# ── Timing ────────────────────────────────────────────────────────────────────

@contextmanager
def _timed(accum, key):
    t = time.perf_counter()
    yield
    accum[key] += time.perf_counter() - t


# ── Profile ───────────────────────────────────────────────────────────────────

def run_profile(images_dir: Path, sample_size: int, n_colors: int, vocab: dict,
                extensions: tuple[str, ...], n_runs: int, select: str) -> None:
    picked = _pick_images(images_dir, extensions, 1, select)
    if picked is None:
        return
    path = picked[0]

    vocab_labs  = vocab["labs"]
    vocab_names = vocab["names"]

    accum = {"open": 0.0, "to_array": 0.0, "sample": 0.0,
             "kmeans": 0.0, "to_lab": 0.0, "nearest": 0.0}

    total_pixels = Image.open(path).size[0] * Image.open(path).size[1]

    for _ in range(n_runs):
        with _timed(accum, "open"):
            img = Image.open(path).convert("RGB")

        with _timed(accum, "to_array"):
            pixels = to_pixel_array(img)

        with _timed(accum, "sample"):
            pixels = sample_from_array(pixels, sample_size)

        with _timed(accum, "kmeans"):
            km = KMeans(n_clusters=n_colors, n_init="auto", random_state=42).fit(pixels)

        counts = np.bincount(km.labels_, minlength=n_colors)
        order  = np.argsort(-(counts / counts.sum()))

        with _timed(accum, "to_lab"):
            labs = [rgb_to_lab(km.cluster_centers_[i]) for i in order]

        with _timed(accum, "nearest"):
            [nearest_color_idx(lab, vocab_labs) for lab in labs]

    avgs    = {k: v / n_runs for k, v in accum.items()}
    total   = sum(avgs.values())
    sampled = min(total_pixels, sample_size)
    w, h    = Image.open(path).size

    print(f"\nProfiling: {path.name}  ({w}×{h}, {total_pixels:,} pixels → {sampled:,} sampled)")
    print(f"Averaged over {n_runs} run(s)  |  sample_size={sample_size:,}  n_colors={n_colors}\n")

    col  = 36
    rows = [
        ("open + decode",                    avgs["open"]),
        ("pixel array (np.array)",           avgs["to_array"]),
        (f"random sampling ({sampled:,} px)", avgs["sample"]),
        (f"KMeans fit ({n_colors} clusters)", avgs["kmeans"]),
        (f"centroid → LAB (×{n_colors})",    avgs["to_lab"]),
        (f"nearest color (×{n_colors})",     avgs["nearest"]),
    ]

    print(f"  {'Step':<{col}}  {'Avg time':>10}  {'% total':>8}")
    print("  " + "─" * (col + 23))
    for label, t in rows:
        pct = t / total * 100 if total > 0 else 0
        print(f"  {label:<{col}}  {t:>9.3f}s  {pct:>7.1f}%")
    print("  " + "─" * (col + 23))
    print(f"  {'total (per image)':<{col}}  {total:>9.3f}s  {'100.0':>7}%")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv()
    cfg = load_config()

    parser = argparse.ArgumentParser(description="Extract painting colors and load into Neo4j")
    parser.add_argument("--images-dir",   type=Path, default=Path("images"),
                        help="Folder containing painting images")
    parser.add_argument("--sample-size",  type=int,  default=cfg["sample_size"],
                        help=f"Random pixels to sample per image (config default: {cfg['sample_size']})")
    parser.add_argument("--n-colors",     type=int,  default=cfg["n_colors"],
                        help=f"Dominant colors to extract per painting (config default: {cfg['n_colors']})")
    parser.add_argument("--benchmark",    action="store_true",
                        help="Time extraction across sample sizes on a few images, then exit")
    parser.add_argument("--profile",      action="store_true",
                        help="Print per-step timing breakdown for one image, then exit")
    parser.add_argument("--profile-runs", type=int, default=5,
                        help="Repetitions to average in --profile mode (default: 5)")
    parser.add_argument("--image-select", choices=["first", "random"], default="first",
                        help="Image selection strategy: 'first' (default) or 'random'")
    parser.add_argument("--limit",        type=str, default=None,
                        help="Process a subset of images: '10' for 10 images, '20%%' for 20 percent")
    args = parser.parse_args()

    print("Building color vocabulary...")
    vocab = build_vocabulary()
    print(f"  {len(vocab['names'])} named colors loaded")

    extensions = tuple(cfg["image_extensions"])

    if args.benchmark:
        run_benchmark(args.images_dir, args.n_colors, vocab, cfg["benchmark"],
                      extensions, args.image_select)
        return

    if args.profile:
        run_profile(args.images_dir, args.sample_size, args.n_colors, vocab,
                    extensions, args.profile_runs, args.image_select)
        return

    image_paths = sorted(str(p) for ext in extensions for p in args.images_dir.rglob(ext))
    if not image_paths:
        print(f"No images found in {args.images_dir}")
        return

    if args.limit is not None:
        if args.limit.endswith("%"):
            n = max(1, int(len(image_paths) * float(args.limit[:-1]) / 100))
        else:
            n = int(args.limit)
        image_paths = (random.sample(image_paths, min(n, len(image_paths)))
                       if args.image_select == "random" else image_paths[:n])

    print(f"Found {len(image_paths)} images")

    nworkers   = os.cpu_count() or 1
    batch_size = cfg["neo4j_batch_size"]

    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
    )

    extract_fn = partial(
        extract_single,
        sample_size=args.sample_size,
        n_colors=args.n_colors,
        vocab_labs=vocab["labs"],
        vocab_names=vocab["names"],
    )

    with driver.session() as session:
        print("Setting up color nodes and vector index...")
        setup_graph(session, vocab, cfg["color_batch_size"])

        print(f"Extracting with {nworkers} workers, writing every {batch_size} paintings...")
        t0        = time.perf_counter()
        batch:    list[dict] = []
        n_written = 0

        with Pool(processes=nworkers) as pool:
            for result in tqdm(pool.imap_unordered(extract_fn, image_paths),
                               total=len(image_paths), unit="img"):
                if result is None:
                    continue
                batch.append(result)
                if len(batch) >= batch_size:
                    write_paintings(session, batch, batch_size)
                    n_written += len(batch)
                    batch.clear()

        if batch:
            write_paintings(session, batch, batch_size)
            n_written += len(batch)

    driver.close()
    elapsed = time.perf_counter() - t0
    print(f"Done. {n_written} paintings stored in Neo4j in {elapsed:.1f}s ({n_written / elapsed:.1f} img/s)")


if __name__ == "__main__":
    main()
