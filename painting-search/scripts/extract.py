from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from colors import rgb_to_lab


def to_pixel_array(img: Image.Image) -> np.ndarray:
    """Flatten a PIL image into a float32 (N, 3) pixel array."""
    return np.array(img).reshape(-1, 3).astype(np.float32)


def sample_from_array(pixels: np.ndarray, sample_size: int) -> np.ndarray:
    """Randomly subsample pixels to at most sample_size rows."""
    if len(pixels) > sample_size:
        idx = np.random.default_rng().choice(len(pixels), sample_size, replace=False)
        return pixels[idx]
    return pixels


def sample_pixels(img: Image.Image, sample_size: int) -> np.ndarray:
    """Convenience wrapper: flatten image then subsample."""
    return sample_from_array(to_pixel_array(img), sample_size)


def nearest_color_idx(lab_centroid: np.ndarray, vocab_labs: np.ndarray) -> int:
    """Return the index of the closest named color to lab_centroid by Euclidean distance."""
    diffs = vocab_labs - lab_centroid
    return int(np.argmin((diffs ** 2).sum(axis=1)))


def extract_single(
    path_str: str,
    sample_size: int,
    n_colors: int,
    vocab_labs: np.ndarray,
    vocab_names: list[str],
) -> dict | None:
    """Extract dominant colors from one image. Returns None on error.

    Designed for Pool.imap_unordered — one task per image, result streamed back immediately.
    """
    path = Path(path_str)
    try:
        img = Image.open(path).convert("RGB")
        width, height = img.size
        pixels = sample_pixels(img, sample_size)

        km = KMeans(n_clusters=n_colors, n_init="auto", random_state=42)
        km.fit(pixels)

        counts = np.bincount(km.labels_, minlength=n_colors)
        percentages = counts / counts.sum()
        order = np.argsort(-percentages)

        colors = [
            {
                "name":       vocab_names[nearest_color_idx(rgb_to_lab(km.cluster_centers_[idx]), vocab_labs)],
                "percentage": float(percentages[idx]),
                "rank":       rank,
            }
            for rank, idx in enumerate(order, 1)
        ]

        return {
            "path":   path_str,
            "name":   path.stem,
            "width":  width,
            "height": height,
            "colors": colors,
        }
    except Exception as e:
        print(f"  [skip] {path.name}: {e}")
        return None


def process_chunk(
    image_paths: list[str],
    sample_size: int,
    n_colors: int,
    vocab_labs: np.ndarray,
    vocab_names: list[str],
) -> list[dict]:
    """Extract dominant colors from a list of image paths.

    Called via Pool.starmap — must be importable without side effects.
    Returns a list of dicts ready to write to Neo4j.
    """
    return [
        r for path_str in image_paths
        if (r := extract_single(path_str, sample_size, n_colors, vocab_labs, vocab_names)) is not None
    ]
