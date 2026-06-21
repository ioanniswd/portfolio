import colorsys

import numpy as np


_ART_PIGMENTS = {
    "titanium white":   "#FAFAFA",
    "zinc white":       "#F8F8F0",
    "naples yellow":    "#FADA5E",
    "cadmium yellow":   "#FFF600",
    "chrome yellow":    "#FFA700",
    "yellow ochre":     "#C8A951",
    "raw sienna":       "#D2691E",
    "burnt sienna":     "#E97451",
    "cadmium red":      "#E30022",
    "alizarin crimson": "#E32636",
    "ultramarine":      "#3F00FF",
    "prussian blue":    "#003153",
    "cerulean blue":    "#2A52BE",
    "viridian":         "#40826D",
    "terre verte":      "#7C9D77",
    "raw umber":        "#826644",
    "burnt umber":      "#6B3A2A",
    "ivory black":      "#1C1C1C",
    "lamp black":       "#0F0F0F",
    "van dyke brown":   "#664228",
}


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert a single RGB triplet (0–255) to CIE LAB (D65 illuminant)."""
    norm = rgb / 255.0
    linear = np.where(norm > 0.04045, ((norm + 0.055) / 1.055) ** 2.4, norm / 12.92)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = M @ linear
    xyz_n = xyz / np.array([0.95047, 1.00000, 1.08883])
    f = np.where(xyz_n > 0.008856, xyz_n ** (1 / 3), (903.3 * xyz_n + 16) / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def _assign_family(rgb: np.ndarray) -> str:
    r, g, b = rgb / 255.0
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    h_deg = h * 360
    if l > 0.88:
        return "Whites & Creams"
    if l < 0.12:
        return "Blacks & Darks"
    if s < 0.12:
        return "Grays & Neutrals"
    if 10 <= h_deg < 50 and s < 0.55 and l < 0.50:
        return "Browns & Earth Tones"
    if h_deg < 15 or h_deg >= 340:
        return "Reds & Pinks"
    if h_deg < 45:
        return "Oranges & Corals"
    if h_deg < 75:
        return "Yellows & Golds"
    if h_deg < 160:
        return "Greens"
    if h_deg < 260:
        return "Blues"
    if h_deg < 300:
        return "Purples & Violets"
    return "Reds & Pinks"


def build_vocabulary() -> dict:
    """Build the full color vocabulary from XKCD colors + art pigments.

    Returns a dict with parallel arrays: names, hexes, rgbs, labs, families.
    """
    import matplotlib.colors as mcolors

    raw = {name.replace("xkcd:", ""): hex_val for name, hex_val in mcolors.XKCD_COLORS.items()}
    raw.update(_ART_PIGMENTS)

    names, hexes, rgbs, labs, families = [], [], [], [], []
    for name, hex_val in raw.items():
        try:
            r, g, b = mcolors.to_rgb(hex_val)
        except ValueError:
            continue
        rgb = np.array([r * 255, g * 255, b * 255])
        names.append(name)
        hexes.append(hex_val)
        rgbs.append(rgb)
        labs.append(rgb_to_lab(rgb))
        families.append(_assign_family(rgb))

    return {
        "names":    names,
        "hexes":    hexes,
        "rgbs":     np.array(rgbs),
        "labs":     np.array(labs),
        "families": families,
    }
