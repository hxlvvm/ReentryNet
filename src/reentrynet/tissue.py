"""Random scar, border-zone, isthmus and fibrosis patterns for 2D tissue sheets."""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def _smooth_field(rng, shape, sigma):
    return ndimage.gaussian_filter(rng.normal(size=shape), sigma, mode="wrap")


def _strokes(rng, shape, n, width):
    m = np.zeros(shape, bool)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    for _ in range(n):
        pts = rng.uniform([10, 20], [shape[0] - 10, shape[1] - 10], size=(rng.integers(2, 5), 2))
        for (y0, x0), (y1, x1) in zip(pts[:-1], pts[1:]):
            for s in np.linspace(0, 1, 60):
                m |= (yy - (y0 + s * (y1 - y0))) ** 2 + (xx - (x0 + s * (x1 - x0))) ** 2 <= (width / 2) ** 2
    return m


GENERATOR = {"scar": (0.04, 0.25), "bz": (0.1, 0.5), "channels": (0, 3), "fibrosis_p": 0.9, "fib_density": (0.1, 0.45)}


def _rings(rng, shape):
    """Annular scars with 0-2 gaps: a held-out shape family, never used for training."""
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    m = np.zeros(shape, bool)
    for _ in range(rng.integers(1, 3)):
        cy, cx = rng.uniform(30, shape[0] - 30), rng.uniform(35, shape[1] - 25)
        r, w = rng.uniform(10, 25), rng.uniform(3, 7)
        d = np.hypot(yy - cy, xx - cx)
        ring = np.abs(d - r) <= w / 2
        ang = np.arctan2(yy - cy, xx - cx)
        for _ in range(rng.integers(0, 3)):
            a0, half = rng.uniform(-np.pi, np.pi), rng.uniform(0.15, 0.35)
            ring &= np.abs(np.angle(np.exp(1j * (ang - a0)))) > half
        m |= ring
    return m


def make_tissue(rng: np.random.Generator, shape=(128, 128), family: str = "blobs", **override) -> np.ndarray:
    """Conductivity map: 1 healthy, 0 scar, in between for border zone, isthmus channels and fibrosis."""
    g = {**GENERATOR, **override}
    c = np.ones(shape, np.float32)
    if family == "blobs":
        field = _smooth_field(rng, shape, rng.uniform(5, 12))
        scar = field > np.quantile(field, 1 - rng.uniform(*g["scar"]))
    elif family == "strokes":
        scar = _strokes(rng, shape, rng.integers(1, 4), rng.uniform(4, 10))
    elif family == "rings":
        scar = _rings(rng, shape)
    else:
        raise ValueError(f"unknown family {family}")
    bz_cond = rng.uniform(*g["bz"])
    border = ndimage.binary_dilation(scar, iterations=int(rng.integers(1, 5))) & ~scar
    c[border] = bz_cond
    for _ in range(rng.integers(*g["channels"])):
        ys, xs = np.nonzero(scar)
        if len(ys) == 0:
            break
        k = rng.integers(len(ys))
        angle = rng.uniform(0, np.pi)
        line = np.zeros(shape, bool)
        for s in np.linspace(-40, 40, 160):
            y, x = int(ys[k] + s * np.sin(angle)), int(xs[k] + s * np.cos(angle))
            if 0 <= y < shape[0] and 0 <= x < shape[1]:
                line[max(0, y - 1): y + 2, max(0, x - 1): x + 2] = True
        scar &= ~line
        c[line & (c < 1)] = bz_cond
    c[scar] = 0.0
    if rng.random() < g["fibrosis_p"]:
        field = _smooth_field(rng, shape, 10)
        region = field > np.quantile(field, 0.6)
        holes = region & (rng.random(shape) < rng.uniform(*g["fib_density"]))
        c[holes] = 0.0
    c[:, :8] = 1.0
    return c
