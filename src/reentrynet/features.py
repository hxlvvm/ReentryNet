"""Global tissue descriptors for the baseline models."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

NAMES = ["scar_fraction", "border_zone_fraction", "scar_components", "largest_component", "small_components",
         "scar_perimeter", "border_zone_mean_conductivity"]


def features(c: np.ndarray) -> np.ndarray:
    scar = c == 0
    bz = (c > 0) & (c < 1)
    lab, n = ndimage.label(scar)
    sizes = np.bincount(lab.ravel())[1:] if n else np.zeros(1)
    perimeter = (scar ^ ndimage.binary_erosion(scar)).sum()
    return np.array([scar.mean(), bz.mean(), n, sizes.max() / scar.size, (sizes < 5).sum(), perimeter / scar.size,
                     c[bz].mean() if bz.any() else 1.0], np.float32)
