from functools import cache

import numpy as np
from PIL import Image, ImageFilter

HALFTONE_SHAPES = {
    "circle": "Kreis",
    "ellipse": "Ellipse",
    "cross": "Kreuz",
}


@cache
def threshold_map(shape: str, size: int = 256) -> np.ndarray:
    """Return a ranked periodic threshold cell for an AM dot shape."""
    if shape not in HALFTONE_SHAPES:
        raise ValueError(f"Unbekannte Rasterform: {shape}")
    coordinates = (np.arange(size, dtype=np.float32) + 0.5) / size - 0.5
    x, y = np.meshgrid(coordinates, coordinates)
    absolute_x = np.abs(x)
    absolute_y = np.abs(y)

    if shape == "circle":
        score = x**2 + y**2
    elif shape == "ellipse":
        score = (x / 1.45) ** 2 + (y / 0.69) ** 2
    else:
        horizontal = np.maximum(absolute_x, absolute_y / 0.30)
        vertical = np.maximum(absolute_x / 0.30, absolute_y)
        score = np.minimum(horizontal, vertical)

    order = np.argsort(score.ravel(), kind="stable")
    ranked = np.empty(size * size, dtype=np.float32)
    ranked[order] = (np.arange(size * size, dtype=np.float32) + 0.5) / (
        size * size
    )
    return ranked.reshape(size, size)


def apply_tone_curve(
    coverage: np.ndarray,
    gamma: float,
    minimum_dot: float,
    maximum_dot: float,
) -> np.ndarray:
    """Apply a simple dot-gain curve and printable dot limits."""
    gamma = max(0.1, float(gamma))
    minimum_dot = float(np.clip(minimum_dot, 0.0, 1.0))
    maximum_dot = float(np.clip(maximum_dot, minimum_dot, 1.0))
    adjusted = np.power(
        np.clip(np.asarray(coverage, dtype=np.float32), 0.0, 1.0),
        1.0 / gamma,
    )
    adjusted = np.where(adjusted < minimum_dot, 0.0, adjusted)
    return np.where(adjusted > maximum_dot, 1.0, adjusted).astype(np.float32)


def smooth_coverages(coverages: np.ndarray, size: int) -> np.ndarray:
    if size < 3:
        return np.asarray(coverages, dtype=np.float32)
    if size % 2 == 0:
        raise ValueError("Die Klassenglättung muss ungerade sein.")
    radius = size / 2.0
    output = np.empty_like(coverages, dtype=np.float32)
    for channel in range(coverages.shape[-1]):
        source = Image.fromarray(
            np.rint(np.clip(coverages[..., channel], 0.0, 1.0) * 255).astype(
                np.uint8
            ),
            mode="L",
        )
        try:
            filtered = source.filter(ImageFilter.GaussianBlur(radius=radius))
            try:
                output[..., channel] = (
                    np.asarray(filtered, dtype=np.float32) / 255.0
                )
            finally:
                filtered.close()
        finally:
            source.close()
    return output


def rasterize_coverage(
    coverage: np.ndarray,
    *,
    dpi: float,
    frequency_lpi: float,
    angle: float,
    shape: str,
    x_offset: int = 0,
    y_offset: int = 0,
) -> np.ndarray:
    """Rasterize one continuous ink coverage using global pixel coordinates."""
    frequency_lpi = max(1.0, float(frequency_lpi))
    period = max(1.0, float(dpi) / frequency_lpi)
    angle_radians = np.radians(float(angle))
    cosine = np.cos(angle_radians)
    sine = np.sin(angle_radians)
    height, width = coverage.shape
    x = np.arange(x_offset, x_offset + width, dtype=np.float32)[None, :]
    y = np.arange(y_offset, y_offset + height, dtype=np.float32)[:, None]
    u = (x * cosine + y * sine) / period
    v = (-x * sine + y * cosine) / period

    cell = threshold_map(shape)
    cell_size = cell.shape[0]
    cell_x = np.floor(np.mod(u + 0.5, 1.0) * cell_size).astype(np.intp)
    cell_y = np.floor(np.mod(v + 0.5, 1.0) * cell_size).astype(np.intp)
    return np.asarray(coverage, dtype=np.float32) >= cell[cell_y, cell_x]


def rasterize_coverages(
    coverages: np.ndarray,
    *,
    dpi: float,
    frequency_lpi: float,
    angles: list[float] | tuple[float, ...],
    shape: str,
    x_offset: int = 0,
    y_offset: int = 0,
) -> np.ndarray:
    masks = np.empty(coverages.shape, dtype=bool)
    for channel, angle in enumerate(angles):
        masks[..., channel] = rasterize_coverage(
            coverages[..., channel],
            dpi=dpi,
            frequency_lpi=frequency_lpi,
            angle=angle,
            shape=shape,
            x_offset=x_offset,
            y_offset=y_offset,
        )
    return masks


def state_indices_from_masks(masks: np.ndarray, palette_masks: np.ndarray) -> np.ndarray:
    """Map actual on/off ink masks back to the palette's state ordering."""
    ink_count = masks.shape[-1]
    bit_weights = 1 << np.arange(ink_count, dtype=np.uint8)
    codes = np.sum(masks.astype(np.uint8) * bit_weights, axis=-1, dtype=np.uint8)
    palette_codes = np.asarray(palette_masks, dtype=np.uint8) @ bit_weights
    lookup = np.zeros(1 << ink_count, dtype=np.uint8)
    lookup[palette_codes] = np.arange(len(palette_masks), dtype=np.uint8)
    return lookup[codes]
