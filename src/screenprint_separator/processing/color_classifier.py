import numpy as np

from screenprint_separator.processing.color_converter import ColorConverter
from screenprint_separator.processing.palette import OverprintPalette


def delta_e_2000(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the perceptual CIEDE2000 distance for broadcastable LAB arrays."""
    lab1 = np.asarray(first, dtype=np.float32)
    lab2 = np.asarray(second, dtype=np.float32)

    l1, a1, b1 = np.moveaxis(lab1, -1, 0)
    l2, a2, b2 = np.moveaxis(lab2, -1, 0)
    c1 = np.hypot(a1, b1)
    c2 = np.hypot(a2, b2)
    c_bar = (c1 + c2) / 2.0
    c_bar_7 = c_bar**7
    g = 0.5 * (1.0 - np.sqrt(c_bar_7 / (c_bar_7 + 25.0**7)))

    a1_prime = (1.0 + g) * a1
    a2_prime = (1.0 + g) * a2
    c1_prime = np.hypot(a1_prime, b1)
    c2_prime = np.hypot(a2_prime, b2)
    h1_prime = np.mod(np.degrees(np.arctan2(b1, a1_prime)), 360.0)
    h2_prime = np.mod(np.degrees(np.arctan2(b2, a2_prime)), 360.0)

    delta_l = l2 - l1
    delta_c = c2_prime - c1_prime
    delta_h_angle = h2_prime - h1_prime
    delta_h_angle = np.where(delta_h_angle > 180.0, delta_h_angle - 360.0, delta_h_angle)
    delta_h_angle = np.where(delta_h_angle < -180.0, delta_h_angle + 360.0, delta_h_angle)
    delta_h_angle = np.where(c1_prime * c2_prime == 0.0, 0.0, delta_h_angle)
    delta_h = 2.0 * np.sqrt(c1_prime * c2_prime) * np.sin(
        np.radians(delta_h_angle) / 2.0
    )

    l_bar = (l1 + l2) / 2.0
    c_bar_prime = (c1_prime + c2_prime) / 2.0
    hue_sum = h1_prime + h2_prime
    hue_difference = np.abs(h1_prime - h2_prime)
    h_bar = np.where(
        c1_prime * c2_prime == 0.0,
        hue_sum,
        np.where(
            hue_difference <= 180.0,
            hue_sum / 2.0,
            np.where(hue_sum < 360.0, (hue_sum + 360.0) / 2.0, (hue_sum - 360.0) / 2.0),
        ),
    )

    t = (
        1.0
        - 0.17 * np.cos(np.radians(h_bar - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * h_bar))
        + 0.32 * np.cos(np.radians(3.0 * h_bar + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * h_bar - 63.0))
    )
    delta_theta = 30.0 * np.exp(-((h_bar - 275.0) / 25.0) ** 2)
    c_bar_prime_7 = c_bar_prime**7
    r_c = 2.0 * np.sqrt(c_bar_prime_7 / (c_bar_prime_7 + 25.0**7))
    s_l = 1.0 + 0.015 * (l_bar - 50.0) ** 2 / np.sqrt(20.0 + (l_bar - 50.0) ** 2)
    s_c = 1.0 + 0.045 * c_bar_prime
    s_h = 1.0 + 0.015 * c_bar_prime * t
    r_t = -np.sin(np.radians(2.0 * delta_theta)) * r_c

    l_term = delta_l / s_l
    c_term = delta_c / s_c
    h_term = delta_h / s_h
    distance = np.sqrt(
        np.maximum(
            0.0,
            l_term**2 + c_term**2 + h_term**2 + r_t * c_term * h_term,
        )
    )
    return distance.astype(np.float32)


class ColorClassifier:
    def __init__(
        self,
        palette: OverprintPalette,
        ink_biases: tuple[float, ...] | list[float],
        *,
        paper_bias: float = 0.0,
        overprint_biases: dict[int, float] | None = None,
    ) -> None:
        self.palette = palette
        self.state_biases = palette.masks @ np.asarray(ink_biases, dtype=np.float32)
        # The substrate is present everywhere; its bias controls bare paper only.
        self.state_biases[0] = paper_bias
        for state, bias in (overprint_biases or {}).items():
            if 0 < state < len(palette.names):
                self.state_biases[state] += float(bias)
        self._coverage_lookups: dict[tuple[int, float], np.ndarray] = {}
        self._neutral_ramp = self._find_neutral_ramp()

    def _find_neutral_ramp(self) -> tuple[int, int] | None:
        """Find neutral endpoints connected by exactly one physical ink channel."""
        chroma = np.hypot(self.palette.lab[:, 1], self.palette.lab[:, 2])
        neutral = np.flatnonzero(chroma <= 2.0)
        best = None
        best_score = (0.0, 0)
        for offset, first in enumerate(neutral):
            for second in neutral[offset + 1:]:
                masks = self.palette.masks[[first, second]]
                if np.count_nonzero(masks[0] != masks[1]) != 1:
                    continue
                span = abs(float(self.palette.lab[first, 0] - self.palette.lab[second, 0]))
                # Prefer a broad tonal range, then fewer plates. An opaque white
                # must not bring along invisible colored underprints as a tie.
                score = (span, -int(masks.sum()))
                if span > 1.0 and score > best_score:
                    best = (int(first), int(second))
                    best_score = score
        return best

    def _preserve_neutral_tones(
        self, pixels_rgb: np.ndarray, coverages: np.ndarray
    ) -> np.ndarray:
        """Represent near-neutral tones through one neutral physical dot ramp."""
        if self._neutral_ramp is None:
            return coverages
        pixels = np.asarray(pixels_rgb, dtype=np.float32).reshape(-1, 3)
        # A cheap prefilter avoids converting every saturated pixel to LAB.
        candidates = np.flatnonzero(np.ptp(pixels, axis=1) <= 48.0)
        if not len(candidates):
            return coverages
        lab = ColorConverter.rgb_to_lab(pixels[candidates])
        chroma = np.hypot(lab[:, 1], lab[:, 2])
        selected = chroma < 12.0
        candidates = candidates[selected]
        if not len(candidates):
            return coverages
        chroma = chroma[selected]
        first, second = self._neutral_ramp
        start, end = self.palette.rgb[[first, second]].astype(np.float32)
        direction = end - start
        length_squared = float(direction @ direction)
        if length_squared < 1e-6:
            return coverages
        # Match the area-averaged RGB of the two rendered endpoint states.
        # Clamp out-of-range highlights/shadows instead of adding colored ink.
        fraction = np.clip(
            (pixels[candidates] - start) @ direction / length_squared, 0.0, 1.0
        )
        span = max(1.0, float(delta_e_2000(
            self.palette.lab[first], self.palette.lab[second]
        )))
        bias_difference = self.state_biases[second] - self.state_biases[first]
        odds = np.exp(float(np.clip(2.0 * bias_difference / span, -20.0, 20.0)))
        # Bias may move the tone along this ramp, but cannot introduce a hue.
        fraction = fraction * odds / (1.0 - fraction + fraction * odds)
        ramp = (
            (1.0 - fraction[:, None]) * self.palette.masks[first]
            + fraction[:, None] * self.palette.masks[second]
        )
        transition = np.clip((chroma - 4.0) / 8.0, 0.0, 1.0)
        strength = 1.0 - transition**2 * (3.0 - 2.0 * transition)
        flat = coverages.reshape(-1, coverages.shape[-1])
        flat[candidates] = (
            ramp * strength[:, None]
            + flat[candidates] * (1.0 - strength[:, None])
        )
        return coverages

    def classify(self, pixels_lab: np.ndarray) -> np.ndarray:
        output_shape = pixels_lab.shape[:-1]
        labels = np.zeros(output_shape, dtype=np.uint8)
        best = np.full(output_shape, np.inf, dtype=np.float32)

        for index, target in enumerate(self.palette.lab):
            distance = delta_e_2000(pixels_lab, target) - self.state_biases[index]
            update = distance < best
            best[update] = distance[update]
            labels[update] = index

        return labels

    def classify_rgb_lut(
        self,
        pixels_rgb: np.ndarray,
        resolution: int = 65,
    ) -> np.ndarray:
        """Classify a preview through a small ΔE00 color lookup table."""
        levels = np.linspace(0, 255, resolution, dtype=np.uint8)
        red, green, blue = np.meshgrid(levels, levels, levels, indexing="ij")
        grid_rgb = np.stack((red, green, blue), axis=-1)
        grid_lab = ColorConverter.rgb_to_lab(grid_rgb)
        lookup = self.classify(grid_lab)

        indices = np.rint(
            np.asarray(pixels_rgb, dtype=np.float32) * (resolution - 1) / 255.0
        ).astype(np.uint8)
        return lookup[indices[..., 0], indices[..., 1], indices[..., 2]]

    def coverage_rgb_lut(
        self,
        pixels_rgb: np.ndarray,
        softness: float,
        resolution: int = 33,
    ) -> np.ndarray:
        """Estimate continuous ink coverages from soft palette-state weights."""
        softness = max(0.1, float(softness))
        key = (int(resolution), round(softness, 4))
        lookup = self._coverage_lookups.get(key)
        if lookup is None:
            levels = np.linspace(0, 255, resolution, dtype=np.uint8)
            red, green, blue = np.meshgrid(levels, levels, levels, indexing="ij")
            grid_rgb = np.stack((red, green, blue), axis=-1)
            grid_lab = ColorConverter.rgb_to_lab(grid_rgb)
            distances = np.empty(
                grid_lab.shape[:-1] + (len(self.palette.lab),), dtype=np.float32
            )
            for index, target in enumerate(self.palette.lab):
                distances[..., index] = (
                    delta_e_2000(grid_lab, target) - self.state_biases[index]
                )
            # Interpolate locally between nearby printable states. Inverse
            # distances preserve a full paper-to-ink tonal ramp; an exponential
            # softmax would only soften the old hard class boundary.
            neighbor_count = min(4, len(self.palette.lab))
            cutoff = np.partition(
                distances, neighbor_count - 1, axis=-1
            )[..., neighbor_count - 1 : neighbor_count]
            power = 1.0 + 2.0 / softness
            weights = np.where(
                distances <= cutoff,
                np.power(np.maximum(distances, 1e-3), -power),
                0.0,
            )
            weights /= np.sum(weights, axis=-1, keepdims=True)
            lookup = np.tensordot(
                weights,
                np.asarray(self.palette.masks, dtype=np.float32),
                axes=([-1], [0]),
            ).astype(np.float32)
            self._coverage_lookups[key] = lookup

        indices = np.rint(
            np.asarray(pixels_rgb, dtype=np.float32) * (resolution - 1) / 255.0
        ).astype(np.uint8)
        coverages = lookup[indices[..., 0], indices[..., 1], indices[..., 2]]
        return self._preserve_neutral_tones(pixels_rgb, coverages)

    def classify_tiled(self, pixels_lab: np.ndarray, tile_size: int = 512) -> np.ndarray:
        height, width = pixels_lab.shape[:2]
        labels = np.empty((height, width), dtype=np.uint8)

        for y0 in range(0, height, tile_size):
            y1 = min(y0 + tile_size, height)
            for x0 in range(0, width, tile_size):
                x1 = min(x0 + tile_size, width)
                labels[y0:y1, x0:x1] = self.classify(
                    pixels_lab[y0:y1, x0:x1]
                )

        return labels
