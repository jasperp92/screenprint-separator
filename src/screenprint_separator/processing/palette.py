from dataclasses import dataclass
from itertools import combinations

import numpy as np

from screenprint_separator.models.ink import Ink
from screenprint_separator.processing.color_converter import ColorConverter


@dataclass(frozen=True)
class OverprintPalette:
    names: tuple[str, ...]
    masks: np.ndarray
    rgb: np.ndarray
    lab: np.ndarray
    @property
    def mixed_states(self) -> tuple[int, ...]:
        """Every printed state is a mixture on the paper substrate."""
        return tuple(range(1, len(self.names)))

    @property
    def levels(self) -> np.ndarray:
        return self.masks.sum(axis=1) + 1


def build_overprint_palette(
    inks: list[Ink],
    paper_rgb: tuple[int, int, int],
    measured_lab: dict[int, tuple[float, float, float]] | None = None,
    *,
    paper_lab: tuple[float, float, float] | None = None,
) -> OverprintPalette:
    if not 1 <= len(inks) <= 5:
        raise ValueError("Für die Separation werden eine bis fünf Farben benötigt.")

    mask_values = [tuple(0 for _ in inks)]
    for count in range(1, len(inks) + 1):
        for active_indices in combinations(range(len(inks)), count):
            mask_values.append(
                tuple(int(index in active_indices) for index in range(len(inks)))
            )
    masks = np.asarray(mask_values, dtype=np.uint8)

    paper = np.asarray(paper_rgb, dtype=np.float32)
    colors = []
    names = []

    for mask in masks:
        result = paper.copy()
        active_names = ["0"]

        for is_active, ink in zip(mask, inks, strict=True):
            if not is_active:
                continue
            ink_rgb = np.asarray(ink.rgb_preview, dtype=np.float32)
            alpha = float(np.clip(ink.opacity, 0.0, 1.0))
            result = result * (1.0 - alpha) + ink_rgb * alpha
            active_names.append(ink.name)

        colors.append(result)
        names.append(" + ".join(active_names))

    rgb = np.clip(np.rint(colors), 0, 255).astype(np.uint8)
    lab = ColorConverter.rgb_to_lab(rgb)

    if paper_lab is not None:
        lab[0] = paper_lab

    if measured_lab:
        for state, values in measured_lab.items():
            if 0 <= state < len(lab):
                lab[state] = values
                rgb[state] = ColorConverter.lab_to_rgb(
                    np.asarray(values, dtype=np.float32)
                )

    return OverprintPalette(tuple(names), masks, rgb, lab)


def mixed_state_indices(ink_count: int = 3) -> tuple[int, ...]:
    """Palette states which contain at least two inks."""
    first_mixed_state = 1 + ink_count
    return tuple(range(first_mixed_state, 2**ink_count))
