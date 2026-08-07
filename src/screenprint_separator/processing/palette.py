from dataclasses import dataclass

import numpy as np

from screenprint_separator.models.ink import Ink
from screenprint_separator.processing.color_converter import ColorConverter


@dataclass(frozen=True)
class OverprintPalette:
    names: tuple[str, ...]
    masks: np.ndarray
    rgb: np.ndarray
    lab: np.ndarray


def build_overprint_palette(
    inks: list[Ink],
    paper_rgb: tuple[int, int, int],
    measured_lab: dict[int, tuple[float, float, float]] | None = None,
) -> OverprintPalette:
    if len(inks) != 3:
        raise ValueError("Für die Separation werden genau drei Farben benötigt.")

    masks = np.array(
        [
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 0),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
        ],
        dtype=np.uint8,
    )

    paper = np.asarray(paper_rgb, dtype=np.float32)
    colors = []
    names = []

    for mask in masks:
        result = paper.copy()
        active_names = []

        for is_active, ink in zip(mask, inks, strict=True):
            if not is_active:
                continue
            ink_rgb = np.asarray(ink.rgb_preview, dtype=np.float32)
            if active_names:
                alpha = float(np.clip(ink.opacity, 0.0, 1.0))
                result = result * (1.0 - alpha) + ink_rgb * alpha
            else:
                result = ink_rgb.copy()
            active_names.append(ink.name)

        colors.append(result)
        names.append("Papier" if not active_names else " + ".join(active_names))

    rgb = np.clip(np.rint(colors), 0, 255).astype(np.uint8)
    lab = ColorConverter.rgb_to_lab(rgb)

    # Preserve user-supplied LAB values for the three solid inks. Only the
    # unmeasured overprints need to use the RGB/opacity approximation.
    for state, ink in zip((1, 2, 3), inks, strict=True):
        lab[state] = ink.lab
        rgb[state] = ink.rgb_preview

    if measured_lab:
        for state, values in measured_lab.items():
            if 0 <= state < len(lab):
                lab[state] = values
                rgb[state] = ColorConverter.lab_to_rgb(
                    np.asarray(values, dtype=np.float32)
                )

    return OverprintPalette(tuple(names), masks, rgb, lab)
