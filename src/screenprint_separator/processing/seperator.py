import numpy as np

from screenprint_separator.models.ink import Ink
from screenprint_separator.processing.color_classifier import ColorClassifier
from screenprint_separator.processing.color_converter import ColorConverter
from screenprint_separator.processing.palette import build_overprint_palette
from screenprint_separator.processing.pipeline import channels_from_classes


class Separator:
    """Compatibility wrapper around the vectorized eight-state separator."""

    def __init__(
        self,
        inks: list[Ink],
        paper_rgb: tuple[int, int, int] = (191, 188, 181),
    ) -> None:
        self.inks = inks
        self.palette = build_overprint_palette(inks, paper_rgb)
        self.classifier = ColorClassifier(
            self.palette,
            [ink.bias for ink in inks],
        )

    def separate(self, image_rgb: np.ndarray) -> dict[str, np.ndarray]:
        image_lab = ColorConverter.rgb_to_lab(image_rgb)
        classes = self.classifier.classify_tiled(image_lab)
        return channels_from_classes(classes, self.palette, self.inks)
