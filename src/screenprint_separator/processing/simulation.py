import numpy as np
from PIL import Image

from screenprint_separator.processing.palette import OverprintPalette


class Simulation:
    @staticmethod
    def render(class_indices: np.ndarray, palette: OverprintPalette) -> Image.Image:
        rgb = palette.rgb[np.asarray(class_indices, dtype=np.uint8)]
        return Image.fromarray(rgb, mode="RGB")
