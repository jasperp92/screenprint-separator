from dataclasses import dataclass

import numpy as np


@dataclass
class ColorClass:
    name: str
    rgb: tuple[int, int, int]


class ColorClassifier:

    def __init__(self) -> None:
        self.colors = [
            ColorClass("paper", (255, 255, 255)),
            ColorClass("green", (0, 255, 0)),
            ColorClass("yellow", (255, 255, 0)),
            ColorClass("blue", (0, 0, 255)),
        ]

    def classify(self, pixels: np.ndarray) -> np.ndarray:
        ...