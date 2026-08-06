import numpy as np

from screenprint_separator.models.ink import Ink


class Separator:

    def __init__(self, inks: list[Ink]):
        self.inks = inks

    def separate(self, image: np.ndarray) -> dict:

        channels = {}

        for ink in self.inks:
            channels[ink.name] = np.zeros(
                image.shape[:2],
                dtype=np.uint8,
            )

        return channels