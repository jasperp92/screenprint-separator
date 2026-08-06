from dataclasses import dataclass


@dataclass
class Ink:
    name: str

    lab: tuple[float, float, float]

    rgb_preview: tuple[int, int, int]

    opacity: float = 1.0