from dataclasses import dataclass


@dataclass
class Ink:
    name: str
    lab: tuple[float, float, float]
    rgb_preview: tuple[int, int, int]
    pantone: str = ""
    opacity: float = 0.72
    bias: float = 0.0
