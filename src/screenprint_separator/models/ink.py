from dataclasses import dataclass


@dataclass
class Ink:
    name: str
    lab: tuple[float, float, float]
    rgb_preview: tuple[int, int, int]
    pantone: str = ""
    color_source: str = "cmyk"
    cmyk: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 100.0)
    library_id: str = ""
    opacity: float = 0.72
    bias: float = 0.0
