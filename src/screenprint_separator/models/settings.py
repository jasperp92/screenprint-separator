from dataclasses import dataclass


@dataclass
class Settings:
    dpi: int = 300

    threshold: int = 128

    paper: tuple[int, int, int] = (255, 255, 255)
    green: tuple[int, int, int] = (0, 255, 0)
    yellow: tuple[int, int, int] = (255, 255, 0)
    blue: tuple[int, int, int] = (0, 0, 255)