import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from screenprint_separator.processing.color_converter import ColorConverter


@dataclass(frozen=True)
class ColorEntry:
    id: str
    name: str
    system: str
    cmyk: tuple[float, float, float, float] | None
    lab: tuple[float, float, float]
    rgb: tuple[int, int, int]


def cmyk_to_rgb(cmyk: tuple[float, float, float, float]) -> tuple[int, int, int]:
    """Convert generic device CMYK percentages to an sRGB preview."""
    cyan, magenta, yellow, black = np.clip(
        np.asarray(cmyk, dtype=np.float32) / 100.0,
        0.0,
        1.0,
    )
    rgb = 255.0 * np.array(
        [
            (1.0 - cyan) * (1.0 - black),
            (1.0 - magenta) * (1.0 - black),
            (1.0 - yellow) * (1.0 - black),
        ]
    )
    return tuple(int(value) for value in np.rint(rgb))


def rgb_to_lab_tuple(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    lab = ColorConverter.rgb_to_lab(np.asarray(rgb, dtype=np.uint8))
    return tuple(float(value) for value in lab)


def color_from_cmyk(
    cmyk: tuple[float, float, float, float],
) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
    rgb = cmyk_to_rgb(cmyk)
    return rgb, rgb_to_lab_tuple(rgb)


class ColorLibrary:
    def __init__(self, entries: list[ColorEntry], path: Path) -> None:
        self.entries = entries
        self.path = path
        self.by_id = {entry.id: entry for entry in entries}

    @classmethod
    def load(cls, path: Path) -> "ColorLibrary":
        if not path.exists():
            return cls([], path)

        document = json.loads(path.read_text(encoding="utf-8"))
        raw_entries = document.get("colors", document) if isinstance(document, dict) else document
        if not isinstance(raw_entries, list):
            raise TypeError("color_library.json muss eine Liste 'colors' enthalten.")

        entries = [cls._parse_entry(item) for item in raw_entries]
        if len({entry.id for entry in entries}) != len(entries):
            raise ValueError("Jede Farbe in color_library.json braucht eine eindeutige id.")
        return cls(entries, path)

    @staticmethod
    def _parse_entry(item: object) -> ColorEntry:
        if not isinstance(item, dict):
            raise TypeError("Jeder Farbeintrag muss ein JSON-Objekt sein.")

        identifier = str(item.get("id", "")).strip()
        name = str(item.get("name", "")).strip()
        system = str(item.get("system", "custom")).strip().lower()
        if not identifier or not name:
            raise ValueError("Jede Bibliotheksfarbe braucht 'id' und 'name'.")

        cmyk_value = item.get("cmyk")
        cmyk = None
        if cmyk_value is not None:
            if not isinstance(cmyk_value, list) or len(cmyk_value) != 4:
                raise ValueError(f"{name}: 'cmyk' braucht vier Prozentwerte.")
            cmyk = tuple(float(np.clip(value, 0.0, 100.0)) for value in cmyk_value)

        lab_value = item.get("lab")
        rgb_value = item.get("rgb")
        if lab_value is not None:
            if not isinstance(lab_value, list) or len(lab_value) != 3:
                raise ValueError(f"{name}: 'lab' braucht drei Werte.")
            lab = tuple(float(value) for value in lab_value)
        else:
            lab = None

        if rgb_value is not None:
            if not isinstance(rgb_value, list) or len(rgb_value) != 3:
                raise ValueError(f"{name}: 'rgb' braucht drei Werte.")
            rgb = tuple(int(np.clip(value, 0, 255)) for value in rgb_value)
        elif lab is not None:
            converted = ColorConverter.lab_to_rgb(np.asarray(lab, dtype=np.float32))
            rgb = tuple(int(value) for value in converted)
        elif cmyk is not None:
            rgb = cmyk_to_rgb(cmyk)
        else:
            raise ValueError(f"{name}: 'lab', 'rgb' oder 'cmyk' fehlt.")

        if lab is None:
            lab = rgb_to_lab_tuple(rgb)

        return ColorEntry(identifier, name, system, cmyk, lab, rgb)
