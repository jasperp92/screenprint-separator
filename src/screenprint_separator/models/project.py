from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from screenprint_separator.models.effect import ImageEffect
from screenprint_separator.models.ink import Ink
from screenprint_separator.models.settings import Settings


@dataclass
class Project:
    settings: Settings
    inks: list[Ink] = field(default_factory=list)
    effects: list[ImageEffect] = field(default_factory=list)

    image: Image.Image | None = None
    preview_image: Image.Image | None = None

    rgb_array: np.ndarray | None = None
    preview_array: np.ndarray | None = None

    channels: dict[str, np.ndarray] | None = None
    class_indices: np.ndarray | None = None
    measured_overprints: dict[int, tuple[float, float, float]] = field(
        default_factory=dict
    )
    manual_overprint_states: set[int] = field(default_factory=set)
    overprint_sources: dict[int, str] = field(default_factory=dict)
    overprint_library_ids: dict[int, str] = field(default_factory=dict)
