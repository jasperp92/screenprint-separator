from dataclasses import dataclass

import numpy as np
from PIL import Image

from screenprint_separator.models.settings import Settings


@dataclass
class Project:
    settings: Settings

    image: Image.Image | None = None
    rgb_array: np.ndarray | None = None
    lab_array: np.ndarray | None = None