from io import BytesIO

import numpy as np
from PIL import Image


class ImageLoader:

    @staticmethod
    def load(data: bytes) -> tuple[Image.Image, np.ndarray]:
        image = Image.open(BytesIO(data))
        image = image.convert("RGB")

        return image, np.array(image)

    @staticmethod
    def create_preview(image: Image.Image) -> bytes:
        preview = image.copy()
        preview.thumbnail((800, 800))

        buffer = BytesIO()
        preview.save(buffer, format="PNG")

        return buffer.getvalue()