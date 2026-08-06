import numpy as np
from PIL import ImageCms


class ColorConverter:

    @staticmethod
    def rgb_to_lab(image):
        srgb = ImageCms.createProfile("sRGB")
        lab = ImageCms.createProfile("LAB")

        transform = ImageCms.buildTransformFromOpenProfiles(
            srgb,
            lab,
            "RGB",
            "LAB",
        )

        return ImageCms.applyTransform(
            image,
            transform,
        )