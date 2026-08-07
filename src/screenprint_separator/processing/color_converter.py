import numpy as np
from PIL import Image


class ColorConverter:
    """Conversions between sRGB and CIELAB (D50, 2° observer)."""

    _RGB_TO_XYZ_D65 = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    )
    _XYZ_D65_TO_RGB = np.linalg.inv(_RGB_TO_XYZ_D65).astype(np.float32)
    _D65_TO_D50 = np.array(
        [
            [1.0479298, 0.0229468, -0.0501922],
            [0.0296278, 0.9904345, -0.0170738],
            [-0.0092430, 0.0150552, 0.7518743],
        ],
        dtype=np.float32,
    )
    _D50_TO_D65 = np.linalg.inv(_D65_TO_D50).astype(np.float32)
    _D50_WHITE = np.array([0.96422, 1.0, 0.82521], dtype=np.float32)

    @staticmethod
    def rgb_to_lab(image: Image.Image | np.ndarray) -> np.ndarray:
        rgb = np.asarray(image, dtype=np.float32) / 255.0
        linear = np.where(
            rgb <= 0.04045,
            rgb / 12.92,
            ((rgb + 0.055) / 1.055) ** 2.4,
        )
        xyz_d65 = linear @ ColorConverter._RGB_TO_XYZ_D65.T
        xyz_d50 = xyz_d65 @ ColorConverter._D65_TO_D50.T
        scaled = xyz_d50 / ColorConverter._D50_WHITE

        delta = 6.0 / 29.0
        transformed = np.where(
            scaled > delta**3,
            np.cbrt(scaled),
            scaled / (3.0 * delta**2) + 4.0 / 29.0,
        )

        lightness = 116.0 * transformed[..., 1] - 16.0
        a = 500.0 * (transformed[..., 0] - transformed[..., 1])
        b = 200.0 * (transformed[..., 1] - transformed[..., 2])
        return np.stack((lightness, a, b), axis=-1).astype(np.float32)

    @staticmethod
    def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
        values = np.asarray(lab, dtype=np.float32)
        fy = (values[..., 0] + 16.0) / 116.0
        fx = fy + values[..., 1] / 500.0
        fz = fy - values[..., 2] / 200.0
        transformed = np.stack((fx, fy, fz), axis=-1)

        delta = 6.0 / 29.0
        scaled = np.where(
            transformed > delta,
            transformed**3,
            3.0 * delta**2 * (transformed - 4.0 / 29.0),
        )
        xyz_d50 = scaled * ColorConverter._D50_WHITE
        xyz_d65 = xyz_d50 @ ColorConverter._D50_TO_D65.T
        linear = xyz_d65 @ ColorConverter._XYZ_D65_TO_RGB.T
        linear = np.clip(linear, 0.0, 1.0)
        rgb = np.where(
            linear <= 0.0031308,
            12.92 * linear,
            1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
        )
        return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
