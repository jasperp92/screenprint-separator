import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image

from screenprint_separator.models.effect import ImageEffect
from screenprint_separator.processing.effects import apply_effect
from screenprint_separator.ui.main_view import MainView


class SelectiveColorTests(unittest.TestCase):
    @staticmethod
    def _effect(target: tuple[int, int, int]) -> ImageEffect:
        effect = ImageEffect.create("selective_color")
        effect.parameters.update(
            {
                "target_r": float(target[0]),
                "target_g": float(target[1]),
                "target_b": float(target[2]),
                "tolerance": 5.0,
                "softness": 5.0,
                "amount": 1.0,
            }
        )
        return effect

    def test_target_color_is_desaturated_without_changing_distant_color(self) -> None:
        dark_green = (24, 72, 42)
        red = (210, 35, 30)
        image = Image.fromarray(np.asarray([[dark_green, red]], dtype=np.uint8))
        result = apply_effect(image, self._effect(dark_green))
        pixels = np.asarray(result)

        selected = pixels[0, 0]
        self.assertLess(int(selected.max()) - int(selected.min()), 4)
        np.testing.assert_array_equal(pixels[0, 1], red)

    def test_lightness_is_part_of_the_perceptual_selection(self) -> None:
        dark_green = (24, 72, 42)
        light_green = (110, 210, 140)
        image = Image.fromarray(
            np.asarray([[dark_green, light_green]], dtype=np.uint8)
        )
        result = apply_effect(image, self._effect(dark_green))

        np.testing.assert_array_equal(np.asarray(result)[0, 1], light_green)


class EyedropperTests(unittest.TestCase):
    def test_preview_coordinates_sample_median_from_original_image(self) -> None:
        pixels = np.zeros((10, 20, 3), dtype=np.uint8)
        pixels[3:8, 13:18] = (18, 76, 44)
        view = MainView.__new__(MainView)
        view.project = SimpleNamespace(image=Image.fromarray(pixels))
        view._input_preview_size = (10, 5)

        sampled = view._sample_input_color(7.5, 2.5)

        self.assertEqual(sampled, (18, 76, 44))


if __name__ == "__main__":
    unittest.main()
