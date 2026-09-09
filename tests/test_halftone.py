import unittest

import numpy as np

from screenprint_separator.models.ink import Ink
from screenprint_separator.processing.color_classifier import ColorClassifier
from screenprint_separator.processing.color_converter import ColorConverter
from screenprint_separator.processing.halftone import (
    apply_tone_curve,
    rasterize_coverage,
    state_indices_from_masks,
)
from screenprint_separator.processing.palette import build_overprint_palette


class HalftoneTests(unittest.TestCase):
    def test_grayscale_becomes_continuous_ink_coverage(self) -> None:
        black = (0, 0, 0)
        black_lab = tuple(
            float(value)
            for value in ColorConverter.rgb_to_lab(
                np.asarray(black, dtype=np.uint8)
            )
        )
        palette = build_overprint_palette(
            [Ink("Schwarz", black_lab, black, opacity=1.0)], (255, 255, 255)
        )
        classifier = ColorClassifier(palette, [0.0])
        levels = np.asarray([0, 64, 128, 192, 255], dtype=np.uint8)
        grayscale = np.stack((levels, levels, levels), axis=-1)
        coverage = classifier.coverage_rgb_lut(grayscale, softness=4.0)[..., 0]

        self.assertGreater(coverage[0], 0.99)
        self.assertLess(coverage[-1], 0.01)
        self.assertTrue(np.all(np.diff(coverage) < 0))
        self.assertGreater(coverage[2], 0.3)
        self.assertLess(coverage[2], 0.7)

    def test_dot_shapes_preserve_requested_area(self) -> None:
        coverage = np.full((800, 800), 0.37, dtype=np.float32)
        for shape in ("circle", "ellipse", "cross"):
            with self.subTest(shape=shape):
                mask = rasterize_coverage(
                    coverage,
                    dpi=300,
                    frequency_lpi=30,
                    angle=22.5,
                    shape=shape,
                )
                self.assertAlmostEqual(float(mask.mean()), 0.37, delta=0.01)

    def test_tiled_raster_matches_one_piece_raster(self) -> None:
        rng = np.random.default_rng(1234)
        coverage = rng.random((137, 191), dtype=np.float32)
        expected = rasterize_coverage(
            coverage,
            dpi=600,
            frequency_lpi=45,
            angle=75,
            shape="ellipse",
        )
        actual = np.empty_like(expected)
        split_x, split_y = 83, 61
        for x0, x1 in ((0, split_x), (split_x, coverage.shape[1])):
            for y0, y1 in ((0, split_y), (split_y, coverage.shape[0])):
                actual[y0:y1, x0:x1] = rasterize_coverage(
                    coverage[y0:y1, x0:x1],
                    dpi=600,
                    frequency_lpi=45,
                    angle=75,
                    shape="ellipse",
                    x_offset=x0,
                    y_offset=y0,
                )
        np.testing.assert_array_equal(actual, expected)

    def test_state_codes_map_to_palette_order(self) -> None:
        palette_masks = np.array(
            [[0, 0], [1, 0], [0, 1], [1, 1]], dtype=bool
        )
        masks = np.array(
            [[[0, 0], [1, 0]], [[0, 1], [1, 1]]], dtype=bool
        )
        indices = state_indices_from_masks(masks, palette_masks)
        np.testing.assert_array_equal(indices, [[0, 1], [2, 3]])

    def test_tone_curve_applies_dot_limits(self) -> None:
        values = np.array([0.01, 0.25, 0.99], dtype=np.float32)
        adjusted = apply_tone_curve(values, 1.0, 0.02, 0.98)
        np.testing.assert_allclose(adjusted, [0.0, 0.25, 1.0])


if __name__ == "__main__":
    unittest.main()
