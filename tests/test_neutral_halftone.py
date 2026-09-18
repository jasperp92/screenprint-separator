import io
import shutil
import unittest
import zipfile

import numpy as np
from PIL import Image

from screenprint_separator.export.plate_exporter import export_project
from screenprint_separator.models.ink import Ink
from screenprint_separator.models.settings import Settings
from screenprint_separator.processing.color_classifier import ColorClassifier
from screenprint_separator.processing.palette import build_overprint_palette
from screenprint_separator.processing.pipeline import process_image


class NeutralHalftoneTests(unittest.TestCase):
    def setUp(self):
        self.inks = [
            Ink("Rot", (0, 0, 0), (217, 7, 49), opacity=0.9),
            Ink("Gelb", (0, 0, 0), (255, 218, 31), opacity=0.8),
            Ink("Weiss", (100, 0, 0), (255, 255, 255), opacity=0.79),
        ]
        self.palette = build_overprint_palette(self.inks, (0, 0, 0))

    def test_gray_ramp_uses_white_only_and_matches_area_average(self):
        levels = np.arange(256, dtype=np.uint8)
        rgb = np.repeat(levels[:, None], 3, axis=1)
        classifier = ColorClassifier(self.palette, [0, 0, 0])
        coverage = classifier.coverage_rgb_lut(rgb, 3)
        np.testing.assert_array_equal(coverage[:, :2], 0)
        np.testing.assert_allclose(coverage[:, 2], np.minimum(levels / 201.0, 1), atol=1e-6)
        self.assertTrue(np.all(np.diff(coverage[:, 2]) >= 0))

    def test_near_neutral_tones_do_not_acquire_color_from_bias(self):
        rgb = np.asarray([[64, 65, 64], [128, 129, 128], [240, 241, 240]], dtype=np.uint8)
        classifier = ColorClassifier(
            self.palette, [10, 10, 0], paper_bias=7.5, overprint_biases={4: 20}
        )
        coverage = classifier.coverage_rgb_lut(rgb, 1.5)
        np.testing.assert_array_equal(coverage[:, :2], 0)
        self.assertGreater(coverage[1, 2], coverage[0, 2])
        self.assertEqual(coverage[-1, 2], 1)

    def test_saturated_red_and_orange_keep_mixture_assignment(self):
        rgb = np.asarray([[180, 10, 30], [255, 107, 36]], dtype=np.uint8)
        classifier = ColorClassifier(self.palette, [0, 0, 0])
        expected = classifier.coverage_rgb_lut(rgb, 3)
        classifier._neutral_ramp = None
        np.testing.assert_array_equal(classifier.coverage_rgb_lut(rgb, 3), expected)

    def test_neutrality_transition_has_no_abrupt_jump(self):
        red = np.linspace(120, 155, 200, dtype=np.float32)
        pixels = np.stack((red, np.full_like(red, 120), np.full_like(red, 120)), axis=-1)
        classifier = ColorClassifier(self.palette, [0, 0, 0])
        # Isolate the chroma transition from the existing quantized color LUT.
        coverage = classifier._preserve_neutral_tones(pixels, np.ones((200, 3), dtype=np.float32))
        self.assertEqual(coverage[0, 0], 0)
        self.assertEqual(coverage[-1, 0], 1)
        self.assertLess(np.max(np.abs(np.diff(coverage[:, 0]))), 0.04)

    def test_white_paper_and_black_ink_also_form_a_neutral_ramp(self):
        inks = [Ink("Rot", (0, 0, 0), (255, 0, 0)), Ink("Schwarz", (0, 0, 0), (0, 0, 0), opacity=1)]
        classifier = ColorClassifier(build_overprint_palette(inks, (255, 255, 255)), [0, 0])
        coverage = classifier.coverage_rgb_lut(np.asarray([[128, 128, 128]], dtype=np.uint8), 3)
        np.testing.assert_allclose(coverage, [[0, 127 / 255]], atol=1e-6)

    def test_colored_paper_without_neutral_ramp_keeps_existing_behavior(self):
        classifier = ColorClassifier(build_overprint_palette(self.inks[:2], (40, 20, 0)), [0, 0])
        self.assertIsNone(classifier._neutral_ramp)
        coverage = classifier.coverage_rgb_lut(np.asarray([[128, 128, 128]], dtype=np.uint8), 3)
        self.assertTrue(np.isfinite(coverage).all())
        self.assertGreater(coverage.sum(), 0)

    def test_opaque_white_does_not_select_colored_underprints(self):
        self.inks[-1].opacity = 1
        classifier = ColorClassifier(build_overprint_palette(self.inks, (0, 0, 0)), [0, 0, 0])
        coverage = classifier.coverage_rgb_lut(np.asarray([[128, 128, 128]], dtype=np.uint8), 3)
        np.testing.assert_allclose(coverage, [[0, 0, 128 / 255]], atol=1e-6)

    def test_gray_preview_and_export_have_no_colored_dots(self):
        settings = Settings(
            paper=(0, 0, 0), halftone_mode="halftone", halftone_gamma=0.75,
            print_width_cm=1, print_height_cm=1, dpi=72, output_dpi=72,
            preview_width=16, preview_height=16, tile_size=8,
        )
        with Image.new("RGB", (16, 16), (128, 128, 128)) as source:
            result = process_image(source, settings, self.inks)
            try:
                np.testing.assert_array_equal(result.channels["Rot"], 0)
                np.testing.assert_array_equal(result.channels["Gelb"], 0)
                white = result.channels["Weiss"]
                self.assertGreater(white.sum(), 0)
                self.assertTrue((white == 0).any())
            finally:
                result.working_image.close()
                result.simulation.close()
            path = export_project(source, settings, self.inks)
        try:
            with zipfile.ZipFile(path) as archive:
                with Image.open(io.BytesIO(archive.read("simulation.png"))) as simulation:
                    rgb = np.asarray(simulation)
                    np.testing.assert_array_equal(rgb[..., 0], rgb[..., 1])
                    np.testing.assert_array_equal(rgb[..., 1], rgb[..., 2])
                plates = [name for name in archive.namelist() if name.endswith('.tif')]
                self.assertEqual(len(plates), 3)
                for name in plates[:2]:
                    with Image.open(io.BytesIO(archive.read(name))) as plate:
                        self.assertTrue(np.asarray(plate).all())
        finally:
            shutil.rmtree(path.parent)
