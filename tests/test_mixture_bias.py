import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from screenprint_separator.export.plate_exporter import export_project
from screenprint_separator.models.ink import Ink
from screenprint_separator.models.project import Project
from screenprint_separator.models.settings import Settings
from screenprint_separator.persistence import SessionStore
from screenprint_separator.processing.color_classifier import ColorClassifier
from screenprint_separator.processing.palette import build_overprint_palette
from screenprint_separator.processing.pipeline import process_image


class MixtureBiasTests(unittest.TestCase):
    def setUp(self):
        self.inks = [
            Ink("Rot", (0, 0, 0), (217, 7, 49), opacity=0.7),
            Ink("Gelb", (0, 0, 0), (255, 218, 31), opacity=0.7),
            Ink("Weiss", (100, 0, 0), (255, 255, 255), opacity=0.79),
        ]
        self.palette = build_overprint_palette(self.inks, (0, 0, 0))

    def test_bias_only_changes_exact_mixture(self):
        classifier = ColorClassifier(
            self.palette, [1, 2, 3], paper_bias=4, overprint_biases={4: 10}
        )
        np.testing.assert_array_equal(classifier.state_biases, [4, 1, 2, 3, 13, 4, 5, 6])

    def test_orange_bias_reduces_white_in_orange_raster(self):
        rgb = np.asarray([[255, 107, 36]], dtype=np.uint8)
        neutral = ColorClassifier(self.palette, [0, 0, 0]).coverage_rgb_lut(rgb, 3)
        boosted = ColorClassifier(
            self.palette, [0, 0, 0], overprint_biases={4: 10}
        ).coverage_rgb_lut(rgb, 3)
        self.assertGreater(boosted[0, 0], neutral[0, 0])
        self.assertGreater(boosted[0, 1], neutral[0, 1])
        self.assertLess(boosted[0, 2], neutral[0, 2])

    def test_negative_bias_reduces_orange_and_zero_preserves_default(self):
        rgb = np.asarray([[255, 107, 36]], dtype=np.uint8)
        neutral = ColorClassifier(self.palette, [0, 0, 0]).coverage_rgb_lut(rgb, 3)
        zero = ColorClassifier(
            self.palette, [0, 0, 0], overprint_biases={4: 0}
        ).coverage_rgb_lut(rgb, 3)
        reduced = ColorClassifier(
            self.palette, [0, 0, 0], overprint_biases={4: -10}
        ).coverage_rgb_lut(rgb, 3)
        np.testing.assert_array_equal(zero, neutral)
        self.assertLess(reduced[0, 0], neutral[0, 0])

    def test_state_bias_also_affects_solid_assignment(self):
        target = self.palette.lab[2:3]
        neutral = ColorClassifier(self.palette, [0, 0, 0])
        boosted = ColorClassifier(self.palette, [0, 0, 0], overprint_biases={4: 20})
        self.assertEqual(int(neutral.classify(target)[0]), 2)
        self.assertEqual(int(boosted.classify(target)[0]), 4)

    def test_pipeline_uses_bias(self):
        settings = Settings(
            paper=(0, 0, 0), halftone_mode="halftone", halftone_softness=3,
            preview_width=4, preview_height=4, class_smooth_size=1,
        )
        with Image.new("RGB", (4, 4), (255, 107, 36)) as image:
            neutral = process_image(image, settings, self.inks)
            boosted = process_image(image, settings, self.inks, overprint_biases={4: 10})
            try:
                self.assertLess(boosted.coverages[..., 2].mean(), neutral.coverages[..., 2].mean())
                np.testing.assert_array_equal(boosted.palette.rgb, neutral.palette.rgb)
            finally:
                for result in (neutral, boosted):
                    result.working_image.close()
                    result.simulation.close()

    def test_bias_survives_session_and_is_recorded_in_export(self):
        project = Project(Settings(), self.inks, overprint_biases={4: 10})
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            store.save(project, None)
            restored, _ = store.load(Project(Settings()))
            self.assertEqual(restored.overprint_biases, {4: 10})
        settings = Settings(
            print_width_cm=0.5, print_height_cm=0.5, paper=(0, 0, 0),
            dpi=72, output_dpi=72, halftone_mode="halftone",
        )
        with Image.new("RGB", (4, 4), (255, 107, 36)) as image:
            path = export_project(image, settings, self.inks, overprint_biases={4: 10})
        try:
            with zipfile.ZipFile(path) as archive:
                self.assertEqual(json.loads(archive.read("projekt.json"))["overprint_biases"], {"4": 10})
        finally:
            shutil.rmtree(path.parent)
