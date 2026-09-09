import io
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
from screenprint_separator.processing.color_converter import ColorConverter
from screenprint_separator.processing.palette import build_overprint_palette
from screenprint_separator.processing.pipeline import process_image
from screenprint_separator.processing.simulation import Simulation


class PaperColorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.inks = [Ink("Schwarz", (0.0, 0.0, 0.0), (0, 0, 0), opacity=1.0)]
        self.palette = build_overprint_palette(self.inks, (255, 255, 255))

    def test_paper_and_printed_state_assignment(self):
        classifier = ColorClassifier(self.palette, [0.0])
        np.testing.assert_array_equal(classifier.classify(self.palette.lab), [0, 1])
        coverage = classifier.coverage_rgb_lut(self.palette.rgb, softness=4.0)
        self.assertTrue(np.isfinite(coverage).all())
        self.assertLess(coverage[0, 0], 0.01)
        self.assertGreater(coverage[1, 0], 0.99)

    def test_paper_bias_changes_solid_and_halftone_assignment(self):
        neutral = ColorClassifier(self.palette, [0.0])
        preferred = ColorClassifier(self.palette, [0.0], paper_bias=20.0)
        lab = np.asarray([[45.0, 0.0, 0.0]], dtype=np.float32)
        self.assertEqual(int(neutral.classify(lab)[0]), 1)
        self.assertEqual(int(preferred.classify(lab)[0]), 0)
        rgb = np.asarray([[128, 128, 128]], dtype=np.uint8)
        self.assertLess(
            preferred.coverage_rgb_lut(rgb, 4.0)[0, 0],
            neutral.coverage_rgb_lut(rgb, 4.0)[0, 0],
        )

    def test_preview_and_full_resolution_keep_bare_paper(self):
        with Image.new("RGB", (4, 4), "white") as image:
            for preview in (True, False):
                settings = Settings(
                    preview_width=4, preview_height=4, class_smooth_size=1,
                )
                result = process_image(image, settings, self.inks, preview=preview)
                try:
                    np.testing.assert_array_equal(result.channels["Schwarz"], 0)
                finally:
                    result.working_image.close()
                    result.simulation.close()

    def test_lab_reference_is_preserved(self):
        reference = (93.0, 2.0, -3.0)
        palette = build_overprint_palette(
            self.inks, (255, 255, 255), paper_lab=reference
        )
        np.testing.assert_allclose(palette.lab[0], reference)

    def test_session_restores_paper_bias(self):
        project = Project(Settings(paper_bias=7.5), self.inks)
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            store.save(project, None)
            restored, _ = store.load(Project(Settings(), self.inks))
        self.assertEqual(restored.settings.paper_bias, 7.5)


class PaperMixtureTests(unittest.TestCase):
    def test_mixtures_use_paper_as_bottom_color_and_ink_opacity(self):
        inks = [
            Ink("Rot", (50, 60, 40), (200, 0, 0), opacity=0.5),
            Ink("Blau", (30, 20, -50), (0, 0, 200), opacity=0.25),
        ]
        palette = build_overprint_palette(inks, (240, 200, 100))
        self.assertEqual(len(palette.names), 4)
        self.assertEqual(palette.names[1:], (
            "0 + Rot", "0 + Blau", "0 + Rot + Blau",
        ))
        np.testing.assert_array_equal(palette.rgb[1], [220, 100, 50])
        np.testing.assert_array_equal(palette.rgb[3], [165, 75, 88])
        self.assertEqual(palette.masks.shape, (4, 2))
        self.assertEqual(palette.mixed_states, (1, 2, 3))
        np.testing.assert_array_equal(palette.levels, [1, 2, 2, 3])
        np.testing.assert_allclose(
            palette.lab, ColorConverter.rgb_to_lab(palette.rgb)
        )

    def test_bias_controls_bare_paper(self):
        ink = Ink("Schwarz", (0, 0, 0), (0, 0, 0))
        palette = build_overprint_palette([ink], (255, 255, 255))
        classifier = ColorClassifier(palette, [3], paper_bias=7)
        np.testing.assert_array_equal(classifier.state_biases, [7, 3])

    def test_five_inks_have_32_unique_states_and_five_plates(self):
        inks = [Ink(str(i), (i * 10, 0, 0), (i * 40,) * 3) for i in range(5)]
        palette = build_overprint_palette(inks, (255, 255, 255))
        self.assertEqual(palette.masks.shape, (32, 5))
        self.assertEqual(len(palette.mixed_states), 31)

    def test_paper_mixture_accepts_measured_lab(self):
        ink = Ink("Schwarz", (0, 0, 0), (0, 0, 0))
        palette = build_overprint_palette(
            [ink], (255, 255, 255), {1: (45, 3, 2)}
        )
        np.testing.assert_allclose(palette.lab[1], [45, 3, 2])


    def test_export_includes_paper_mixtures_without_extra_plate(self):
        ink = Ink("Schwarz", (0, 0, 0), (0, 0, 0), opacity=0.5)
        for mode in ("solid", "halftone"):
            with self.subTest(mode=mode), Image.new("RGB", (8, 8), (100, 100, 100)) as image:
                settings = Settings(
                    print_width_cm=0.5, print_height_cm=0.5,
                    dpi=72, output_dpi=72, tile_size=8,
                    halftone_mode=mode,
                )
                path = export_project(image, settings, [ink])
                try:
                    with zipfile.ZipFile(path) as archive:
                        manifest = json.loads(archive.read("projekt.json"))
                        self.assertEqual(len(manifest["palette"]), 2)
                        self.assertEqual(manifest["palette"][1]["name"], "0 + Schwarz")
                        self.assertTrue(manifest["palette"][1]["paper"])
                        self.assertEqual(manifest["palette"][1]["rgb"], [128, 128, 128])
                        self.assertEqual([p["level"] for p in manifest["palette"]], [1, 2])
                        with Image.open(io.BytesIO(archive.read("simulation.png"))) as simulation:
                            colors = set(simulation.convert("RGB").get_flattened_data())
                            self.assertIn((128, 128, 128), colors)
                            self.assertLessEqual(colors, {(128, 128, 128), (255, 255, 255)})
                        self.assertEqual(len([n for n in archive.namelist() if n.endswith((".tif", ".tiff"))]), 1)
                finally:
                    shutil.rmtree(path.parent)


    def test_opacity_extremes_and_first_ink_half_opacity(self):
        for opacity, expected in ((0, (255, 255, 255)), (0.5, (128, 128, 128)), (1, (0, 0, 0))):
            ink = Ink("Schwarz", (0, 0, 0), (0, 0, 0), opacity=opacity)
            palette = build_overprint_palette([ink], (255, 255, 255))
            np.testing.assert_array_equal(palette.rgb[1], expected)

    def test_print_order_changes_mixture_on_paper(self):
        red = Ink("Rot", (50, 60, 40), (200, 0, 0), opacity=0.5)
        blue = Ink("Blau", (30, 20, -50), (0, 0, 200), opacity=0.25)
        palette = build_overprint_palette([blue, red], (240, 200, 100))
        np.testing.assert_array_equal(palette.rgb[3], [190, 75, 62])

    def test_legacy_paper_measurement_is_migrated_to_single_state(self):
        ink = Ink("Schwarz", (0, 0, 0), (0, 0, 0))
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            project = Project(Settings(), [ink])
            project.measured_overprints[2] = (45, 3, 2)
            project.manual_overprint_states.add(2)
            project.overprint_sources[2] = "lab"
            store.save(project, None)
            data = json.loads(store.state_path.read_text())
            data["version"] = 1
            data["settings"]["paper_as_color"] = True
            store.state_path.write_text(json.dumps(data))
            restored, _ = store.load(Project(Settings(), [ink]))
            self.assertEqual(restored.measured_overprints, {1: (45, 3, 2)})
            self.assertEqual(restored.manual_overprint_states, {1})
            self.assertEqual(restored.overprint_sources, {1: "lab"})
            store.save(restored, None)
            reloaded, _ = store.load(Project(Settings(), [ink]))
            self.assertEqual(reloaded.measured_overprints, restored.measured_overprints)


    def test_half_opacity_is_used_in_preview_raster_and_trapping(self):
        ink = Ink("Schwarz", (0, 0, 0), (0, 0, 0), opacity=0.5)
        with Image.new("RGB", (12, 12), (128, 128, 128)) as image:
            for mode in ("solid", "halftone"):
                result = process_image(
                    image,
                    Settings(preview_width=12, preview_height=12, halftone_mode=mode),
                    [ink],
                )
                try:
                    colors = set(result.simulation.get_flattened_data())
                    self.assertIn((128, 128, 128), colors)
                    self.assertLessEqual(colors, {(128, 128, 128), (255, 255, 255)})
                    with Simulation.render_trapped(result.class_indices, result.palette, 1) as trapped:
                        self.assertLessEqual(
                            set(trapped.get_flattened_data()), {(128, 128, 128), (255, 255, 255)}
                        )
                finally:
                    result.working_image.close()
                    result.simulation.close()
