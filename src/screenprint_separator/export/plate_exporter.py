import json
import re
import shutil
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from screenprint_separator.models.ink import Ink
from screenprint_separator.models.settings import Settings
from screenprint_separator.processing.color_classifier import ColorClassifier
from screenprint_separator.processing.color_converter import ColorConverter
from screenprint_separator.processing.palette import build_overprint_palette
from screenprint_separator.processing.pipeline import (
    adjust_input_image,
    smooth_classes,
    smooth_texture,
)
from screenprint_separator.processing.simulation import Simulation


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_")
    return normalized.lower() or "farbe"


def _work_size(settings: Settings) -> tuple[int, int]:
    return (
        round(settings.print_width_cm / 2.54 * settings.dpi),
        round(settings.print_height_cm / 2.54 * settings.dpi),
    )


def _output_size(settings: Settings) -> tuple[int, int]:
    return (
        round(settings.print_width_cm / 2.54 * settings.output_dpi),
        round(settings.print_height_cm / 2.54 * settings.output_dpi),
    )


def _trapping_pixels(settings: Settings) -> int:
    """Convert the physical trap width to pixels at final plate resolution."""
    return max(0, round(settings.trapping_mm / 25.4 * settings.output_dpi))


def _resize_for_print(image: Image.Image, settings: Settings) -> Image.Image:
    size = _work_size(settings)
    if settings.resize_mode == "fit":
        return ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)
    if settings.resize_mode == "pad":
        return ImageOps.pad(
            image,
            size,
            method=Image.Resampling.LANCZOS,
            color=settings.paper,
        )
    raise ValueError('Skalierungsmodus muss "fit" oder "pad" sein.')


def _classify_tiled(
    image: Image.Image,
    classifier: ColorClassifier,
    settings: Settings,
    path: Path,
) -> np.memmap:
    width, height = image.size
    labels = np.memmap(path, dtype=np.uint8, mode="w+", shape=(height, width))
    blur_padding = int(np.ceil(settings.texture_blur_radius * 3.0))
    smooth_padding = settings.class_smooth_size // 2
    padding = max(blur_padding, smooth_padding)

    for y0 in range(0, height, settings.tile_size):
        y1 = min(y0 + settings.tile_size, height)
        for x0 in range(0, width, settings.tile_size):
            x1 = min(x0 + settings.tile_size, width)
            crop_x0 = max(0, x0 - padding)
            crop_y0 = max(0, y0 - padding)
            crop_x1 = min(width, x1 + padding)
            crop_y1 = min(height, y1 + padding)

            tile = image.crop((crop_x0, crop_y0, crop_x1, crop_y1))
            tile = smooth_texture(tile, settings)
            tile_lab = ColorConverter.rgb_to_lab(tile)
            tile_labels = classifier.classify_tiled(tile_lab, settings.tile_size)
            tile_labels = smooth_classes(tile_labels, settings.class_smooth_size)

            inner_x0 = x0 - crop_x0
            inner_y0 = y0 - crop_y0
            inner_x1 = inner_x0 + x1 - x0
            inner_y1 = inner_y0 + y1 - y0
            labels[y0:y1, x0:x1] = tile_labels[
                inner_y0:inner_y1,
                inner_x0:inner_x1,
            ]

    labels.flush()
    return labels


def _close_memmap(array: np.memmap) -> None:
    """Flush and release the OS mapping immediately instead of waiting for GC."""
    array.flush()
    mapping = getattr(array, "_mmap", None)
    if mapping is not None:
        mapping.close()


def export_project(
    image: Image.Image,
    settings: Settings,
    inks: list[Ink],
    measured_lab: dict[int, tuple[float, float, float]] | None = None,
) -> Path:
    export_dir = Path(tempfile.mkdtemp(prefix="screenprint_separator_"))
    labels_path = export_dir / "classification.dat"
    labels: np.memmap | None = None
    work_image: Image.Image | None = None
    try:
        resized = _resize_for_print(image, settings)
        try:
            work_image = adjust_input_image(resized, settings)
        finally:
            resized.close()

        palette = build_overprint_palette(inks, settings.paper, measured_lab)
        classifier = ColorClassifier(palette, [ink.bias for ink in inks])
        labels = _classify_tiled(work_image, classifier, settings, labels_path)
        work_image.close()
        work_image = None

        simulation_path = export_dir / "simulation.png"
        simulation = Simulation.render(labels, palette)
        try:
            simulation.save(
                simulation_path,
                dpi=(settings.dpi, settings.dpi),
            )
        finally:
            simulation.close()

        output_size = _output_size(settings)
        trapping_pixels = _trapping_pixels(settings)
        plate_paths = []
        for channel, ink in enumerate(inks):
            # Index only one channel at a time.  Indexing all masks at once creates
            # an H x W x ink-count allocation (hundreds of MB at production DPI).
            channel_states = np.asarray(
                palette.masks[:, channel], dtype=np.uint8
            ) * np.uint8(255)
            active = np.empty(labels.shape, dtype=np.uint8)
            np.take(channel_states, labels, out=active)
            active_image = Image.fromarray(active, mode="L")
            del active

            resized_plate = active_image.resize(
                output_size,
                resample=Image.Resampling.NEAREST,
            )
            active_image.close()
            if trapping_pixels > 0:
                trapped = resized_plate.filter(
                    ImageFilter.MaxFilter(size=trapping_pixels * 2 + 1)
                )
                resized_plate.close()
                resized_plate = trapped

            inverted = ImageOps.invert(resized_plate)
            resized_plate.close()
            plate = inverted.point(
                lambda value: 255 if value >= settings.threshold else 0,
                mode="1",
            )
            inverted.close()
            try:
                plate_path = (
                    export_dir / f"platte_{channel + 1}_{_safe_name(ink.name)}.tif"
                )
                plate.save(
                    plate_path,
                    format="TIFF",
                    dpi=(settings.output_dpi, settings.output_dpi),
                    compression="group4",
                )
            finally:
                plate.close()
            plate_paths.append(plate_path)

        manifest = {
            "format": "screenprint-separator-project-v1",
            "settings": asdict(settings),
            "print_order": [ink.name for ink in inks],
            "inks": [asdict(ink) for ink in inks],
            "palette": [
                {
                    "state": index,
                    "name": palette.names[index],
                    "mask": palette.masks[index].tolist(),
                    "rgb": palette.rgb[index].tolist(),
                    "lab_d50": [round(float(value), 4) for value in palette.lab[index]],
                    "measured": index in (measured_lab or {}),
                }
                for index in range(len(palette.names))
            ],
        }
        manifest_path = export_dir / "projekt.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        _close_memmap(labels)
        labels = None
        labels_path.unlink(missing_ok=True)

        archive_path = export_dir / "screenprint_export.zip"
        with zipfile.ZipFile(
            archive_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for path in [simulation_path, manifest_path, *plate_paths]:
                archive.write(path, path.name)

        return archive_path
    except Exception:
        shutil.rmtree(export_dir, ignore_errors=True)
        raise
    finally:
        if work_image is not None:
            work_image.close()
        if labels is not None:
            _close_memmap(labels)
        labels_path.unlink(missing_ok=True)
