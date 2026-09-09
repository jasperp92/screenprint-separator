import json
import re
import shutil
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from screenprint_separator.models.effect import ImageEffect
from screenprint_separator.models.ink import Ink
from screenprint_separator.models.settings import Settings
from screenprint_separator.processing.color_classifier import ColorClassifier
from screenprint_separator.processing.color_converter import ColorConverter
from screenprint_separator.processing.effects import apply_effects
from screenprint_separator.processing.framing import crop_box_pixels
from screenprint_separator.processing.halftone import (
    apply_tone_curve,
    rasterize_coverage,
    rasterize_coverages,
    smooth_coverages,
    state_indices_from_masks,
)
from screenprint_separator.processing.palette import build_overprint_palette
from screenprint_separator.processing.pipeline import smooth_classes, smooth_texture
from screenprint_separator.processing.simulation import Simulation


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_")
    return normalized.lower() or "farbe"


def save_export_archive(
    archive_path: Path,
    downloads_directory: Path | None = None,
) -> Path:
    """Move an export archive into Downloads without overwriting older exports."""
    directory = downloads_directory or Path.home() / "Downloads"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / archive_path.name
    counter = 1
    while destination.exists():
        destination = directory / (
            f"{archive_path.stem} ({counter}){archive_path.suffix}"
        )
        counter += 1
    return Path(shutil.move(str(archive_path), destination))


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
    if settings.halftone_mode == "halftone":
        return 0
    return max(0, round(settings.trapping_mm / 25.4 * settings.output_dpi))


def _resize_for_print(image: Image.Image, settings: Settings) -> Image.Image:
    size = _work_size(settings)
    if settings.resize_mode in {"fit", "free"}:
        return image.resize(
            size,
            resample=Image.Resampling.LANCZOS,
            box=crop_box_pixels(settings.crop_box, image.size),
        )
    if settings.resize_mode == "pad":
        return ImageOps.pad(
            image,
            size,
            method=Image.Resampling.LANCZOS,
            color=settings.paper,
        )
    raise ValueError('Skalierungsmodus muss "fit", "pad" oder "free" sein.')


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


def _coverage_tiled(
    image: Image.Image,
    classifier: ColorClassifier,
    settings: Settings,
    path: Path,
    ink_count: int,
) -> np.memmap:
    """Calculate continuous ink coverages without keeping the full image in RAM."""
    width, height = image.size
    coverages = np.memmap(
        path,
        dtype=np.uint8,
        mode="w+",
        shape=(height, width, ink_count),
    )
    blur_padding = int(np.ceil(settings.texture_blur_radius * 3.0))
    coverage_padding = int(np.ceil(settings.class_smooth_size * 1.5))
    padding = max(blur_padding, coverage_padding)

    for y0 in range(0, height, settings.tile_size):
        y1 = min(y0 + settings.tile_size, height)
        for x0 in range(0, width, settings.tile_size):
            x1 = min(x0 + settings.tile_size, width)
            crop_x0 = max(0, x0 - padding)
            crop_y0 = max(0, y0 - padding)
            crop_x1 = min(width, x1 + padding)
            crop_y1 = min(height, y1 + padding)

            source = image.crop((crop_x0, crop_y0, crop_x1, crop_y1))
            try:
                tile = smooth_texture(source, settings)
            finally:
                source.close()
            try:
                tile_rgb = np.asarray(tile, dtype=np.uint8)
                tile_coverages = classifier.coverage_rgb_lut(
                    tile_rgb,
                    settings.halftone_softness,
                )
            finally:
                tile.close()
            tile_coverages = smooth_coverages(
                tile_coverages, settings.class_smooth_size
            )
            tile_coverages = apply_tone_curve(
                tile_coverages,
                settings.halftone_gamma,
                settings.halftone_min_dot,
                settings.halftone_max_dot,
            )

            inner_x0 = x0 - crop_x0
            inner_y0 = y0 - crop_y0
            inner_x1 = inner_x0 + x1 - x0
            inner_y1 = inner_y0 + y1 - y0
            inner = tile_coverages[inner_y0:inner_y1, inner_x0:inner_x1]
            coverages[y0:y1, x0:x1] = np.rint(inner * 255).astype(np.uint8)

    coverages.flush()
    return coverages


def _rasterize_working_coverages(
    coverages: np.memmap,
    palette_masks: np.ndarray,
    settings: Settings,
    inks: list[Ink],
    path: Path,
) -> np.memmap:
    height, width = coverages.shape[:2]
    labels = np.memmap(path, dtype=np.uint8, mode="w+", shape=(height, width))
    angles = [ink.screen_angle for ink in inks]
    for y0 in range(0, height, settings.tile_size):
        y1 = min(y0 + settings.tile_size, height)
        for x0 in range(0, width, settings.tile_size):
            x1 = min(x0 + settings.tile_size, width)
            tile = (
                np.asarray(coverages[y0:y1, x0:x1], dtype=np.float32) / 255.0
            )
            masks = rasterize_coverages(
                tile,
                dpi=settings.dpi,
                frequency_lpi=settings.halftone_frequency_lpi,
                angles=angles,
                shape=settings.halftone_shape,
                x_offset=x0,
                y_offset=y0,
            )
            labels[y0:y1, x0:x1] = state_indices_from_masks(
                masks, palette_masks
            )
    labels.flush()
    return labels


def _close_memmap(array: np.memmap) -> None:
    """Flush and release the OS mapping immediately instead of waiting for GC."""
    array.flush()
    mapping = getattr(array, "_mmap", None)
    if mapping is not None:
        mapping.close()


def _render_plate(
    active_image: Image.Image,
    output_size: tuple[int, int],
    settings: Settings,
) -> Image.Image:
    """Render a smooth 1-bit plate in bounded-memory output tiles."""
    output_width, output_height = output_size
    source_width, source_height = active_image.size
    plate = Image.new("1", output_size, color=1)
    trapping_pixels = _trapping_pixels(settings)
    # Include neighboring output pixels so LANCZOS resampling and the rounded
    # trapping blur do not create visible seams at tile boundaries.
    resize_scale = max(
        output_width / source_width,
        output_height / source_height,
        1.0,
    )
    resample_padding = int(np.ceil(3.0 * resize_scale))
    trapping_padding = int(np.ceil(trapping_pixels * 1.5))
    padding = trapping_padding + resample_padding
    tile_size = max(512, settings.tile_size * 2)

    for y0 in range(0, output_height, tile_size):
        y1 = min(y0 + tile_size, output_height)
        for x0 in range(0, output_width, tile_size):
            x1 = min(x0 + tile_size, output_width)
            outer_x0 = max(0, x0 - padding)
            outer_y0 = max(0, y0 - padding)
            outer_x1 = min(output_width, x1 + padding)
            outer_y1 = min(output_height, y1 + padding)
            outer_size = (outer_x1 - outer_x0, outer_y1 - outer_y0)
            source_box = (
                outer_x0 * source_width / output_width,
                outer_y0 * source_height / output_height,
                outer_x1 * source_width / output_width,
                outer_y1 * source_height / output_height,
            )
            resized = active_image.resize(
                outer_size,
                resample=Image.Resampling.LANCZOS,
                box=source_box,
            )
            threshold = settings.threshold
            if trapping_pixels > 0:
                # A square MaxFilter turns every isolated classified pixel into
                # a conspicuous block. Remove sub-print-sized specks first and
                # use a radial blur with a low cutoff for rounded expansion.
                cleaned = resized.filter(ImageFilter.MedianFilter(size=3))
                resized.close()
                trapped = cleaned.filter(
                    ImageFilter.GaussianBlur(
                        radius=max(0.5, trapping_pixels / 2.0)
                    )
                )
                cleaned.close()
                resized = trapped
                # With sigma=radius/2, a cutoff around 6/255 expands a solid
                # edge by approximately the requested radius.
                threshold = max(threshold, 249)
            inverted = ImageOps.invert(resized)
            resized.close()
            bilevel = inverted.point(
                lambda value, threshold=threshold: (
                    255 if value >= threshold else 0
                ),
                mode="1",
            )
            inverted.close()
            inner_box = (
                x0 - outer_x0,
                y0 - outer_y0,
                x1 - outer_x0,
                y1 - outer_y0,
            )
            inner = bilevel.crop(inner_box)
            bilevel.close()
            plate.paste(inner, (x0, y0))
            inner.close()
    return plate


def _render_halftone_plate(
    coverage_image: Image.Image,
    output_size: tuple[int, int],
    settings: Settings,
    angle: float,
) -> Image.Image:
    """Rasterize continuous coverage at final output resolution."""
    output_width, output_height = output_size
    source_width, source_height = coverage_image.size
    plate = Image.new("1", output_size, color=1)
    resize_scale = max(
        output_width / source_width,
        output_height / source_height,
        1.0,
    )
    padding = int(np.ceil(3.0 * resize_scale))
    tile_size = max(512, settings.tile_size * 2)

    for y0 in range(0, output_height, tile_size):
        y1 = min(y0 + tile_size, output_height)
        for x0 in range(0, output_width, tile_size):
            x1 = min(x0 + tile_size, output_width)
            outer_x0 = max(0, x0 - padding)
            outer_y0 = max(0, y0 - padding)
            outer_x1 = min(output_width, x1 + padding)
            outer_y1 = min(output_height, y1 + padding)
            outer_size = (outer_x1 - outer_x0, outer_y1 - outer_y0)
            source_box = (
                outer_x0 * source_width / output_width,
                outer_y0 * source_height / output_height,
                outer_x1 * source_width / output_width,
                outer_y1 * source_height / output_height,
            )
            resized = coverage_image.resize(
                outer_size,
                resample=Image.Resampling.LANCZOS,
                box=source_box,
            )
            try:
                coverage = np.asarray(resized, dtype=np.float32) / 255.0
                active = rasterize_coverage(
                    coverage,
                    dpi=settings.output_dpi,
                    frequency_lpi=settings.halftone_frequency_lpi,
                    angle=angle,
                    shape=settings.halftone_shape,
                    x_offset=outer_x0,
                    y_offset=outer_y0,
                )
                rendered = Image.fromarray(
                    np.where(active, 0, 255).astype(np.uint8), mode="L"
                ).convert("1")
            finally:
                resized.close()
            inner_box = (
                x0 - outer_x0,
                y0 - outer_y0,
                x1 - outer_x0,
                y1 - outer_y0,
            )
            inner = rendered.crop(inner_box)
            rendered.close()
            plate.paste(inner, (x0, y0))
            inner.close()
    return plate


def export_project(
    image: Image.Image,
    settings: Settings,
    inks: list[Ink],
    measured_lab: dict[int, tuple[float, float, float]] | None = None,
    effects: list[ImageEffect] | None = None,
    overprint_biases: dict[int, float] | None = None,
) -> Path:
    export_dir = Path(tempfile.mkdtemp(prefix="screenprint_separator_"))
    labels_path = export_dir / "classification.dat"
    coverages_path = export_dir / "coverages.dat"
    labels: np.memmap | None = None
    coverages: np.memmap | None = None
    work_image: Image.Image | None = None
    try:
        resized = _resize_for_print(image, settings)
        try:
            work_image = apply_effects(resized, effects or [])
        finally:
            resized.close()

        palette = build_overprint_palette(
            inks, settings.paper, measured_lab,
            paper_lab=(settings.paper_lab if settings.paper_source == "lab" else None)
        )
        classifier = ColorClassifier(
            palette,
            [ink.bias for ink in inks],
            paper_bias=settings.paper_bias,
            overprint_biases=overprint_biases,
        )
        if settings.halftone_mode == "halftone":
            coverages = _coverage_tiled(
                work_image,
                classifier,
                settings,
                coverages_path,
                len(inks),
            )
            labels = _rasterize_working_coverages(
                coverages,
                palette.masks,
                settings,
                inks,
                labels_path,
            )
        else:
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
        plate_paths = []
        for channel, ink in enumerate(inks):
            if coverages is not None:
                active_image = Image.fromarray(
                    np.array(coverages[..., channel], dtype=np.uint8), mode="L"
                )
                try:
                    plate = _render_halftone_plate(
                        active_image,
                        output_size,
                        settings,
                        ink.screen_angle,
                    )
                finally:
                    active_image.close()
            else:
                # Index only one channel at a time. Indexing all masks at once
                # creates a large H x W x ink-count allocation.
                channel_states = np.asarray(
                    palette.masks[:, channel], dtype=np.uint8
                ) * np.uint8(255)
                active = np.empty(labels.shape, dtype=np.uint8)
                np.take(channel_states, labels, out=active)
                active_image = Image.fromarray(active, mode="L")
                del active
                try:
                    plate = _render_plate(active_image, output_size, settings)
                finally:
                    active_image.close()
            try:
                plate_path = export_dir / (
                    f"farbauszug_{channel + 1}_{_safe_name(ink.name)}.tif"
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
            "effects": [asdict(effect) for effect in effects or []],
            "overprint_biases": overprint_biases or {},
            "print_order": [ink.name for ink in inks],
            "inks": [asdict(ink) for ink in inks],
            "palette": [
                {
                    "state": index,
                    "name": palette.names[index],
                    "mask": palette.masks[index].tolist(),
                    "paper": True,
                    "level": int(palette.levels[index]),
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
        if coverages is not None:
            _close_memmap(coverages)
            coverages = None
            coverages_path.unlink(missing_ok=True)

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
        if coverages is not None:
            _close_memmap(coverages)
        labels_path.unlink(missing_ok=True)
        coverages_path.unlink(missing_ok=True)
