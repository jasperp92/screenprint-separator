from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

from screenprint_separator.models.effect import ImageEffect
from screenprint_separator.models.ink import Ink
from screenprint_separator.models.settings import Settings
from screenprint_separator.processing.color_classifier import ColorClassifier
from screenprint_separator.processing.color_converter import ColorConverter
from screenprint_separator.processing.effects import apply_effects
from screenprint_separator.processing.halftone import (
    apply_tone_curve,
    rasterize_coverages,
    smooth_coverages,
    state_indices_from_masks,
)
from screenprint_separator.processing.palette import (
    OverprintPalette,
    build_overprint_palette,
)
from screenprint_separator.processing.preview_processor import PreviewProcessor
from screenprint_separator.processing.simulation import Simulation


@dataclass
class SeparationResult:
    working_image: Image.Image
    simulation: Image.Image
    class_indices: np.ndarray
    channels: dict[str, np.ndarray]
    palette: OverprintPalette
    coverages: np.ndarray | None = None


def smooth_texture(image: Image.Image, settings: Settings) -> Image.Image:
    if settings.texture_blur_radius <= 0 or settings.texture_amount >= 1.0:
        return image.copy()
    blurred = image.filter(
        ImageFilter.GaussianBlur(radius=settings.texture_blur_radius)
    )
    return Image.blend(image, blurred, 1.0 - settings.texture_amount)


def smooth_classes(indices: np.ndarray, size: int) -> np.ndarray:
    if size < 3:
        return indices
    if size % 2 == 0:
        raise ValueError("Die Klassenglättung muss ungerade sein.")
    return np.asarray(
        Image.fromarray(indices).filter(ImageFilter.ModeFilter(size=size)),
        dtype=np.uint8,
    )


def channels_from_classes(
    indices: np.ndarray,
    palette: OverprintPalette,
    inks: list[Ink],
) -> dict[str, np.ndarray]:
    active = palette.masks[indices]
    return {
        ink.name: (active[..., channel] * 255).astype(np.uint8)
        for channel, ink in enumerate(inks)
    }


def preview_dpi(image_size: tuple[int, int], settings: Settings) -> float:
    """Estimate the on-screen pixels per inch for the selected print framing."""
    width, height = image_size
    print_width_in = settings.print_width_cm / 2.54
    print_height_in = settings.print_height_cm / 2.54
    if settings.resize_mode == "pad":
        image_ratio = width / height
        output_ratio = print_width_in / print_height_in
        return (
            width / print_width_in
            if image_ratio > output_ratio
            else height / print_height_in
        )
    left, top, right, bottom = settings.crop_box
    horizontal = max(1.0, (right - left) * width) / print_width_in
    vertical = max(1.0, (bottom - top) * height) / print_height_in
    return (horizontal + vertical) / 2.0


def process_image(
    image: Image.Image,
    settings: Settings,
    inks: list[Ink],
    measured_lab: dict[int, tuple[float, float, float]] | None = None,
    effects: list[ImageEffect] | None = None,
    *,
    preview: bool = True,
) -> SeparationResult:
    working = (
        PreviewProcessor.create_working_image(
            image,
            (settings.preview_width, settings.preview_height),
        )
        if preview
        else image.copy()
    )
    adjusted = apply_effects(working, effects or [])
    working.close()
    working = adjusted
    smoothed = smooth_texture(working, settings)
    working.close()
    working = smoothed
    rgb = np.asarray(working, dtype=np.uint8)
    palette = build_overprint_palette(inks, settings.paper, measured_lab)
    classifier = ColorClassifier(palette, [ink.bias for ink in inks])
    coverages = None
    if settings.halftone_mode == "halftone":
        coverages = classifier.coverage_rgb_lut(rgb, settings.halftone_softness)
        coverages = smooth_coverages(coverages, settings.class_smooth_size)
        coverages = apply_tone_curve(
            coverages,
            settings.halftone_gamma,
            settings.halftone_min_dot,
            settings.halftone_max_dot,
        )
        masks = rasterize_coverages(
            coverages,
            dpi=preview_dpi(working.size, settings) if preview else settings.dpi,
            frequency_lpi=settings.halftone_frequency_lpi,
            angles=[ink.screen_angle for ink in inks],
            shape=settings.halftone_shape,
        )
        indices = state_indices_from_masks(masks, palette.masks)
        channels = {
            ink.name: (masks[..., channel] * 255).astype(np.uint8)
            for channel, ink in enumerate(inks)
        }
    else:
        if preview:
            indices = classifier.classify_rgb_lut(rgb)
        else:
            lab = ColorConverter.rgb_to_lab(rgb)
            indices = classifier.classify_tiled(lab, settings.tile_size)
        indices = smooth_classes(indices, settings.class_smooth_size)
        channels = channels_from_classes(indices, palette, inks)
    simulation = Simulation.render(indices, palette)
    return SeparationResult(
        working, simulation, indices, channels, palette, coverages
    )
