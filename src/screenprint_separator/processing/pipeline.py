from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from screenprint_separator.models.ink import Ink
from screenprint_separator.models.settings import Settings
from screenprint_separator.processing.color_classifier import ColorClassifier
from screenprint_separator.processing.color_converter import ColorConverter
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


def adjust_input_image(image: Image.Image, settings: Settings) -> Image.Image:
    """Apply the non-destructive input tone controls."""
    adjusted = ImageEnhance.Brightness(image).enhance(settings.brightness)
    return ImageEnhance.Contrast(adjusted).enhance(settings.contrast)


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


def process_image(
    image: Image.Image,
    settings: Settings,
    inks: list[Ink],
    measured_lab: dict[int, tuple[float, float, float]] | None = None,
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
    working = adjust_input_image(working, settings)
    working = smooth_texture(working, settings)
    rgb = np.asarray(working, dtype=np.uint8)
    palette = build_overprint_palette(inks, settings.paper, measured_lab)
    classifier = ColorClassifier(palette, [ink.bias for ink in inks])
    if preview:
        indices = classifier.classify_rgb_lut(rgb)
    else:
        lab = ColorConverter.rgb_to_lab(rgb)
        indices = classifier.classify_tiled(lab, settings.tile_size)
    indices = smooth_classes(indices, settings.class_smooth_size)
    channels = channels_from_classes(indices, palette, inks)
    simulation = Simulation.render(indices, palette)
    return SeparationResult(working, simulation, indices, channels, palette)
