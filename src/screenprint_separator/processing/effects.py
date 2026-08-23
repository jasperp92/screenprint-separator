import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from screenprint_separator.models.effect import ImageEffect
from screenprint_separator.processing.color_converter import ColorConverter


def _apply_saturation_vibrance(
    image: Image.Image,
    effect: ImageEffect,
) -> Image.Image:
    saturation = float(np.clip(effect.value("saturation"), 0.0, 2.0))
    vibrance = float(np.clip(effect.value("vibrance"), 0.0, 2.0))
    saturated = ImageEnhance.Color(image).enhance(saturation)
    if abs(vibrance - 1.0) < 1e-7:
        return saturated

    rgb = np.asarray(saturated, dtype=np.float32) / 255.0
    gray = np.sum(
        rgb * np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32),
        axis=-1,
        keepdims=True,
    )
    chroma = np.max(rgb, axis=-1, keepdims=True) - np.min(
        rgb, axis=-1, keepdims=True
    )
    scale = 1.0 + (vibrance - 1.0) * (1.0 - chroma)
    result = gray + (rgb - gray) * scale
    saturated.close()
    return Image.fromarray(
        np.clip(np.rint(result * 255.0), 0, 255).astype(np.uint8),
        mode="RGB",
    )


def _apply_levels(image: Image.Image, effect: ImageEffect) -> Image.Image:
    black_point = float(np.clip(effect.value("black_point"), 0.0, 254.0))
    white_point = float(np.clip(effect.value("white_point"), 1.0, 255.0))
    if white_point <= black_point:
        white_point = min(255.0, black_point + 1.0)
    gamma = float(np.clip(effect.value("gamma"), 0.1, 3.0))

    rgb = np.asarray(image, dtype=np.float32)
    normalized = np.clip(
        (rgb - black_point) / (white_point - black_point),
        0.0,
        1.0,
    )
    corrected = np.power(normalized, 1.0 / gamma)
    return Image.fromarray(
        np.clip(np.rint(corrected * 255.0), 0, 255).astype(np.uint8),
        mode="RGB",
    )


def _apply_hue(image: Image.Image, effect: ImageEffect) -> Image.Image:
    degrees = float(np.clip(effect.value("degrees"), -180.0, 180.0))
    hsv_image = image.convert("HSV")
    hsv = np.asarray(hsv_image, dtype=np.uint8).copy()
    hsv_image.close()
    offset = round(degrees / 360.0 * 255.0)
    hue = hsv[..., 0].astype(np.int16)
    hsv[..., 0] = np.mod(hue + offset, 256).astype(np.uint8)
    shifted = Image.fromarray(hsv, mode="HSV")
    try:
        return shifted.convert("RGB")
    finally:
        shifted.close()


def _apply_selective_color(image: Image.Image, effect: ImageEffect) -> Image.Image:
    """Remove chroma around a perceptual CIELAB target color."""
    rgb = np.asarray(image, dtype=np.uint8)
    lab = ColorConverter.rgb_to_lab(rgb)
    target_rgb = np.asarray(
        [
            np.clip(effect.value("target_r"), 0.0, 255.0),
            np.clip(effect.value("target_g"), 0.0, 255.0),
            np.clip(effect.value("target_b"), 0.0, 255.0),
        ],
        dtype=np.uint8,
    )
    target_lab = ColorConverter.rgb_to_lab(target_rgb)
    distance = np.sqrt(np.sum((lab - target_lab) ** 2, axis=-1))
    tolerance = float(np.clip(effect.value("tolerance"), 0.0, 100.0))
    softness = float(np.clip(effect.value("softness"), 0.0, 100.0))
    amount = float(np.clip(effect.value("amount"), 0.0, 1.0))

    if softness <= 1e-7:
        selection = (distance <= tolerance).astype(np.float32)
    else:
        selection = np.clip(
            (tolerance + softness - distance) / softness,
            0.0,
            1.0,
        )
        selection = selection * selection * (3.0 - 2.0 * selection)

    neutral_lab = lab.copy()
    neutral_lab[..., 1:] = 0.0
    neutral_rgb = ColorConverter.lab_to_rgb(neutral_lab).astype(np.float32)
    blend = (selection * amount)[..., np.newaxis]
    result = rgb.astype(np.float32) + (neutral_rgb - rgb) * blend
    return Image.fromarray(
        np.clip(np.rint(result), 0, 255).astype(np.uint8),
        mode="RGB",
    )


def apply_effect(image: Image.Image, effect: ImageEffect) -> Image.Image:
    if effect.kind == "saturation_vibrance":
        return _apply_saturation_vibrance(image, effect)
    if effect.kind == "brightness_contrast":
        brightness = float(np.clip(effect.value("brightness"), 0.25, 2.0))
        contrast = float(np.clip(effect.value("contrast"), 0.25, 2.0))
        brightened = ImageEnhance.Brightness(image).enhance(brightness)
        try:
            return ImageEnhance.Contrast(brightened).enhance(contrast)
        finally:
            brightened.close()
    if effect.kind == "black_white":
        amount = float(np.clip(effect.value("amount"), 0.0, 1.0))
        grayscale = ImageOps.grayscale(image).convert("RGB")
        try:
            return Image.blend(image, grayscale, amount)
        finally:
            grayscale.close()
    if effect.kind == "levels":
        return _apply_levels(image, effect)
    if effect.kind == "hue":
        return _apply_hue(image, effect)
    if effect.kind == "selective_color":
        return _apply_selective_color(image, effect)
    raise ValueError(f"Unbekannter Bildeffekt: {effect.kind}")


def apply_effects(image: Image.Image, effects: list[ImageEffect]) -> Image.Image:
    current = image.copy()
    for effect in effects:
        adjusted = apply_effect(current, effect)
        current.close()
        current = adjusted
    return current
