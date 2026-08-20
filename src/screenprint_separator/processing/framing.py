from collections.abc import Sequence

CropBox = tuple[float, float, float, float]


def normalize_crop_box(box: Sequence[float]) -> CropBox:
    """Return a normalized, ordered crop box within the unit square."""
    if len(box) != 4:
        return (0.0, 0.0, 1.0, 1.0)
    left, top, right, bottom = (float(value) for value in box)
    left, right = sorted((max(0.0, min(1.0, left)), max(0.0, min(1.0, right))))
    top, bottom = sorted((max(0.0, min(1.0, top)), max(0.0, min(1.0, bottom))))
    if right - left <= 1e-6 or bottom - top <= 1e-6:
        return (0.0, 0.0, 1.0, 1.0)
    return (left, top, right, bottom)


def fit_crop_box(
    image_size: tuple[int, int],
    output_size: tuple[float, float],
) -> CropBox:
    """Return the centered source area retained by a cover/fit resize."""
    image_width, image_height = image_size
    output_width, output_height = output_size
    if min(image_width, image_height, output_width, output_height) <= 0:
        return (0.0, 0.0, 1.0, 1.0)

    image_ratio = image_width / image_height
    output_ratio = output_width / output_height
    if image_ratio > output_ratio:
        retained_width = output_ratio / image_ratio
        left = (1.0 - retained_width) / 2.0
        return (left, 0.0, 1.0 - left, 1.0)
    retained_height = image_ratio / output_ratio
    top = (1.0 - retained_height) / 2.0
    return (0.0, top, 1.0, 1.0 - top)


def crop_aspect_ratio(box: Sequence[float], image_size: tuple[int, int]) -> float:
    """Return the selected source aspect ratio in image pixels."""
    left, top, right, bottom = normalize_crop_box(box)
    image_width, image_height = image_size
    return ((right - left) * image_width) / ((bottom - top) * image_height)


def crop_box_pixels(
    box: Sequence[float],
    image_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Convert a normalized crop box to Pillow source coordinates."""
    left, top, right, bottom = normalize_crop_box(box)
    width, height = image_size
    return (left * width, top * height, right * width, bottom * height)


def constrain_crop_box_to_aspect(
    box: Sequence[float],
    image_size: tuple[int, int],
    output_size: tuple[float, float],
) -> CropBox:
    """Adjust a crop around its center to match an output aspect ratio."""
    left, top, right, bottom = normalize_crop_box(box)
    image_width, image_height = image_size
    output_width, output_height = output_size
    normalized_ratio = (output_width / output_height) / (
        image_width / image_height
    )
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    area = (right - left) * (bottom - top)
    crop_width = (area * normalized_ratio) ** 0.5
    crop_height = crop_width / normalized_ratio
    scale = min(
        1.0,
        2.0 * min(center_x, 1.0 - center_x) / crop_width,
        2.0 * min(center_y, 1.0 - center_y) / crop_height,
    )
    crop_width *= scale
    crop_height *= scale
    return normalize_crop_box(
        (
            center_x - crop_width / 2.0,
            center_y - crop_height / 2.0,
            center_x + crop_width / 2.0,
            center_y + crop_height / 2.0,
        )
    )


def move_crop_box(box: Sequence[float], dx: float, dy: float) -> CropBox:
    """Move a normalized crop box without letting it leave the image."""
    left, top, right, bottom = normalize_crop_box(box)
    dx = max(-left, min(1.0 - right, dx))
    dy = max(-top, min(1.0 - bottom, dy))
    return (left + dx, top + dy, right + dx, bottom + dy)


def resize_crop_box(
    box: Sequence[float],
    action: str,
    point: tuple[float, float],
    *,
    aspect_ratio: float | None = None,
    minimum_size: tuple[float, float] = (0.01, 0.01),
) -> CropBox:
    """Resize a crop from an edge/corner, optionally locking its aspect."""
    left, top, right, bottom = normalize_crop_box(box)
    point_x = max(0.0, min(1.0, point[0]))
    point_y = max(0.0, min(1.0, point[1]))
    minimum_width, minimum_height = minimum_size

    if aspect_ratio is None:
        if "left" in action:
            left = min(point_x, right - minimum_width)
        if "right" in action:
            right = max(point_x, left + minimum_width)
        if "top" in action:
            top = min(point_y, bottom - minimum_height)
        if "bottom" in action:
            bottom = max(point_y, top + minimum_height)
        return normalize_crop_box((left, top, right, bottom))

    ratio = max(1e-6, aspect_ratio)
    horizontal = "left" in action or "right" in action
    vertical = "top" in action or "bottom" in action
    if horizontal and vertical:
        anchor_x = right if "left" in action else left
        anchor_y = bottom if "top" in action else top
        sign_x = -1.0 if "left" in action else 1.0
        sign_y = -1.0 if "top" in action else 1.0
        height_from_x = (point_x - anchor_x) * sign_x / ratio
        height_from_y = (point_y - anchor_y) * sign_y
        crop_height = max(minimum_height, (height_from_x + height_from_y) / 2.0)
        max_width = anchor_x if sign_x < 0 else 1.0 - anchor_x
        max_height = anchor_y if sign_y < 0 else 1.0 - anchor_y
        crop_height = min(crop_height, max_height, max_width / ratio)
        crop_width = crop_height * ratio
        new_x = anchor_x + crop_width * sign_x
        new_y = anchor_y + crop_height * sign_y
        return normalize_crop_box(
            (
                min(anchor_x, new_x),
                min(anchor_y, new_y),
                max(anchor_x, new_x),
                max(anchor_y, new_y),
            )
        )

    if horizontal:
        anchor_x = right if "left" in action else left
        sign_x = -1.0 if "left" in action else 1.0
        center_y = (top + bottom) / 2.0
        crop_width = max(minimum_width, (point_x - anchor_x) * sign_x)
        max_width = anchor_x if sign_x < 0 else 1.0 - anchor_x
        max_height = 2.0 * min(center_y, 1.0 - center_y)
        crop_width = min(crop_width, max_width, max_height * ratio)
        crop_height = crop_width / ratio
        new_x = anchor_x + crop_width * sign_x
        return normalize_crop_box(
            (
                min(anchor_x, new_x),
                center_y - crop_height / 2.0,
                max(anchor_x, new_x),
                center_y + crop_height / 2.0,
            )
        )

    anchor_y = bottom if "top" in action else top
    sign_y = -1.0 if "top" in action else 1.0
    center_x = (left + right) / 2.0
    crop_height = max(minimum_height, (point_y - anchor_y) * sign_y)
    max_height = anchor_y if sign_y < 0 else 1.0 - anchor_y
    max_width = 2.0 * min(center_x, 1.0 - center_x)
    crop_height = min(crop_height, max_height, max_width / ratio)
    crop_width = crop_height * ratio
    new_y = anchor_y + crop_height * sign_y
    return normalize_crop_box(
        (
            center_x - crop_width / 2.0,
            min(anchor_y, new_y),
            center_x + crop_width / 2.0,
            max(anchor_y, new_y),
        )
    )
