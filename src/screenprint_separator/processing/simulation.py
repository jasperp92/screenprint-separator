import numpy as np
from PIL import Image, ImageFilter

from screenprint_separator.processing.palette import OverprintPalette


class Simulation:
    @staticmethod
    def render(class_indices: np.ndarray, palette: OverprintPalette) -> Image.Image:
        rgb = palette.rgb[np.asarray(class_indices, dtype=np.uint8)]
        return Image.fromarray(rgb, mode="RGB")

    @staticmethod
    def render_trapped(
        class_indices: np.ndarray,
        palette: OverprintPalette,
        radius: float,
    ) -> Image.Image:
        """Render a preview with rounded expansion of every ink channel."""
        indices = np.asarray(class_indices, dtype=np.uint8)
        if radius <= 0:
            return Simulation.render(indices, palette)

        ink_count = palette.masks.shape[1]
        state_codes = np.zeros(indices.shape, dtype=np.uint8)
        blur_radius = max(0.5, radius / 2.0)

        for channel in range(ink_count):
            active = (palette.masks[indices, channel] * 255).astype(np.uint8)
            mask = Image.fromarray(active, mode="L")
            try:
                expanded_image = mask.filter(
                    ImageFilter.GaussianBlur(radius=blur_radius)
                )
                try:
                    # This matches the low cutoff used by the export renderer:
                    # with sigma=radius/2 it expands a solid edge by roughly radius.
                    expanded = np.asarray(expanded_image) > 6
                finally:
                    expanded_image.close()
            finally:
                mask.close()
            state_codes |= expanded.astype(np.uint8) << channel

        bit_weights = 1 << np.arange(ink_count, dtype=np.uint8)
        palette_codes = palette.masks @ bit_weights
        state_lookup = np.zeros(1 << ink_count, dtype=np.uint8)
        state_lookup[palette_codes] = np.arange(len(palette.masks), dtype=np.uint8)
        return Simulation.render(state_lookup[state_codes], palette)
