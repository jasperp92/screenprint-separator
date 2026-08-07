from PIL import Image


class PreviewProcessor:
    @staticmethod
    def create_working_image(
        image: Image.Image,
        max_size: int = 1400,
    ) -> Image.Image:
        preview = image.copy()

        preview.thumbnail(
            (max_size, max_size),
            Image.Resampling.LANCZOS,
        )

        return preview
