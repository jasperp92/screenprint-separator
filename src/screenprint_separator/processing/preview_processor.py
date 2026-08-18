from PIL import Image


class PreviewProcessor:
    @staticmethod
    def create_working_image(
        image: Image.Image,
        size: tuple[int, int] = (1400, 1400),
    ) -> Image.Image:
        return image.resize(size, Image.Resampling.LANCZOS)
