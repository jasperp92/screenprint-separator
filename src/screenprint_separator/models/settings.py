from dataclasses import dataclass


@dataclass
class Settings:
    print_width_cm: float = 34.0
    print_height_cm: float = 38.0
    dpi: int = 300
    output_dpi: int = 600
    resize_mode: str = "fit"
    upscale_algorithm: str = "nearest"
    preview_max_size: int = 1200
    preview_width: int = 1200
    preview_height: int = 1200
    tile_size: int = 512

    texture_amount: float = 0.45
    texture_blur_radius: float = 2.0
    class_smooth_size: int = 3
    threshold: int = 128
    trapping_px: int = 0
    brightness: float = 1.0
    contrast: float = 1.0

    paper_id: str = "paper-bright-white"
    paper: tuple[int, int, int] = (255, 255, 255)
    paper_source: str = "library"
    paper_lab: tuple[float, float, float] = (100.0, 0.0, 0.0)
    overprint_mode: str = "automatic"
