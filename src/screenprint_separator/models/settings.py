from dataclasses import dataclass


@dataclass
class Settings:
    print_width_cm: float = 34.0
    print_height_cm: float = 38.0
    dpi: int = 300
    output_dpi: int = 600
    resize_mode: str = "fit"
    preview_max_size: int = 1200
    tile_size: int = 512

    texture_amount: float = 0.45
    texture_blur_radius: float = 2.0
    class_smooth_size: int = 3
    threshold: int = 128
    trapping_px: int = 0

    paper_id: str = "paper-warm-gray"
    paper: tuple[int, int, int] = (191, 188, 181)
