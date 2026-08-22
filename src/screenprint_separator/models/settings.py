from dataclasses import dataclass


@dataclass
class Settings:
    print_width_cm: float = 34.0
    print_height_cm: float = 38.0
    dpi: int = 300
    output_dpi: int = 600
    resize_mode: str = "fit"
    crop_box: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    preview_max_size: int = 1200
    preview_width: int = 1200
    preview_height: int = 1200
    tile_size: int = 512

    texture_amount: float = 0.45
    texture_blur_radius: float = 2.0
    class_smooth_size: int = 3
    threshold: int = 128
    trapping_mm: float = 0.0
    halftone_mode: str = "solid"
    halftone_frequency_lpi: float = 45.0
    halftone_shape: str = "circle"
    halftone_softness: float = 4.0
    halftone_gamma: float = 1.0
    halftone_min_dot: float = 0.02
    halftone_max_dot: float = 0.98
    brightness: float = 1.0
    contrast: float = 1.0

    paper_id: str = "paper-bright-white"
    paper: tuple[int, int, int] = (255, 255, 255)
    paper_source: str = "library"
    paper_cmyk: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    paper_lab: tuple[float, float, float] = (100.0, 0.0, 0.0)
    overprint_mode: str = "automatic"
