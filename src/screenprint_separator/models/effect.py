from dataclasses import dataclass, field

EFFECT_DEFAULTS: dict[str, dict[str, float]] = {
    "saturation_vibrance": {
        "saturation": 1.0,
        "vibrance": 1.0,
    },
    "brightness_contrast": {
        "brightness": 1.0,
        "contrast": 1.0,
    },
    "black_white": {
        "amount": 1.0,
    },
    "levels": {
        "black_point": 0.0,
        "gamma": 1.0,
        "white_point": 255.0,
    },
    "hue": {
        "degrees": 0.0,
    },
    "selective_color": {
        "target_r": 32.0,
        "target_g": 72.0,
        "target_b": 48.0,
        "tolerance": 18.0,
        "softness": 12.0,
        "amount": 1.0,
    },
}

EFFECT_NAMES = {
    "saturation_vibrance": "Sättigung / Dynamik",
    "brightness_contrast": "Helligkeit / Kontrast",
    "black_white": "Schwarzweiß",
    "levels": "Tonwertkorrektur",
    "hue": "Farbton",
    "selective_color": "Selektive Farbe",
}


@dataclass
class ImageEffect:
    kind: str
    parameters: dict[str, float] = field(default_factory=dict)

    @classmethod
    def create(cls, kind: str) -> "ImageEffect":
        if kind not in EFFECT_DEFAULTS:
            raise ValueError(f"Unbekannter Bildeffekt: {kind}")
        return cls(kind=kind, parameters=dict(EFFECT_DEFAULTS[kind]))

    def value(self, name: str) -> float:
        return float(self.parameters.get(name, EFFECT_DEFAULTS[self.kind][name]))
