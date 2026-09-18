import json
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
from PIL import Image

from screenprint_separator.models.effect import EFFECT_DEFAULTS, ImageEffect
from screenprint_separator.models.ink import Ink
from screenprint_separator.models.project import Project
from screenprint_separator.models.settings import Settings


class SessionStore:
    def __init__(self, root: Path) -> None:
        self.directory = root / ".screenprint_separator_cache"
        self.state_path = self.directory / "session.json"
        self.image_path = self.directory / "input.png"

    def load(self, fallback: Project) -> tuple[Project, str | None]:
        if not self.state_path.exists():
            return fallback, None
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            allowed_settings = {field.name for field in fields(Settings)}
            settings_data = {
                key: value
                for key, value in data.get("settings", {}).items()
                if key in allowed_settings
            }
            for key in ("paper", "paper_cmyk", "paper_lab", "crop_box"):
                if key in settings_data:
                    settings_data[key] = tuple(settings_data[key])
            settings = Settings(**settings_data)

            inks = []
            default_angles = (15.0, 75.0, 45.0, 0.0, 30.0)
            allowed_ink_fields = {field.name for field in fields(Ink)}
            for index, values in enumerate(data.get("inks", [])):
                values = dict(values)
                values = {
                    key: value
                    for key, value in values.items()
                    if key in allowed_ink_fields
                }
                for key in ("lab", "rgb_preview", "cmyk"):
                    values[key] = tuple(values[key])
                values.setdefault(
                    "screen_angle", default_angles[index % len(default_angles)]
                )
                inks.append(Ink(**values))
            if not 1 <= len(inks) <= 5:
                inks = fallback.inks

            effects = []
            stored_effects = data.get("effects")
            if isinstance(stored_effects, list):
                for values in stored_effects:
                    if not isinstance(values, dict):
                        continue
                    kind = values.get("kind")
                    parameters = values.get("parameters", {})
                    if kind not in EFFECT_DEFAULTS or not isinstance(parameters, dict):
                        continue
                    normalized = {
                        name: float(parameters.get(name, default))
                        for name, default in EFFECT_DEFAULTS[kind].items()
                    }
                    effects.append(ImageEffect(kind=kind, parameters=normalized))
            elif settings.brightness != 1.0 or settings.contrast != 1.0:
                effects.append(
                    ImageEffect(
                        kind="brightness_contrast",
                        parameters={
                            "brightness": settings.brightness,
                            "contrast": settings.contrast,
                        },
                    )
                )

            project = Project(settings=settings, inks=inks, effects=effects)
            project.measured_overprints = {
                int(state): tuple(values)
                for state, values in data.get("measured_overprints", {}).items()
            }
            project.manual_overprint_states = {
                int(state) for state in data.get("manual_overprint_states", [])
            }
            project.overprint_sources = {
                int(state): source
                for state, source in data.get("overprint_sources", {}).items()
            }
            project.overprint_library_ids = {
                int(state): identifier
                for state, identifier in data.get("overprint_library_ids", {}).items()
            }
            project.overprint_biases = {
                int(state): float(bias)
                for state, bias in data.get("overprint_biases", {}).items()
            }
            if data.get("version", 1) < 2:
                self._migrate_paper_states(project)
            if self.image_path.exists():
                with Image.open(self.image_path) as cached:
                    project.image = cached.convert("RGB")
                project.rgb_array = np.asarray(project.image)
            return project, data.get("filename")
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
            return fallback, None

    @staticmethod
    def _migrate_paper_states(project: Project) -> None:
        """Merge the former optional paper variants into physical plate states."""
        count = 2 ** len(project.inks)
        mappings = (
            project.measured_overprints,
            project.overprint_sources,
            project.overprint_library_ids,
            project.overprint_biases,
        )
        states = set().union(*mappings, project.manual_overprint_states)
        winners = {}
        for state in sorted(states):
            if not 1 <= state < 2 * count - 1:
                continue
            target = state if state < count else state - count + 1
            priority = (state in project.manual_overprint_states, state >= count)
            if target not in winners or priority > winners[target][0]:
                winners[target] = (priority, state)
        for mapping in mappings:
            remapped = {
                target: mapping[state]
                for target, (_, state) in winners.items()
                if state in mapping
            }
            mapping.clear()
            mapping.update(remapped)
        project.manual_overprint_states = {
            target for target, (_, state) in winners.items()
            if state in project.manual_overprint_states
        }

    def save(self, project: Project, filename: str | None) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "filename": filename,
            "settings": asdict(project.settings),
            "inks": [asdict(ink) for ink in project.inks],
            "effects": [asdict(effect) for effect in project.effects],
            "measured_overprints": project.measured_overprints,
            "manual_overprint_states": sorted(project.manual_overprint_states),
            "overprint_sources": project.overprint_sources,
            "overprint_library_ids": project.overprint_library_ids,
            "overprint_biases": project.overprint_biases,
        }
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.state_path)

    def save_image(self, image: Image.Image) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        image.save(self.image_path, format="PNG")
