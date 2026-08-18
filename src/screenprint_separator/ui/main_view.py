from copy import deepcopy
from pathlib import Path

import numpy as np
from nicegui import events, run, ui
from PIL import Image, ImageOps

from screenprint_separator.export.plate_exporter import export_project
from screenprint_separator.models.ink import Ink
from screenprint_separator.models.project import Project
from screenprint_separator.models.settings import Settings
from screenprint_separator.persistence import SessionStore
from screenprint_separator.processing.color_converter import ColorConverter
from screenprint_separator.processing.color_library import (
    ColorEntry,
    ColorLibrary,
    color_from_cmyk,
)
from screenprint_separator.processing.image_loader import ImageLoader
from screenprint_separator.processing.palette import (
    build_overprint_palette,
    mixed_state_indices,
)
from screenprint_separator.processing.pipeline import (
    SeparationResult,
    adjust_input_image,
    process_image,
)


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#" + "".join(f"{value:02x}" for value in rgb)


def _ink_from_cmyk(
    name: str,
    cmyk: tuple[float, float, float, float],
) -> Ink:
    rgb, lab = color_from_cmyk(cmyk)
    return Ink(
        name=name,
        lab=lab,
        rgb_preview=rgb,
        cmyk=cmyk,
        opacity=0.72,
    )


def _ink_from_entry(entry: ColorEntry) -> Ink:
    return Ink(
        name=entry.name,
        pantone=entry.name if entry.system == "pantone" else "",
        lab=entry.lab,
        rgb_preview=entry.rgb,
        color_source="library",
        cmyk=entry.cmyk or (0.0, 0.0, 0.0, 0.0),
        library_id=entry.id,
        opacity=0.72,
    )


class MainView:
    def __init__(self) -> None:
        library_path = Path.cwd() / "color_library.json"
        if not library_path.exists():
            library_path = Path(__file__).resolve().parents[3] / "color_library.json"
        self.color_library = ColorLibrary.load(library_path)

        settings = Settings()
        paper_entry = self.color_library.by_id.get(settings.paper_id)
        if paper_entry is not None and paper_entry.system == "paper":
            settings.paper = paper_entry.rgb

        defaults = [
            entry for entry in self.color_library.entries if entry.system != "paper"
        ][:3]
        default_inks = (
            [_ink_from_entry(entry) for entry in defaults]
            if len(defaults) == 3
            else [
                _ink_from_cmyk("Farbe 1", (94.0, 0.0, 15.0, 81.0)),
                _ink_from_cmyk("Farbe 2", (0.0, 5.0, 39.0, 0.0)),
                _ink_from_cmyk("Farbe 3", (22.0, 16.0, 0.0, 0.0)),
            ]
        )
        fallback_project = Project(
            settings=settings,
            inks=default_inks,
        )
        self.session_store = SessionStore(library_path.parent)
        self.project, self._cached_filename = self.session_store.load(fallback_project)
        self._preview_busy = False
        self._preview_dirty = False
        self._ink_controls: dict[int, dict[str, object]] = {}
        self._plate_previews: list[tuple[Ink, object]] = []
        self._mixture_controls: dict[int, dict[str, object]] = {}
        self._syncing_mixture_controls = False
        self._syncing_paper_controls = False
        self._syncing_preview_size = False

        ui.add_css(
            """
            body { background: #f4f1ea; }
            .settings-card { background: white; border: 1px solid #ded8cc; }
            .preview-card { background: #252525; color: white; min-height: 420px; }
            .preview-image > img { object-fit: contain !important; }
            .input-uploader .q-uploader__list { display: none; }
            .warm-control {
                background: #fffdf8 !important;
                border: 1px solid #ded8cc !important;
                box-shadow: 0 2px 8px rgba(91, 72, 43, .11);
            }
            .warm-control:hover { border-color: #cbbfae !important; }
            .export-settings .q-field__label,
            .export-settings .q-field__native,
            .export-settings .q-field__input,
            .export-settings .q-field__marginal,
            .export-settings .q-field__suffix,
            .export-settings .q-field__prefix { color: white !important; }
            .export-settings .q-field__control:before {
                border-color: rgba(255, 255, 255, .35) !important;
            }
            .ink-drag-handle { cursor: grab; touch-action: none; }
            .ink-drag-handle:active { cursor: grabbing; }
            .layout-column { flex: 1 1 0; min-width: 280px; }
            .simulation-column { flex: 2 1 0; min-width: 460px; }
            """
        )

        with ui.column().classes("w-full max-w-[1900px] mx-auto p-4 gap-4"):
            ui.label("Screenprint Separator").classes("text-3xl font-bold")
            ui.label(
                "Eine bis fünf Druckfarben · automatische Überdruckzustände"
            ).classes("text-grey-7")

            with ui.row().classes("w-full items-start gap-4 flex-wrap xl:flex-nowrap"):
                with ui.column().classes(
                    "settings-card layout-column rounded-xl p-4 gap-4 w-full"
                ):
                    self._build_input_output_settings()

                with ui.column().classes(
                    "settings-card layout-column rounded-xl p-4 gap-4 w-full"
                ):
                    self._build_ink_settings()

                with ui.column().classes(
                    "preview-card simulation-column rounded-xl p-4 gap-3 w-full"
                ):
                    self._build_preview()

        self.preview_timer = ui.timer(
            0.3,
            self._run_scheduled_preview,
            active=False,
            immediate=False,
        )
        self._restore_cached_image()

    def _build_input_output_settings(self) -> None:
        ui.label("Eingabe").classes("text-xl font-semibold")
        self.upload = ui.upload(
            label="Bild laden",
            on_upload=self.load_image,
            auto_upload=True,
        ).props(
            "accept=.png,.jpg,.jpeg,.tif,.tiff flat bordered no-thumbnails"
        ).classes("w-full input-uploader")

        with ui.column().classes(
            "w-full gap-2 rounded-lg bg-green-1 text-green-9 p-2"
        ) as upload_status:
            self.input_preview = ui.image("").classes("w-full rounded-md")
            with ui.row().classes("w-full items-center gap-2 px-1"):
                ui.icon("check_circle").classes("text-green-7")
                self.upload_status_text = ui.label().classes("grow text-sm")
                ui.label("100 %").classes("text-sm font-medium")
            self.input_size = ui.label().classes("text-xs text-green-9 px-1")
        self.upload_status = upload_status
        self.upload_status.set_visibility(False)

        with ui.expansion("Helligkeit und Kontrast", icon="contrast").classes(
            "w-full"
        ):
            ui.label("Helligkeit").classes("text-sm")
            ui.slider(
                min=0.25,
                max=2.0,
                step=0.01,
                value=self.project.settings.brightness,
                on_change=lambda event: self._change_input_tone(
                    "brightness", event.value
                ),
            ).props("label-always")

            ui.label("Kontrast").classes("text-sm")
            ui.slider(
                min=0.25,
                max=2.0,
                step=0.01,
                value=self.project.settings.contrast,
                on_change=lambda event: self._change_input_tone(
                    "contrast", event.value
                ),
            ).props("label-always")

        with ui.expansion("Textur und Glättung", icon="texture").classes(
            "w-full"
        ):
            ui.label("Erhalt der Originaltextur").classes("text-sm")
            ui.slider(
                min=0.0,
                max=1.0,
                step=0.05,
                value=self.project.settings.texture_amount,
                on_change=lambda event: self._change_setting(
                    "texture_amount", event.value, float
                ),
            ).props("label-always")
            self._build_info_label(
                "Glättungsradius",
                "Weichzeichnet das Eingabebild vor der Separation. Höhere Werte "
                "reduzieren feine Details und Bildrauschen.",
            )
            ui.slider(
                min=0.0,
                max=5.0,
                step=0.25,
                value=self.project.settings.texture_blur_radius,
                on_change=lambda event: self._change_setting(
                    "texture_blur_radius", event.value, float
                ),
            ).props("label-always")
            self._build_info_label(
                "Klassenglättung",
                "Glättet die fertig zugeordneten Farbklassen. Größere Werte "
                "entfernen kleine isolierte Punkte, können aber Details verlieren.",
            )
            ui.select(
                [1, 3, 5, 7],
                value=self.project.settings.class_smooth_size,
                on_change=lambda event: self._change_setting(
                    "class_smooth_size", event.value, int
                ),
            ).props('aria-label="Klassenglättung"').classes("w-full")

        with ui.expansion("Simulationsvorschau", icon="photo_size_select_large").classes(
            "w-full"
        ):
            ui.label(
                "Breite und Höhe bleiben im Seitenverhältnis des Eingabebildes."
            ).classes("text-xs text-grey-7")
            with ui.row().classes("w-full gap-2"):
                self.preview_width_input = ui.number(
                    "Breite (px)",
                    value=self.project.settings.preview_width,
                    min=50,
                    max=5000,
                    step=1,
                    on_change=lambda event: self._change_preview_dimension(
                        "width", event.value
                    ),
                ).classes("grow")
                self.preview_height_input = ui.number(
                    "Höhe (px)",
                    value=self.project.settings.preview_height,
                    min=50,
                    max=5000,
                    step=1,
                    on_change=lambda event: self._change_preview_dimension(
                        "height", event.value
                    ),
                ).classes("grow")
            self.preview_width_input.set_enabled(self.project.image is not None)
            self.preview_height_input.set_enabled(self.project.image is not None)

    def _build_ink_settings(self) -> None:
        with ui.row().classes("w-full items-center"):
            ui.label("Druckfarben").classes("text-xl font-semibold grow")
            ui.button(icon="add", on_click=self._add_ink).props(
                "flat dense round"
            ).tooltip("Druckfarbe hinzufügen")

        paper_card = ui.expansion().props("dense expand-separator").classes(
            "warm-control w-full rounded-lg"
        )
        with (
            paper_card.add_slot("header"),
            ui.row().classes("w-full items-center gap-2 no-wrap"),
        ):
            ui.label("Papier").classes("font-medium w-16 shrink-0")
            self.paper_swatch = ui.element("div").classes("shrink-0")
            self.paper_summary = ui.label().classes("text-sm grow truncate")
        with paper_card:
            self.paper_source = ui.toggle(
                {"library": "Papier", "lab": "LAB-Referenzfarbe"},
                value=self.project.settings.paper_source,
                on_change=self._change_paper_source,
            ).props("spread no-caps").classes("w-full")
            with ui.column().classes("w-full px-2 pb-2") as paper_library_group:
                self.paper_select = ui.select(
                    self._paper_options(),
                    label="Papier aus JSON",
                    value=self.project.settings.paper_id,
                    on_change=self._change_paper_selection,
                ).classes("w-full")
            with (
                ui.column().classes("w-full px-2 pb-2") as paper_lab_group,
                ui.row().classes("w-full gap-2"),
            ):
                self.paper_lab_inputs = []
                for component, (caption, minimum, maximum) in enumerate(
                    (("L*", 0, 100), ("a*", -128, 127), ("b*", -128, 127))
                ):
                    field = ui.number(
                        caption,
                        value=self.project.settings.paper_lab[component],
                        min=minimum,
                        max=maximum,
                        step=0.1,
                        on_change=lambda event, component=component: (
                            self._change_paper_lab(component, event.value)
                        ),
                    ).classes("grow min-w-[70px]")
                    self.paper_lab_inputs.append(field)
            self.paper_library_group = paper_library_group
            self.paper_lab_group = paper_lab_group
        self._sync_paper_controls()

        ui.label(
            "CMYK direkt eingeben oder eine Farbe aus color_library.json wählen."
        ).classes("text-xs text-grey-7")

        with ui.column().classes("w-full gap-3") as ink_cards:
            self.ink_cards = ink_cards
            for ink in list(self.project.inks):
                self._build_ink_controls(ink)
        self.ink_cards.make_sortable(
            handle=".ink-drag-handle",
            on_end=self._drag_ink,
            ghost_class="opacity-40",
        )

        self.order_label = ui.label().classes("text-sm font-medium")
        self._update_order_label()
        with ui.column().classes("w-full gap-2") as overprint_container:
            self.overprint_container = overprint_container
            self._build_overprint_controls()
        ui.button(
            "Farbbibliothek neu laden",
            icon="refresh",
            on_click=self._reload_color_library,
        ).props("flat dense")

    def _build_ink_controls(self, ink: Ink) -> None:
        card = ui.expansion().props("dense expand-separator").classes(
            "warm-control w-full rounded-lg"
        )
        with (
            card.add_slot("header"),
            ui.row().classes("w-full items-center gap-2 no-wrap"),
        ):
                ui.icon("drag_indicator").classes(
                    "ink-drag-handle text-grey-6 shrink-0"
                )
                number = ui.label().classes("font-semibold w-5 shrink-0")
                summary_swatch = ui.element("div").classes("shrink-0")
                summary = ui.label().classes("text-sm grow truncate")
                opacity_label = ui.label().classes("text-sm font-medium shrink-0")
                ui.button(
                    icon="delete_outline",
                    color="negative",
                    on_click=lambda ink=ink: self._delete_ink(ink),
                ).props("flat dense round size=sm").tooltip("Druckfarbe löschen")

        with card:
            source_select = ui.toggle(
                {
                    "library": "Pantone",
                    "cmyk": "CMYK",
                },
                value=ink.color_source,
                on_change=lambda event, ink=ink: self._change_ink_source(
                    ink, event.value
                ),
            ).props("spread no-caps").classes("w-full")

            with ui.column().classes("w-full gap-2 px-2") as cmyk_group:
                name_input = ui.input(
                    "Bezeichnung",
                    value=ink.name,
                    on_change=lambda event, ink=ink: self._change_ink_name(
                        ink, event.value
                    ),
                ).classes("w-full")
                with ui.row().classes("w-full gap-2"):
                    cmyk_inputs = []
                    for index, label in enumerate(("C %", "M %", "Y %", "K %")):
                        field = ui.number(
                            label,
                            value=round(ink.cmyk[index], 1),
                            min=0,
                            max=100,
                            step=0.5,
                            on_change=lambda event, ink=ink, index=index: (
                                self._change_ink_cmyk(ink, index, event.value)
                            ),
                        ).classes("grow min-w-[65px]")
                        cmyk_inputs.append(field)

            with ui.column().classes("w-full gap-1 px-2") as library_group:
                palette_select = ui.select(
                    self._library_options(),
                    label="Pantone / gespeicherte Farbe",
                    value=ink.library_id or None,
                    with_input=True,
                    on_change=lambda event, ink=ink: self._change_library_color(
                        ink, event.value
                    ),
                ).classes("w-full")

            swatch = ui.element("div").style(
                self._swatch_style(ink.rgb_preview)
            ).classes("mx-2")

            self._ink_controls[id(ink)] = {
                "card": card,
                "number": number,
                "summary_swatch": summary_swatch,
                "summary": summary,
                "opacity_label": opacity_label,
                "source": source_select,
                "swatch": swatch,
                "cmyk_group": cmyk_group,
                "library_group": library_group,
                "name": name_input,
                "cmyk_inputs": cmyk_inputs,
                "palette": palette_select,
            }
            self._sync_ink_control_visibility(ink)

            with ui.element("div").classes("px-2"):
                self._build_info_label(
                    "Überdruckstärke",
                    "Legt die Deckkraft der obenliegenden Farbe bei automatisch "
                    "berechneten Überdruckfarben fest. Der Wert wirkt nur im "
                    "Automatikmodus; Pantone- und LAB-Messwerte bleiben unverändert.",
                )
            ui.slider(
                min=0.0,
                max=1.0,
                step=0.01,
                value=ink.opacity,
                on_change=lambda event, ink=ink: self._change_ink_value(
                    ink, "opacity", event.value
                ),
            ).props("label-always").classes("px-2")
            with ui.element("div").classes("px-2"):
                self._build_info_label(
                    "Klassifikations-Bias (ΔE)",
                    "Verschiebt die Farbzuordnung zugunsten oder zulasten dieser "
                    "Druckfarbe. Positive Werte bevorzugen Zustände mit dieser "
                    "Farbe; negative Werte machen ihre Auswahl zurückhaltender.",
                )
            ui.slider(
                min=-10.0,
                max=20.0,
                step=0.5,
                value=ink.bias,
                on_change=lambda event, ink=ink: self._change_ink_value(
                    ink, "bias", event.value
                ),
            ).props("label-always").classes("px-2")
        self._update_ink_summary(ink)

    def _build_overprint_controls(self) -> None:
        ui.separator()
        ui.label("Überdruckfarben").classes("text-lg font-semibold uppercase")
        self._ensure_measured_overprints()
        palette = self._automatic_palette()
        for plate_count in range(2, len(self.project.inks) + 1):
            states = tuple(
                state
                for state in mixed_state_indices(len(self.project.inks))
                if int(palette.masks[state].sum()) == plate_count
            )
            ui.label(f"{plate_count} Platten").classes(
                "text-xs font-medium text-grey-7 mt-1"
            )
            for state in states:
                row = ui.expansion().props("dense expand-separator").classes(
                    "warm-control w-full rounded-lg"
                )
                with (
                    row.add_slot("header"),
                    ui.row().classes("w-full items-center gap-2 no-wrap"),
                ):
                        combination = ui.label().classes("font-medium w-16 shrink-0")
                        swatch = ui.element("div").classes("shrink-0")
                        status = ui.label("Automatisch").classes("text-sm grow")
                with row:
                    mode = ui.toggle(
                        {
                            "automatic": "Automatisch",
                            "pantone": "Pantone (JSON)",
                            "lab": "LAB-Messwert",
                        },
                        value="automatic",
                        on_change=lambda event, state=state: (
                            self._change_overprint_state_mode(state, event.value)
                        ),
                    ).props("spread no-caps").classes("w-full")
                    with ui.column().classes("w-full px-2 pb-2") as pantone_group:
                        palette_select = ui.select(
                            self._library_options(),
                            label="Pantone / gespeicherte Farbe",
                            with_input=True,
                            on_change=lambda event, state=state: (
                                self._change_overprint_library(state, event.value)
                            ),
                        ).classes("w-full")
                    with (
                        ui.column().classes("w-full px-2 pb-2") as lab_group,
                        ui.row().classes("w-full gap-2"),
                    ):
                            lab_inputs = []
                            for component, (caption, minimum, maximum) in enumerate(
                                (("L*", 0, 100), ("a*", -128, 127), ("b*", -128, 127))
                            ):
                                field = ui.number(
                                    caption,
                                    min=minimum,
                                    max=maximum,
                                    step=0.1,
                                    on_change=lambda event, state=state, component=component: (
                                        self._change_mixture_lab(
                                            state, component, event.value
                                        )
                                    ),
                                ).classes("grow min-w-[70px]")
                                lab_inputs.append(field)
                self._mixture_controls[state] = {
                    "combination": combination,
                    "swatch": swatch,
                    "status": status,
                    "mode": mode,
                    "pantone_group": pantone_group,
                    "palette": palette_select,
                    "lab_group": lab_group,
                    "lab": lab_inputs,
                }
                pantone_group.set_visibility(False)
                lab_group.set_visibility(False)
        self._sync_overprint_controls()

    def _automatic_palette(self):
        return build_overprint_palette(
            self.project.inks,
            self.project.settings.paper,
        )

    def _ensure_measured_overprints(self) -> None:
        palette = self._automatic_palette()
        for state in mixed_state_indices(len(self.project.inks)):
            self.project.measured_overprints.setdefault(
                state,
                tuple(float(value) for value in palette.lab[state]),
            )

    def _sync_mixture_controls(self) -> None:
        if not self._mixture_controls:
            return
        self._syncing_mixture_controls = True
        try:
            automatic = self._automatic_palette()
            for state, controls in self._mixture_controls.items():
                indices = [
                    str(index + 1)
                    for index, active in enumerate(automatic.masks[state])
                    if active
                ]
                controls["combination"].set_text("+".join(indices))
                lab = self.project.measured_overprints[state]
                source = self.project.overprint_sources.get(state, "automatic")
                rgb = (
                    ColorConverter.lab_to_rgb(np.asarray(lab, dtype=np.float32))
                    if source != "automatic"
                    else automatic.rgb[state]
                )
                rgb_tuple = tuple(int(value) for value in rgb)
                controls["swatch"].style(
                    replace=(
                        f"background:{_rgb_to_hex(rgb_tuple)};width:34px;height:22px;"
                        "border-radius:4px;border:1px solid rgba(0,0,0,.25)"
                    )
                )
                status = {
                    "automatic": "Automatisch",
                    "pantone": "Pantone",
                    "lab": "LAB-Messwert",
                }[source]
                controls["status"].set_text(status)
                controls["mode"].set_value(source)
                controls["pantone_group"].set_visibility(source == "pantone")
                controls["lab_group"].set_visibility(source == "lab")
                controls["palette"].set_value(
                    self.project.overprint_library_ids.get(state)
                )
                for field, component in zip(controls["lab"], lab, strict=True):
                    field.set_value(round(float(component), 2))
        finally:
            self._syncing_mixture_controls = False

    def _sync_automatic_mixture_controls(self) -> None:
        self._sync_mixture_controls()

    def _sync_overprint_controls(self) -> None:
        self._sync_mixture_controls()

    def _change_overprint_state_mode(self, state: int, value: str | None) -> None:
        if self._syncing_mixture_controls or value not in {
            "automatic",
            "pantone",
            "lab",
        }:
            return
        self.project.overprint_sources[state] = value
        if value == "automatic":
            self.project.manual_overprint_states.discard(state)
        else:
            self.project.manual_overprint_states.add(state)
        self._sync_mixture_controls()
        self.schedule_preview()

    def _change_overprint_library(self, state: int, identifier: str | None) -> None:
        if self._syncing_mixture_controls or not identifier:
            return
        entry = self.color_library.by_id.get(identifier)
        if entry is None or entry.system == "paper":
            ui.notify("Unbekannte Überdruckfarbe", type="negative")
            return
        self.project.overprint_library_ids[state] = identifier
        self.project.measured_overprints[state] = entry.lab
        self.project.overprint_sources[state] = "pantone"
        self.project.manual_overprint_states.add(state)
        self._sync_mixture_controls()
        self.schedule_preview()

    def _change_mixture_lab(
        self, state: int, component: int, value: float | None
    ) -> None:
        if self._syncing_mixture_controls or value is None:
            return
        lab = list(self.project.measured_overprints[state])
        limits = ((0.0, 100.0), (-128.0, 127.0), (-128.0, 127.0))
        lab[component] = float(np.clip(value, *limits[component]))
        self.project.measured_overprints[state] = tuple(lab)
        self.project.overprint_sources[state] = "lab"
        self.project.manual_overprint_states.add(state)
        self._sync_mixture_controls()
        self.schedule_preview()

    def _build_preview(self) -> None:
        with ui.row().classes("w-full items-center"):
            ui.label("Simulation und Ausgabe").classes("text-xl font-semibold grow")
            with ui.element("div").classes(
                "w-5 h-5 shrink-0 flex items-center justify-center"
            ):
                self.preview_spinner = ui.spinner(size="sm")
                self.preview_spinner.set_visibility(False)

        self.status = ui.label("Noch kein Bild geladen").classes("text-grey-4")
        self.preview = ui.interactive_image("").classes(
            "preview-image rounded-lg self-center"
        ).style("width: 100%; max-width: 100%")
        self.preview.set_visibility(False)
        self.preview_size = ui.label("Vorschaugröße: –").classes("text-sm text-grey-4")
        self.preview_size.set_visibility(False)

        self.palette_row = ui.row().classes("w-full gap-2 flex-wrap")

        with ui.expansion("Plattenvorschau", icon="layers").classes("w-full"):
            self.plate_preview_row = ui.row().classes("w-full gap-3 flex-wrap")
            self._rebuild_plate_previews()

        self._build_export_settings()

    def _rebuild_plate_previews(self) -> None:
        self._plate_previews = []
        self.plate_preview_row.clear()
        with self.plate_preview_row:
            for index, ink in enumerate(self.project.inks, start=1):
                with ui.column().classes("grow min-w-[150px]"):
                    ui.label(f"{index} · {ink.name}").classes("font-medium")
                    plate = ui.image("").classes("w-full rounded")
                    plate.set_visibility(False)
                    self._plate_previews.append((ink, plate))

    def _build_export_settings(self) -> None:
        ui.separator().classes("bg-grey-7")
        ui.label("Export").classes("text-lg font-semibold")
        with ui.expansion("Druckformat und Skalierung", icon="straighten").classes(
            "export-settings w-full text-white"
        ):
            with ui.row().classes("w-full"):
                ui.number(
                    "Breite (cm)",
                    value=self.project.settings.print_width_cm,
                    min=1,
                    on_change=lambda event: self._change_export_setting(
                        "print_width_cm", event.value, float
                    ),
                ).classes("grow")
                ui.number(
                    "Höhe (cm)",
                    value=self.project.settings.print_height_cm,
                    min=1,
                    on_change=lambda event: self._change_export_setting(
                        "print_height_cm", event.value, float
                    ),
                ).classes("grow")
            with ui.row().classes("w-full"):
                with ui.column().classes("grow gap-0"):
                    self._build_info_label(
                        "Arbeits-DPI",
                        "Auflösung, in der das Bild beim Export tatsächlich getrennt "
                        "und klassifiziert wird. Mehr DPI liefern feinere Details, "
                        "benötigen aber deutlich mehr Zeit und Speicher.",
                    )
                    ui.number(
                        value=self.project.settings.dpi,
                        min=72,
                        on_change=lambda event: self._change_export_setting(
                            "dpi", event.value, int
                        ),
                    ).props('aria-label="Arbeits-DPI"').classes("w-full")
                with ui.column().classes("grow gap-0"):
                    self._build_info_label(
                        "Ausgabe-DPI",
                        "Pixelauflösung der finalen 1-Bit-TIFF-Druckplatten. Von der "
                        "Arbeitsauflösung wird mit dem gewählten Algorithmus auf diese "
                        "Größe skaliert.",
                    )
                    ui.number(
                        value=self.project.settings.output_dpi,
                        min=72,
                        on_change=lambda event: self._change_export_setting(
                            "output_dpi", event.value, int
                        ),
                    ).props('aria-label="Ausgabe-DPI"').classes("w-full")
            ui.select(
                {"fit": "Format füllen", "pad": "Einpassen mit Rand"},
                label="Separationsskalierung",
                value=self.project.settings.resize_mode,
                on_change=lambda event: self._change_export_setting(
                    "resize_mode", event.value, str
                ),
            ).classes("w-full")
            self._build_info_label(
                "Upscaling-Algorithmus",
                "Bestimmt, wie die separierten Platten von Arbeits-DPI auf "
                "Ausgabe-DPI vergrößert werden. Nearest Neighbour erhält harte "
                "Pixelkanten; Lanczos, bikubisch und bilinear interpolieren.",
            )
            ui.select(
                {
                    "nearest": "Pixelgenau (Nearest Neighbour)",
                    "bilinear": "Bilinear",
                    "bicubic": "Bikubisch",
                    "lanczos": "Lanczos",
                },
                value=self.project.settings.upscale_algorithm,
                on_change=lambda event: self._change_export_setting(
                    "upscale_algorithm", event.value, str
                ),
            ).props('aria-label="Upscaling-Algorithmus"').classes("w-full")
            self.export_size_label = ui.label().classes("text-sm text-white")
            self._update_export_size_label()

        self.export_button = ui.button(
            "Simulation und Platten exportieren",
            icon="download",
            on_click=self.export,
        ).classes("w-full")
        self.export_button.disable()

    def _change_setting(
        self,
        name: str,
        value: object,
        converter: type = float,
        refresh: bool = True,
    ) -> None:
        if value is None:
            return
        setattr(self.project.settings, name, converter(value))
        self._save_session()
        if refresh:
            self.schedule_preview()

    def _change_export_setting(
        self,
        name: str,
        value: object,
        converter: type,
    ) -> None:
        if value is None:
            return
        setattr(self.project.settings, name, converter(value))
        self._update_export_size_label()
        self._save_session()

    def _update_export_size_label(self) -> None:
        settings = self.project.settings
        work_width = round(settings.print_width_cm / 2.54 * settings.dpi)
        work_height = round(settings.print_height_cm / 2.54 * settings.dpi)
        output_width = round(settings.print_width_cm / 2.54 * settings.output_dpi)
        output_height = round(settings.print_height_cm / 2.54 * settings.output_dpi)
        self.export_size_label.set_text(
            f"Separation: {work_width} × {work_height} px · "
            f"Ausgabe: {output_width} × {output_height} px"
        )

    def _change_input_tone(self, name: str, value: float | None) -> None:
        if value is None:
            return
        setattr(self.project.settings, name, float(value))
        self._refresh_input_preview()
        self.schedule_preview()

    def _change_preview_dimension(
        self,
        dimension: str,
        value: float | None,
    ) -> None:
        if self._syncing_preview_size or value is None or self.project.image is None:
            return
        image_width, image_height = self.project.image.size
        if dimension == "width":
            width = int(np.clip(round(value), 50, 5000))
            height = max(1, round(width * image_height / image_width))
        else:
            height = int(np.clip(round(value), 50, 5000))
            width = max(1, round(height * image_width / image_height))
        self._set_preview_dimensions(width, height)
        self.schedule_preview()

    def _set_preview_dimensions(self, width: int, height: int) -> None:
        self.project.settings.preview_width = width
        self.project.settings.preview_height = height
        self._syncing_preview_size = True
        try:
            self.preview_width_input.set_value(width)
            self.preview_height_input.set_value(height)
        finally:
            self._syncing_preview_size = False

    def _refresh_input_preview(self) -> None:
        if self.project.image is None:
            return
        preview = self.project.image.copy()
        preview.thumbnail((800, 800), Image.Resampling.LANCZOS)
        preview = adjust_input_image(preview, self.project.settings)
        self.input_preview.set_source(preview)

    def _change_paper_selection(
        self,
        event: events.ValueChangeEventArguments,
    ) -> None:
        if self._syncing_paper_controls:
            return
        identifier = event.value
        entry = self.color_library.by_id.get(identifier)
        if entry is None or entry.system != "paper":
            ui.notify("Unbekannte Papierfarbe", type="negative")
            return
        self.project.settings.paper_id = entry.id
        self.project.settings.paper = entry.rgb
        self.project.settings.paper_lab = entry.lab
        self.project.settings.paper_source = "library"
        self._sync_paper_controls()
        self._sync_automatic_mixture_controls()
        self.schedule_preview()

    def _change_paper_source(self, event: events.ValueChangeEventArguments) -> None:
        if self._syncing_paper_controls or event.value not in {"library", "lab"}:
            return
        self.project.settings.paper_source = event.value
        if event.value == "library":
            entry = self.color_library.by_id.get(self.project.settings.paper_id)
            if entry is not None and entry.system == "paper":
                self.project.settings.paper = entry.rgb
                self.project.settings.paper_lab = entry.lab
        else:
            rgb = ColorConverter.lab_to_rgb(
                np.asarray(self.project.settings.paper_lab, dtype=np.float32)
            )
            self.project.settings.paper = tuple(int(value) for value in rgb)
        self._sync_paper_controls()
        self._sync_automatic_mixture_controls()
        self.schedule_preview()

    def _change_paper_lab(self, component: int, value: float | None) -> None:
        if self._syncing_paper_controls or value is None:
            return
        lab = list(self.project.settings.paper_lab)
        limits = ((0.0, 100.0), (-128.0, 127.0), (-128.0, 127.0))
        lab[component] = float(np.clip(value, *limits[component]))
        self.project.settings.paper_lab = tuple(lab)
        rgb = ColorConverter.lab_to_rgb(np.asarray(lab, dtype=np.float32))
        self.project.settings.paper = tuple(int(component) for component in rgb)
        self.project.settings.paper_source = "lab"
        self._sync_paper_controls()
        self._sync_automatic_mixture_controls()
        self.schedule_preview()

    def _sync_paper_controls(self) -> None:
        self._syncing_paper_controls = True
        try:
            source = self.project.settings.paper_source
            entry = self.color_library.by_id.get(self.project.settings.paper_id)
            self.paper_summary.set_text(
                entry.name
                if source == "library" and entry is not None
                else "LAB-Referenzfarbe"
            )
            self.paper_swatch.style(
                replace=(
                    f"background:{_rgb_to_hex(self.project.settings.paper)};"
                    "width:34px;height:22px;border-radius:4px;"
                    "border:1px solid rgba(0,0,0,.25)"
                )
            )
            self.paper_source.set_value(source)
            self.paper_library_group.set_visibility(source == "library")
            self.paper_lab_group.set_visibility(source == "lab")
            self.paper_select.set_value(self.project.settings.paper_id)
            for field, value in zip(
                self.paper_lab_inputs,
                self.project.settings.paper_lab,
                strict=True,
            ):
                field.set_value(round(float(value), 2))
        finally:
            self._syncing_paper_controls = False

    def _change_ink_name(self, ink: Ink, value: str | None) -> None:
        if not value:
            return
        ink.name = value
        self._update_ink_summary(ink)
        self._update_order_label()
        self._sync_overprint_controls()
        self.schedule_preview()

    @staticmethod
    def _build_info_label(text: str, tooltip: str) -> None:
        with ui.row().classes("items-center gap-1"):
            ui.label(text).classes("text-sm")
            ui.icon("info_outline").classes(
                "text-grey-6 text-base cursor-help"
            ).tooltip(tooltip)

    @staticmethod
    def _swatch_style(rgb: tuple[int, int, int]) -> str:
        return (
            f"background:{_rgb_to_hex(rgb)};height:36px;"
            "width:calc(100% - 16px);border-radius:6px;"
            "border:1px solid rgba(0,0,0,.25)"
        )

    def _library_options(self) -> dict[str, str]:
        return {
            entry.id: (
                f"PANTONE · {entry.name}"
                if entry.system == "pantone"
                else entry.name
            )
            for entry in self.color_library.entries
            if entry.system != "paper"
        }

    def _paper_options(self) -> dict[str, str]:
        options = {
            entry.id: entry.name
            for entry in self.color_library.entries
            if entry.system == "paper"
        }
        return options

    def _sync_ink_control_visibility(self, ink: Ink) -> None:
        controls = self._ink_controls[id(ink)]
        controls["cmyk_group"].set_visibility(ink.color_source == "cmyk")
        controls["library_group"].set_visibility(ink.color_source == "library")

    def _update_ink_summary(self, ink: Ink) -> None:
        controls = self._ink_controls[id(ink)]
        position = self.project.inks.index(ink)
        controls["number"].set_text(str(position + 1))
        if ink.color_source == "library":
            name = ink.name
            descriptor = name if name.upper().startswith("PANTONE") else f"PANTONE {name}"
        else:
            values = "/".join(f"{value:g}" for value in ink.cmyk)
            descriptor = f"CMYK {values}"
        controls["summary"].set_text(descriptor)
        controls["opacity_label"].set_text(f"{round(ink.opacity * 100)} %")
        controls["summary_swatch"].style(
            replace=(
                f"background:{_rgb_to_hex(ink.rgb_preview)};width:34px;height:22px;"
                "border-radius:4px;border:1px solid rgba(0,0,0,.25)"
            )
        )

    def _change_ink_source(self, ink: Ink, value: str | None) -> None:
        if value not in {"cmyk", "library"}:
            return
        ink.color_source = value
        if value == "cmyk":
            ink.library_id = ""
            ink.pantone = ""
            ink.rgb_preview, ink.lab = color_from_cmyk(ink.cmyk)
            self._update_ink_swatch(ink)
        elif ink.library_id in self.color_library.by_id:
            self._apply_library_entry(ink, self.color_library.by_id[ink.library_id])
        self._sync_ink_control_visibility(ink)
        self._update_ink_summary(ink)
        self._sync_automatic_mixture_controls()
        self.schedule_preview()

    def _change_ink_cmyk(
        self,
        ink: Ink,
        index: int,
        value: float | None,
    ) -> None:
        if value is None:
            return
        components = list(ink.cmyk)
        components[index] = float(np.clip(value, 0.0, 100.0))
        ink.cmyk = tuple(components)
        ink.rgb_preview, ink.lab = color_from_cmyk(ink.cmyk)
        ink.color_source = "cmyk"
        ink.library_id = ""
        ink.pantone = ""
        self._update_ink_swatch(ink)
        self._update_ink_summary(ink)
        self._sync_automatic_mixture_controls()
        self.schedule_preview()

    def _change_library_color(self, ink: Ink, identifier: str | None) -> None:
        if not identifier:
            return
        entry = self.color_library.by_id.get(identifier)
        if entry is None:
            ui.notify(f"Unbekannte Bibliotheksfarbe: {identifier}", type="negative")
            return
        self._apply_library_entry(ink, entry)
        self._sync_automatic_mixture_controls()
        self.schedule_preview()

    def _apply_library_entry(self, ink: Ink, entry: ColorEntry) -> None:
        ink.name = entry.name
        ink.pantone = entry.name if entry.system == "pantone" else ""
        ink.color_source = "library"
        ink.library_id = entry.id
        ink.lab = entry.lab
        ink.rgb_preview = entry.rgb
        if entry.cmyk is not None:
            ink.cmyk = entry.cmyk

        controls = self._ink_controls[id(ink)]
        controls["name"].set_value(ink.name)
        controls["source"].set_value("library")
        controls["palette"].set_value(entry.id)
        if entry.cmyk is not None:
            for field, component in zip(
                controls["cmyk_inputs"],
                entry.cmyk,
                strict=True,
            ):
                field.set_value(component)
        self._update_ink_swatch(ink)
        self._update_ink_summary(ink)
        self._sync_ink_control_visibility(ink)
        self._update_order_label()
        self._sync_overprint_controls()

    def _update_ink_swatch(self, ink: Ink) -> None:
        self._ink_controls[id(ink)]["swatch"].style(
            replace=self._swatch_style(ink.rgb_preview)
        )

    def _reload_color_library(self) -> None:
        try:
            self.color_library = ColorLibrary.load(self.color_library.path)
        except (OSError, TypeError, ValueError) as error:
            ui.notify(f"Farbbibliothek konnte nicht geladen werden: {error}", type="negative")
            return

        options = self._library_options()
        for ink in self.project.inks:
            controls = self._ink_controls[id(ink)]
            value = ink.library_id if ink.library_id in self.color_library.by_id else None
            controls["palette"].set_options(options, value=value)
            if value:
                self._apply_library_entry(ink, self.color_library.by_id[value])
        for state, controls in self._mixture_controls.items():
            identifier = self.project.overprint_library_ids.get(state)
            value = identifier if identifier in self.color_library.by_id else None
            controls["palette"].set_options(options, value=value)
            if identifier and value is None:
                self.project.overprint_library_ids.pop(state, None)
                self.project.overprint_sources[state] = "automatic"
                self.project.manual_overprint_states.discard(state)

        paper_options = self._paper_options()
        paper_id = self.project.settings.paper_id
        paper_entry = self.color_library.by_id.get(paper_id)
        if paper_entry is None or paper_entry.system != "paper":
            paper_id = "paper-bright-white"
            paper_entry = self.color_library.by_id.get(paper_id)
            self.project.settings.paper_id = paper_id
        if self.project.settings.paper_source == "library" and paper_entry is not None:
            self.project.settings.paper = paper_entry.rgb
            self.project.settings.paper_lab = paper_entry.lab
        self.paper_select.set_options(paper_options, value=paper_id)
        self._sync_paper_controls()

        ink_count = sum(
            entry.system != "paper" for entry in self.color_library.entries
        )
        paper_count = sum(
            entry.system == "paper" for entry in self.color_library.entries
        )
        ui.notify(
            f"{ink_count} Druckfarben und {paper_count} Papierfarben geladen",
            type="positive",
        )
        self._sync_overprint_controls()
        self.schedule_preview()

    def _change_ink_value(self, ink: Ink, name: str, value: float | None) -> None:
        if value is None:
            return
        setattr(ink, name, float(value))
        if name == "opacity":
            self._update_ink_summary(ink)
            self._sync_automatic_mixture_controls()
        self.schedule_preview()

    def _capture_overprints_by_inks(self) -> dict[frozenset[int], dict[str, object]]:
        palette = self._automatic_palette()
        captured = {}
        for state in mixed_state_indices(len(self.project.inks)):
            key = frozenset(
                id(ink)
                for active, ink in zip(
                    palette.masks[state], self.project.inks, strict=True
                )
                if active
            )
            captured[key] = {
                "lab": self.project.measured_overprints.get(state),
                "manual": state in self.project.manual_overprint_states,
                "source": self.project.overprint_sources.get(state),
                "library_id": self.project.overprint_library_ids.get(state),
            }
        return captured

    def _restore_overprints_by_inks(
        self,
        captured: dict[frozenset[int], dict[str, object]],
    ) -> None:
        palette = self._automatic_palette()
        self.project.measured_overprints = {}
        self.project.manual_overprint_states = set()
        self.project.overprint_sources = {}
        self.project.overprint_library_ids = {}
        for state in mixed_state_indices(len(self.project.inks)):
            key = frozenset(
                id(ink)
                for active, ink in zip(
                    palette.masks[state], self.project.inks, strict=True
                )
                if active
            )
            values = captured.get(key)
            if values is None:
                continue
            if values["lab"] is not None:
                self.project.measured_overprints[state] = values["lab"]
            if values["manual"]:
                self.project.manual_overprint_states.add(state)
            if values["source"] is not None:
                self.project.overprint_sources[state] = values["source"]
            if values["library_id"] is not None:
                self.project.overprint_library_ids[state] = values["library_id"]

    def _rebuild_overprint_controls(self) -> None:
        self._mixture_controls = {}
        self.overprint_container.clear()
        with self.overprint_container:
            self._build_overprint_controls()

    def _add_ink(self) -> None:
        if len(self.project.inks) >= 5:
            ui.notify("Es sind maximal fünf Druckfarben möglich.", type="warning")
            return
        captured = self._capture_overprints_by_inks()
        ink = _ink_from_cmyk(
            f"Farbe {len(self.project.inks) + 1}",
            (0.0, 0.0, 0.0, 100.0),
        )
        self.project.inks.append(ink)
        with self.ink_cards:
            self._build_ink_controls(ink)
        self._restore_overprints_by_inks(captured)
        self._rebuild_overprint_controls()
        self._rebuild_plate_previews()
        self._update_order_label()
        self.schedule_preview()

    def _delete_ink(self, ink: Ink) -> None:
        if len(self.project.inks) <= 1:
            ui.notify("Mindestens eine Druckfarbe muss erhalten bleiben.", type="warning")
            return
        captured = self._capture_overprints_by_inks()
        self.project.inks.remove(ink)
        controls = self._ink_controls.pop(id(ink))
        controls["card"].delete()
        self._restore_overprints_by_inks(captured)
        for remaining in self.project.inks:
            self._update_ink_summary(remaining)
        self._rebuild_overprint_controls()
        self._rebuild_plate_previews()
        self._update_order_label()
        self.schedule_preview()

    def _drag_ink(self, event: events.SortableEventArguments) -> None:
        if event.old_index == event.new_index:
            return
        ink = self.project.inks[event.old_index]
        direction = 1 if event.new_index > event.old_index else -1
        while self.project.inks.index(ink) != event.new_index:
            self._move_ink(ink, direction)

    def _move_ink(self, ink: Ink, direction: int) -> None:
        index = self.project.inks.index(ink)
        target = index + direction
        if not 0 <= target < len(self.project.inks):
            return
        masks = self._automatic_palette().masks
        measured_by_inks = {
            frozenset(
                id(candidate)
                for active, candidate in zip(masks[state], self.project.inks, strict=True)
                if active
            ): lab
            for state, lab in self.project.measured_overprints.items()
        }
        manual_ink_sets = {
            frozenset(
                id(candidate)
                for active, candidate in zip(masks[state], self.project.inks, strict=True)
                if active
            )
            for state in self.project.manual_overprint_states
        }
        source_by_inks = {
            frozenset(
                id(candidate)
                for active, candidate in zip(masks[state], self.project.inks, strict=True)
                if active
            ): source
            for state, source in self.project.overprint_sources.items()
        }
        library_by_inks = {
            frozenset(
                id(candidate)
                for active, candidate in zip(masks[state], self.project.inks, strict=True)
                if active
            ): identifier
            for state, identifier in self.project.overprint_library_ids.items()
        }
        self.project.inks[index], self.project.inks[target] = (
            self.project.inks[target],
            self.project.inks[index],
        )
        for position, ordered_ink in enumerate(self.project.inks):
            self._ink_controls[id(ordered_ink)]["card"].move(
                self.ink_cards,
                position,
            )
            self._update_ink_summary(ordered_ink)
        remapped_manual_states = set()
        remapped_sources = {}
        remapped_library_ids = {}
        for state in mixed_state_indices(len(self.project.inks)):
            key = frozenset(
                id(candidate)
                for active, candidate in zip(masks[state], self.project.inks, strict=True)
                if active
            )
            if key in measured_by_inks:
                self.project.measured_overprints[state] = measured_by_inks[key]
            if key in manual_ink_sets:
                remapped_manual_states.add(state)
            if key in source_by_inks:
                remapped_sources[state] = source_by_inks[key]
            if key in library_by_inks:
                remapped_library_ids[state] = library_by_inks[key]
        self.project.manual_overprint_states = remapped_manual_states
        self.project.overprint_sources = remapped_sources
        self.project.overprint_library_ids = remapped_library_ids
        self._update_order_label()
        self._sync_overprint_controls()
        self.schedule_preview()

    def _active_measured_overprints(
        self,
    ) -> dict[int, tuple[float, float, float]] | None:
        measured = {
            state: self.project.measured_overprints[state]
            for state in self.project.manual_overprint_states
            if state in self.project.measured_overprints
        }
        return measured or None

    def _update_order_label(self) -> None:
        if not hasattr(self, "order_label"):
            return
        order = " → ".join(ink.name for ink in self.project.inks)
        self.order_label.set_text(f"Druckreihenfolge: {order}")

    async def load_image(self, event: events.UploadEventArguments) -> None:
        try:
            image_bytes = await event.file.read()
            self.project.image, self.project.rgb_array = ImageLoader.load(image_bytes)
        except (OSError, ValueError) as error:
            ui.notify(f"Bild konnte nicht geladen werden: {error}", type="negative")
            return

        width, height = self.project.image.size
        preview_width = self.project.settings.preview_width
        self._set_preview_dimensions(
            preview_width,
            max(1, round(preview_width * height / width)),
        )
        self.preview_width_input.enable()
        self.preview_height_input.enable()
        self._cached_filename = event.file.name
        try:
            self.session_store.save_image(self.project.image)
        except OSError as error:
            ui.notify(f"Bildcache konnte nicht gespeichert werden: {error}", type="warning")
        size = self._format_file_size(len(image_bytes))
        self.upload_status_text.set_text(f"{event.file.name} · {size}")
        self._refresh_input_preview()
        self.upload_status.set_visibility(True)
        self.input_size.set_text(f"Originalgröße: {width} × {height} px")
        self.export_button.enable()
        self._save_session()
        await self.refresh_preview()

    def _restore_cached_image(self) -> None:
        if self.project.image is None:
            return
        filename = self._cached_filename or "Letztes Eingabebild"
        try:
            byte_count = self.session_store.image_path.stat().st_size
            size = self._format_file_size(byte_count)
        except OSError:
            size = "Cache"
        width, height = self.project.image.size
        preview_width = self.project.settings.preview_width
        self._set_preview_dimensions(
            preview_width,
            max(1, round(preview_width * height / width)),
        )
        self.preview_width_input.enable()
        self.preview_height_input.enable()
        self.upload_status_text.set_text(f"{filename} · {size}")
        self._refresh_input_preview()
        self.upload_status.set_visibility(True)
        self.input_size.set_text(f"Originalgröße: {width} × {height} px")
        self.export_button.enable()
        self.schedule_preview()

    def _save_session(self) -> None:
        try:
            self.session_store.save(self.project, self._cached_filename)
        except (OSError, TypeError, ValueError):
            pass

    @staticmethod
    def _format_file_size(size: int) -> str:
        value = float(size)
        for unit in ("B", "KB", "MB", "GB"):
            if value < 1024.0 or unit == "GB":
                precision = 0 if unit == "B" else 1
                return f"{value:.{precision}f} {unit}"
            value /= 1024.0
        return f"{size} B"

    def schedule_preview(self) -> None:
        self._save_session()
        if self.project.image is None or not hasattr(self, "preview_timer"):
            return
        self._preview_dirty = True
        self.preview_timer.activate()

    async def _run_scheduled_preview(self) -> None:
        self.preview_timer.deactivate()
        if not self._preview_dirty:
            return
        self._preview_dirty = False
        await self.refresh_preview()

    async def refresh_preview(self) -> None:
        if self.project.image is None:
            return
        if self._preview_busy:
            self._preview_dirty = True
            return

        self._preview_busy = True
        self.preview_spinner.set_visibility(True)
        self.status.set_text("Farbklassifikation wird berechnet …")

        try:
            result = await run.io_bound(
                process_image,
                self.project.image.copy(),
                deepcopy(self.project.settings),
                deepcopy(self.project.inks),
                deepcopy(self._active_measured_overprints()),
            )
            if result is None:
                return
            self._show_result(result)
        except (OSError, ValueError) as error:
            self.status.set_text("Vorschau fehlgeschlagen")
            ui.notify(str(error), type="negative")
        finally:
            self.preview_spinner.set_visibility(False)
            self._preview_busy = False
            if self._preview_dirty:
                self.preview_timer.activate()

    def _palette_tooltip(self, state: int, name: str) -> str:
        if state == 0:
            if self.project.settings.paper_source == "lab":
                return "Papier · LAB-Referenzfarbe"
            entry = self.color_library.by_id.get(self.project.settings.paper_id)
            return f"Papier · {entry.name if entry is not None else 'Papierfarbe'}"
        if state <= len(self.project.inks):
            ink = self.project.inks[state - 1]
            if ink.color_source == "library":
                value = ink.name
                if not value.upper().startswith("PANTONE"):
                    value = f"PANTONE {value}"
            else:
                components = "/".join(f"{value:g}" for value in ink.cmyk)
                value = f"CMYK {components}"
            return f"{ink.name} · {value}"

        source = self.project.overprint_sources.get(state, "automatic")
        if source == "pantone":
            identifier = self.project.overprint_library_ids.get(state)
            entry = self.color_library.by_id.get(identifier)
            value = entry.name if entry is not None else "Pantone"
            if not value.upper().startswith("PANTONE"):
                value = f"PANTONE {value}"
        elif source == "lab":
            value = "LAB-Messwert"
        else:
            value = "Automatisch"
        return f"{name} · {value}"

    def _show_result(self, result: SeparationResult) -> None:
        self.project.preview_image = result.simulation
        self.project.preview_array = np.asarray(result.simulation)
        self.project.class_indices = result.class_indices
        self.project.channels = result.channels

        self.preview.set_source(result.simulation)
        self.preview.set_visibility(True)
        self.status.set_text("Simulation aktuell")
        width, height = result.simulation.size
        aspect_ratio = width / height
        self.preview.style(
            replace=(
                f"width: min(100%, calc(70vh * {aspect_ratio:.8f})); "
                f"max-width: 100%; aspect-ratio: {width} / {height}"
            )
        )
        self.preview_size.set_text(f"Vorschaugröße: {width} × {height} px")
        self.preview_size.set_visibility(True)

        self.palette_row.clear()
        with self.palette_row:
            previous_plate_count = None
            for state, (name, rgb, mask) in enumerate(zip(
                result.palette.names,
                result.palette.rgb,
                result.palette.masks,
                strict=True,
            )):
                plate_count = int(mask.sum())
                if (
                    previous_plate_count is not None
                    and plate_count != previous_plate_count
                ):
                    ui.element("div").classes(
                        "h-12 border-l border-grey-6 mx-1 self-center"
                    )
                previous_plate_count = plate_count
                color = _rgb_to_hex(tuple(int(value) for value in rgb))
                active = [
                    str(index + 1)
                    for index, is_active in enumerate(mask)
                    if is_active
                ]
                number = "+".join(active) if active else "Papier"
                tooltip = self._palette_tooltip(state, name)
                with ui.column().classes("gap-0 items-center"):
                    ui.element("div").style(
                        f"background:{color};width:48px;height:32px;border-radius:6px;"
                        "border:1px solid rgba(255,255,255,.35)"
                    ).tooltip(tooltip)
                    ui.label(number).classes("text-[10px]").tooltip(tooltip)

        for ink, image_element in self._plate_previews:
            channel = result.channels.get(ink.name)
            if channel is None:
                image_element.set_visibility(False)
                continue
            plate = ImageOps.invert(Image.fromarray(channel, mode="L"))
            image_element.set_source(plate)
            image_element.set_visibility(True)

    async def export(self) -> None:
        if self.project.image is None:
            ui.notify("Bitte zuerst ein Bild laden.", type="warning")
            return

        self.export_button.disable()
        notification = ui.notification(
            "Export wird in voller Auflösung berechnet …",
            spinner=True,
            timeout=None,
        )
        try:
            archive = await run.io_bound(
                export_project,
                self.project.image.copy(),
                deepcopy(self.project.settings),
                deepcopy(self.project.inks),
                deepcopy(self._active_measured_overprints()),
            )
            if archive is not None:
                ui.download(archive, filename="screenprint_export.zip")
                notification.message = "Export ist fertig."
                notification.spinner = False
                notification.type = "positive"
                notification.timeout = 4
                notification.update()
        except (OSError, ValueError) as error:
            notification.dismiss()
            ui.notify(f"Export fehlgeschlagen: {error}", type="negative")
        finally:
            self.export_button.enable()
