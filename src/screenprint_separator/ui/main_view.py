import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
from nicegui import events, run, ui
from nicegui.elements.image import pil_to_tempfile
from PIL import Image, ImageOps

from screenprint_separator.export.plate_exporter import (
    export_project,
    save_export_archive,
)
from screenprint_separator.models.effect import EFFECT_NAMES, ImageEffect
from screenprint_separator.models.ink import Ink
from screenprint_separator.models.project import Project
from screenprint_separator.models.settings import Settings
from screenprint_separator.persistence import SessionStore
from screenprint_separator.processing.color_converter import ColorConverter
from screenprint_separator.processing.color_library import (
    ColorEntry,
    ColorLibrary,
    color_from_cmyk,
    rgb_to_cmyk,
)
from screenprint_separator.processing.effects import apply_effect, apply_effects
from screenprint_separator.processing.framing import (
    constrain_crop_box_to_aspect,
    crop_aspect_ratio,
    crop_box_pixels,
    fit_crop_box,
    move_crop_box,
    normalize_crop_box,
    resize_crop_box,
)
from screenprint_separator.processing.halftone import HALFTONE_SHAPES
from screenprint_separator.processing.image_loader import ImageLoader
from screenprint_separator.processing.palette import (
    OverprintPalette,
    build_overprint_palette,
    mixed_state_indices,
)
from screenprint_separator.processing.pipeline import (
    SeparationResult,
    process_image,
)
from screenprint_separator.processing.simulation import Simulation

DEFAULT_SCREEN_ANGLES = (15.0, 75.0, 45.0, 0.0, 30.0)


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#" + "".join(f"{value:02x}" for value in rgb)


def _hex_to_rgb(value: str | None) -> tuple[int, int, int] | None:
    if not value or len(value) != 7 or not value.startswith("#"):
        return None
    try:
        return tuple(int(value[index : index + 2], 16) for index in (1, 3, 5))
    except ValueError:
        return None


def _set_pil_source(element: object, image: Image.Image) -> None:
    """Give NiceGUI a temporary path instead of retaining a PIL object."""
    element.set_source(pil_to_tempfile(image, "PNG"))


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
        bundle_root = Path(getattr(sys, "_MEIPASS", Path.cwd()))
        library_candidates = (
            Path.cwd() / "color_library.json",
            bundle_root / "color_library.json",
            Path(__file__).resolve().parents[3] / "color_library.json",
        )
        library_path = next(
            (path for path in library_candidates if path.exists()),
            library_candidates[-1],
        )
        self.color_library = ColorLibrary.load(library_path)

        settings = Settings()
        paper_entry = self.color_library.by_id.get(settings.paper_id)
        if paper_entry is not None and paper_entry.system == "paper":
            settings.paper = paper_entry.rgb
            settings.paper_cmyk = rgb_to_cmyk(paper_entry.rgb)
            settings.paper_lab = paper_entry.lab

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
        for index, ink in enumerate(default_inks):
            ink.screen_angle = DEFAULT_SCREEN_ANGLES[index]
        fallback_project = Project(
            settings=settings,
            inks=default_inks,
        )
        session_root = (
            Path.home() / ".screenprint_separator"
            if getattr(sys, "frozen", False)
            else library_path.parent
        )
        self.session_store = SessionStore(session_root)
        self.project, self._cached_filename = self.session_store.load(fallback_project)
        if self.project.settings.paper_source == "rgb":
            # Migrate sessions created before paper CMYK controls were available.
            self.project.settings.paper_cmyk = rgb_to_cmyk(
                self.project.settings.paper
            )
            self.project.settings.paper_source = "cmyk"
        elif self.project.settings.paper_source not in {"library", "cmyk", "lab"}:
            self.project.settings.paper_source = "library"
        self._preview_busy = False
        self._preview_dirty = False
        self._effect_controls: dict[int, dict[str, object]] = {}
        self._ink_controls: dict[int, dict[str, object]] = {}
        self._plate_previews: list[tuple[Ink, object]] = []
        self._mixture_controls: dict[int, dict[str, object]] = {}
        self._syncing_mixture_controls = False
        self._syncing_paper_controls = False
        self._syncing_ink_controls = False
        self._syncing_preview_size = False
        self._syncing_print_size = False
        self._syncing_effect_controls = False
        self._eyedropper_radius = 0
        self._eyedropper_target: tuple[str, object | None] | None = None
        self._eyedropper_original_color: str | None = None
        self._eyedropper_hover_color: str | None = None
        self._input_preview_size: tuple[int, int] | None = None
        self._crop_drag_start: tuple[float, float] | None = None
        self._crop_drag_original: tuple[float, float, float, float] | None = None
        self._crop_drag_action: str | None = None
        self._crop_cursor_class: str | None = None
        self._crop_active = False
        self._base_preview_simulation: Image.Image | None = None
        self._preview_palette: OverprintPalette | None = None

        ui.add_css(
            """
            body { background: #f4f1ea; }
            .settings-card { background: white; border: 1px solid #ded8cc; }
            .preview-card { background: #252525; color: white; min-height: 420px; }
            .preview-image > img { object-fit: contain !important; }
            .preview-image.crop-active > img { cursor: crosshair; }
            .preview-image.crop-cursor-move > img { cursor: move; }
            .preview-image.crop-cursor-ew > img { cursor: ew-resize; }
            .preview-image.crop-cursor-ns > img { cursor: ns-resize; }
            .preview-image.crop-cursor-nwse > img { cursor: nwse-resize; }
            .preview-image.crop-cursor-nesw > img { cursor: nesw-resize; }
            .input-preview.eyedropper-active > img {
                cursor: crosshair !important;
            }
            .plate-preview-row {
                display: flex !important;
                flex-wrap: nowrap !important;
                align-items: flex-start;
                width: 100%;
            }
            .plate-preview-item {
                flex: 1 1 0 !important;
                width: 0;
                min-width: 0 !important;
            }
            .plate-preview-item .q-img { width: 100% !important; }
            .plate-preview-label {
                max-width: 100%;
                overflow: hidden;
                white-space: nowrap;
                text-overflow: ellipsis;
            }
            .preview-image.crop-active { touch-action: none; user-select: none; }
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
            .ink-card > .q-expansion-item__container > .q-item {
                min-width: 0;
                padding-left: 8px;
                padding-right: 4px;
            }
            .ink-card > .q-expansion-item__container > .q-item
            > .q-item__section--main {
                min-width: 0;
                overflow: hidden;
            }
            .ink-card > .q-expansion-item__container > .q-item
            > .q-item__section--side {
                min-width: 24px;
                padding-left: 2px;
                padding-right: 0;
            }
            .ink-card-header {
                width: 100%;
                min-width: 0;
                max-width: 100%;
                overflow: hidden;
            }
            .ink-card-summary {
                min-width: 0;
                overflow: hidden;
                white-space: nowrap;
                text-overflow: ellipsis;
            }
            .effect-drag-handle { cursor: grab; touch-action: none; }
            .effect-drag-handle:active { cursor: grabbing; }
            .effect-card > .q-expansion-item__container > .q-item {
                min-width: 0;
                padding-left: 8px;
                padding-right: 4px;
            }
            .effect-card-header,
            .effect-card-summary {
                min-width: 0;
                overflow: hidden;
            }
            .effect-card-summary {
                white-space: nowrap;
                text-overflow: ellipsis;
            }
            .effect-histogram svg {
                display: block;
                width: 100%;
                height: 104px;
            }
            .hue-gradient-slider .q-slider__track-container {
                height: 32px !important;
                border: 1px solid #777;
                border-radius: 6px;
                background: linear-gradient(to right,
                    #00ffff 0%, #0000ff 16.67%, #ff00ff 33.33%,
                    #ff0000 50%, #ffff00 66.67%, #00ff00 83.33%,
                    #00ffff 100%) !important;
                opacity: 1 !important;
            }
            .hue-gradient-slider .q-slider__track,
            .hue-gradient-slider .q-slider__selection {
                background: transparent !important;
                opacity: 1 !important;
            }
            .hue-gradient-slider .q-slider__thumb {
                filter: drop-shadow(0 0 1px black) drop-shadow(0 0 2px black);
            }
            .hue-gradient-slider .q-slider__thumb-shape {
                fill: white !important;
                stroke: #222 !important;
                stroke-width: 1px;
            }
            .preview-dimension-input {
                width: 72px;
                flex: 0 0 72px;
            }
            .preview-dimension-input .q-field__control,
            .preview-dimension-input .q-field__native,
            .preview-dimension-input .q-field__marginal {
                min-height: 26px !important;
                height: 26px !important;
            }
            .preview-dimension-input .q-field__control {
                padding: 0 5px !important;
            }
            .preview-dimension-input .q-field__native {
                padding: 0 !important;
                color: white !important;
                text-align: right;
                font-size: 12px;
            }
            .layout-column { flex: 1 1 0; min-width: 280px; }
            .simulation-column { flex: 2 1 0; min-width: 460px; }
            .panel-sticky-header { padding-top: 1rem; }
            @media (min-width: 1280px) {
                html,
                body,
                #q-app,
                .nicegui-layout,
                .q-page-container,
                .q-page,
                .nicegui-content {
                    height: 100dvh;
                    max-height: 100dvh;
                    min-height: 0 !important;
                    overflow: hidden;
                }
                .nicegui-content {
                    padding: 0 !important;
                    gap: 0 !important;
                }
                .desktop-app-shell {
                    height: 100dvh;
                    max-height: 100dvh;
                    overflow: hidden;
                }
                .desktop-panel-row {
                    flex: 1 1 0;
                    min-height: 0;
                    overflow: hidden;
                }
                .desktop-scroll-panel {
                    height: 100%;
                    max-height: 100%;
                    min-height: 0;
                    overflow-y: auto;
                    overscroll-behavior: contain;
                    scrollbar-gutter: stable;
                    scrollbar-width: thin;
                }
                .desktop-scroll-panel::-webkit-scrollbar { width: 8px; }
                .desktop-scroll-panel::-webkit-scrollbar-thumb {
                    background: rgba(120, 110, 95, .45);
                    border-radius: 999px;
                }
                .panel-sticky-header {
                    position: sticky;
                    top: 0;
                    z-index: 20;
                    flex: 0 0 auto;
                    padding-bottom: .75rem;
                }
                .settings-card .panel-sticky-header {
                    background: white;
                    border-bottom: 1px solid #ece7de;
                }
                .preview-card .panel-sticky-header {
                    background: #252525;
                    border-bottom: 1px solid #444;
                }
            }
            """
        )

        with ui.column().classes(
            "desktop-app-shell w-full max-w-[1900px] mx-auto p-4 gap-4"
        ):
            ui.label("Screenprint Separator").classes("text-3xl font-bold")
            ui.label(
                "Eine bis fünf Druckfarben · automatische Überdruckzustände"
            ).classes("text-grey-7")

            with ui.row().classes(
                "desktop-panel-row w-full items-start gap-4 flex-wrap xl:flex-nowrap"
            ):
                with ui.column().classes(
                    "settings-card layout-column desktop-scroll-panel "
                    "rounded-xl px-4 pb-4 pt-0 gap-4 w-full"
                ):
                    self._build_input_output_settings()

                with ui.column().classes(
                    "settings-card layout-column desktop-scroll-panel "
                    "rounded-xl px-4 pb-4 pt-0 gap-4 w-full"
                ):
                    self._build_ink_settings()

                with ui.column().classes(
                    "preview-card simulation-column desktop-scroll-panel "
                    "rounded-xl px-4 pb-4 pt-0 gap-3 w-full"
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
        ui.label("Eingabe & Einstellungen").classes(
            "panel-sticky-header text-xl font-semibold w-full"
        )
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
            self.input_preview = ui.interactive_image(
                "",
                on_mouse=self._handle_input_preview_mouse,
                events=["mousemove", "click", "mouseleave"],
            ).classes("input-preview w-full rounded-md")
            with ui.row().classes("w-full items-center gap-2 px-1"):
                ui.icon("check_circle").classes("text-green-7")
                self.upload_status_text = ui.label().classes("grow text-sm")
                ui.label("100 %").classes("text-sm font-medium")
            self.input_size = ui.label().classes("text-xs text-green-9 px-1")
        self.upload_status = upload_status
        self.upload_status.set_visibility(False)

        self._build_effect_settings()
        ui.label("Einstellungen").classes("text-xl font-semibold")

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
                "Glättet die fertig zugeordneten Farbklassen mit einem "
                "Mehrheitsfilter. Ohne Glättung bleiben alle klassifizierten "
                "Pixel unverändert; größere Filter entfernen kleine isolierte "
                "Punkte, können aber Details verlieren.",
            )
            ui.select(
                {
                    1: "Keine Glättung",
                    3: "3 × 3 Pixel",
                    5: "5 × 5 Pixel",
                    7: "7 × 7 Pixel",
                },
                value=self.project.settings.class_smooth_size,
                on_change=lambda event: self._change_setting(
                    "class_smooth_size", event.value, int
                ),
            ).props('aria-label="Klassenglättung"').classes("w-full")

        self._build_halftone_settings()
        self._build_trapping_settings()

    def _build_halftone_settings(self) -> None:
        with ui.expansion("Rasterung", icon="grain").classes("w-full"):
            self.halftone_mode = ui.toggle(
                {"solid": "Vollton", "halftone": "AM-Raster"},
                value=self.project.settings.halftone_mode,
                on_change=self._change_halftone_mode,
            ).props("spread no-caps").classes("w-full")
            with ui.column().classes("w-full gap-3") as halftone_parameters:
                self._build_info_label(
                    "AM-Raster für Verläufe",
                    "Kontinuierliche Flächendeckungen werden in regelmäßig "
                    "angeordnete Rasterpunkte umgesetzt. Beim Export wird direkt "
                    "in der finalen Ausgabeauflösung gerastert.",
                )
                with ui.row().classes("w-full gap-2"):
                    ui.number(
                        "Rasterweite",
                        value=self.project.settings.halftone_frequency_lpi,
                        min=5,
                        max=150,
                        step=1,
                        suffix="lpi",
                        on_change=lambda event: self._change_setting(
                            "halftone_frequency_lpi", event.value
                        ),
                    ).props("debounce=300").classes("grow")
                    ui.select(
                        HALFTONE_SHAPES,
                        label="Punktform",
                        value=self.project.settings.halftone_shape,
                        on_change=lambda event: self._change_setting(
                            "halftone_shape", event.value, str
                        ),
                    ).classes("grow")
                self._build_info_label(
                    "Verlaufsweichheit",
                    "Steuert die distanzgewichtete Interpolation zwischen den "
                    "nächstliegenden Vollton- und Überdruckzuständen. Kleine Werte "
                    "bleiben näher an der bisherigen harten Farbzuordnung.",
                )
                ui.slider(
                    min=0.5,
                    max=15.0,
                    step=0.5,
                    value=self.project.settings.halftone_softness,
                    on_change=lambda event: self._change_setting(
                        "halftone_softness", event.value
                    ),
                ).props("label-always")
                self._build_info_label(
                    "Tonwert / Punktzuwachs",
                    "Werte über 1 vergrößern die Rasterpunkte in den Mitteltönen. "
                    "Minimal- und Maximalpunkt begrenzen nicht stabil druckbare "
                    "Punkte beziehungsweise offene Flächen.",
                )
                ui.slider(
                    min=0.5,
                    max=2.0,
                    step=0.05,
                    value=self.project.settings.halftone_gamma,
                    on_change=lambda event: self._change_setting(
                        "halftone_gamma", event.value
                    ),
                ).props('label-always label prefix="γ "')
                with ui.row().classes("w-full gap-2"):
                    ui.number(
                        "Minimalpunkt",
                        value=self.project.settings.halftone_min_dot * 100,
                        min=0,
                        max=49,
                        step=0.5,
                        suffix="%",
                        on_change=lambda event: self._change_halftone_limit(
                            "halftone_min_dot", event.value
                        ),
                    ).props("debounce=300").classes("grow")
                    ui.number(
                        "Maximalpunkt",
                        value=self.project.settings.halftone_max_dot * 100,
                        min=51,
                        max=100,
                        step=0.5,
                        suffix="%",
                        on_change=lambda event: self._change_halftone_limit(
                            "halftone_max_dot", event.value
                        ),
                    ).props("debounce=300").classes("grow")
                ui.label(
                    "Trapping ist im Rastermodus deaktiviert, weil eine "
                    "Maskenerweiterung wie unkontrollierter Punktzuwachs wirken würde."
                ).classes("text-xs text-grey-6")
            self.halftone_parameters = halftone_parameters
            halftone_parameters.set_visibility(
                self.project.settings.halftone_mode == "halftone"
            )

    def _build_trapping_settings(self) -> None:
        with ui.expansion("Trapping", icon="compare_arrows").classes("w-full"):
            self._build_info_label(
                "Überfüllung der Farbauszüge",
                "Verbreitert jeden Farbauszug in der finalen Ausgabeauflösung um "
                "das gewählte physische Maß. So überlappen benachbarte Farben leicht "
                "und kleine Passerungenauigkeiten erzeugen keine weißen Blitzer. "
                "0 mm deaktiviert das Trapping. Meist reichen 0,05 bis 0,3 mm; "
                "hohe Werte können Zwischenräume schließen und Details verbinden. "
                "Die Auswirkung wird direkt in der Simulationsvorschau angenähert.",
            )
            self.trapping_input = ui.input(
                value=str(self.project.settings.trapping_mm).replace(".", ","),
                on_change=lambda event: self._change_export_setting(
                    "trapping_mm", event.value, float
                ),
            ).props(
                'inputmode=decimal aria-label="Trapping" suffix="mm" debounce=300'
            ).classes("w-full")
            self.trapping_input.set_enabled(
                self.project.settings.halftone_mode != "halftone"
            )

    def _build_effect_settings(self) -> None:
        with ui.row().classes("w-full items-center"):
            ui.label("Effekte").classes("text-xl font-semibold")
            ui.icon("info_outline").classes(
                "text-grey-6 text-base cursor-help shrink-0"
            ).tooltip(
                "Die Effekte werden von oben nach unten auf das Eingabebild "
                "angewandt. Die Reihenfolge kann per Drag & Drop geändert werden."
            )
            ui.element("div").classes("grow")
            add_button = ui.button(icon="add").props("flat dense round")
            add_button.tooltip("Effekt hinzufügen")
            with add_button, ui.menu():
                ui.menu_item(
                    EFFECT_NAMES["saturation_vibrance"],
                    on_click=lambda: self._add_effect("saturation_vibrance"),
                )
                ui.menu_item(
                    EFFECT_NAMES["brightness_contrast"],
                    on_click=lambda: self._add_effect("brightness_contrast"),
                )
                ui.menu_item(
                    EFFECT_NAMES["black_white"],
                    on_click=lambda: self._add_effect("black_white"),
                )
                ui.menu_item(
                    EFFECT_NAMES["levels"],
                    on_click=lambda: self._add_effect("levels"),
                )
                ui.menu_item(
                    EFFECT_NAMES["hue"],
                    on_click=lambda: self._add_effect("hue"),
                )
                ui.menu_item(
                    EFFECT_NAMES["selective_color"],
                    on_click=lambda: self._add_effect("selective_color"),
                )

        with ui.column().classes("w-full gap-3") as effect_cards:
            self.effect_cards = effect_cards
            for effect in list(self.project.effects):
                self._build_effect_controls(effect)
        self.effect_cards.make_sortable(
            handle=".effect-drag-handle",
            on_end=self._drag_effect,
            ghost_class="opacity-40",
        )
        self.effect_empty_label = ui.label("Noch keine Effekte hinzugefügt.").classes(
            "text-sm text-grey-6"
        )
        self._update_effect_summaries()

    def _build_effect_controls(self, effect: ImageEffect) -> None:
        histogram = None
        target_picker = None
        card = ui.expansion().props("dense expand-separator").classes(
            "warm-control effect-card w-full min-w-0 rounded-lg"
        )
        with (
            card.add_slot("header"),
            ui.row().classes("effect-card-header w-full items-center gap-2 no-wrap"),
        ):
            ui.icon("drag_indicator").classes(
                "effect-drag-handle text-grey-6 shrink-0"
            )
            number = ui.label().classes("font-semibold w-5 shrink-0")
            ui.icon(self._effect_icon(effect.kind)).classes("text-grey-7 shrink-0")
            summary = ui.label().classes("effect-card-summary text-sm grow")
            ui.button(
                icon="delete_outline",
                color="negative",
                on_click=lambda effect=effect: self._delete_effect(effect),
            ).props("flat dense round size=sm").classes("shrink-0").tooltip(
                "Effekt löschen"
            )

        with card:
            if effect.kind == "saturation_vibrance":
                self._build_effect_slider(
                    effect,
                    "saturation",
                    "Sättigung",
                    minimum=0.0,
                    maximum=2.0,
                )
                self._build_effect_slider(
                    effect,
                    "vibrance",
                    "Dynamik",
                    minimum=0.0,
                    maximum=2.0,
                )
            elif effect.kind == "brightness_contrast":
                self._build_effect_slider(
                    effect,
                    "brightness",
                    "Helligkeit",
                    minimum=0.25,
                    maximum=2.0,
                )
                self._build_effect_slider(
                    effect,
                    "contrast",
                    "Kontrast",
                    minimum=0.25,
                    maximum=2.0,
                )
            elif effect.kind == "black_white":
                self._build_effect_slider(
                    effect,
                    "amount",
                    "Stärke",
                    minimum=0.0,
                    maximum=1.0,
                )
            elif effect.kind == "levels":
                histogram = ui.html(
                    self._histogram_placeholder(),
                    sanitize=False,
                ).classes("effect-histogram w-full px-2")
                with ui.row().classes(
                    "w-full justify-between text-[10px] text-grey-6 px-2 -mt-2"
                ):
                    ui.label("Schwarz")
                    ui.label("Mitteltöne")
                    ui.label("Weiß")
                self._build_effect_slider(
                    effect,
                    "black_point",
                    "Schwarzpunkt",
                    minimum=0.0,
                    maximum=254.0,
                    step=1.0,
                )
                self._build_effect_slider(
                    effect,
                    "gamma",
                    "Mitteltöne (Gamma)",
                    minimum=0.1,
                    maximum=3.0,
                )
                self._build_effect_slider(
                    effect,
                    "white_point",
                    "Weißpunkt",
                    minimum=1.0,
                    maximum=255.0,
                    step=1.0,
                )
            elif effect.kind == "hue":
                self._build_hue_slider(effect)
            elif effect.kind == "selective_color":
                target_picker = self._build_selective_color_controls(effect)

        self._effect_controls[id(effect)] = {
            "card": card,
            "number": number,
            "summary": summary,
            "histogram": histogram,
            "target_picker": target_picker,
        }

    def _build_effect_slider(
        self,
        effect: ImageEffect,
        parameter: str,
        label: str,
        *,
        minimum: float,
        maximum: float,
        step: float = 0.01,
    ) -> None:
        ui.label(label).classes("text-sm px-2")
        ui.slider(
            min=minimum,
            max=maximum,
            step=step,
            value=effect.value(parameter),
            on_change=lambda event, effect=effect, parameter=parameter: (
                self._change_effect_value(effect, parameter, event.value)
            ),
        ).props("label-always").classes("px-2")

    @staticmethod
    def _effect_icon(kind: str) -> str:
        return {
            "saturation_vibrance": "palette",
            "brightness_contrast": "contrast",
            "black_white": "filter_b_and_w",
            "levels": "linear_scale",
            "hue": "colorize",
            "selective_color": "colorize",
        }.get(kind, "auto_fix_high")

    def _build_selective_color_controls(self, effect: ImageEffect) -> object:
        ui.label("Zu entfernende Farbe").classes("text-sm px-2")
        with ui.row().classes("w-full items-center gap-2 px-2"):
            target_picker = ui.color_input(
                "Zielfarbe",
                value=_rgb_to_hex(self._effect_target_rgb(effect)),
                preview=True,
                on_change=lambda event, effect=effect: (
                    self._change_selective_target(effect, event.value)
                ),
            ).classes("grow")
            target_picker.button.props("icon=palette")
            self._build_eyedropper_button(
                "effect",
                effect,
                "Zielfarbe aus dem Eingabebild aufnehmen",
            )
        self._build_effect_slider(
            effect,
            "tolerance",
            "Toleranz (ΔE)",
            minimum=0.0,
            maximum=60.0,
            step=1.0,
        )
        self._build_effect_slider(
            effect,
            "softness",
            "Weicher Übergang",
            minimum=0.0,
            maximum=60.0,
            step=1.0,
        )
        self._build_effect_slider(
            effect,
            "amount",
            "Farbanteil entfernen",
            minimum=0.0,
            maximum=1.0,
        )
        ui.label(
            "Die Auswahl basiert auf dem wahrnehmungsnahen CIELAB-Farbabstand. "
            "Dadurch kann beispielsweise ein dunkles Grün getrennt von helleren "
            "Grüntönen entsättigt werden."
        ).classes("text-xs text-grey-6 px-2")
        return target_picker

    def _build_hue_slider(self, effect: ImageEffect) -> None:
        slider = ui.slider(
            min=-180.0,
            max=180.0,
            step=1.0,
            value=effect.value("degrees"),
            on_change=lambda event, effect=effect: self._change_hue_value(
                effect, event.value, slider
            ),
        ).props(
            f'label label-value="{effect.value("degrees"):+.0f}°"'
        ).classes("hue-gradient-slider mx-2")
        with ui.row().classes(
            "w-full justify-between text-[10px] text-grey-6 px-2"
        ):
            ui.label("−180°")
            ui.label("0°")
            ui.label("+180°")

    def _change_hue_value(
        self,
        effect: ImageEffect,
        value: float | None,
        slider: object,
    ) -> None:
        if value is None:
            return
        slider.props(f'label label-value="{float(value):+.0f}°"')
        self._change_effect_value(effect, "degrees", value)

    @staticmethod
    def _histogram_placeholder() -> str:
        return (
            '<svg viewBox="0 0 256 104" preserveAspectRatio="none" '
            'role="img" aria-label="Noch kein Histogramm verfügbar">'
            '<rect width="256" height="104" rx="5" fill="#202124"/>'
            '<text x="128" y="55" text-anchor="middle" fill="#9e9e9e" '
            'font-size="11">Bild laden für Histogramm</text></svg>'
        )

    @staticmethod
    def _histogram_svg(image: Image.Image) -> str:
        rgb = np.asarray(image, dtype=np.uint8)
        counts = np.stack(
            [np.bincount(rgb[..., channel].ravel(), minlength=256) for channel in range(3)]
        ).astype(np.float64)
        display_counts = np.sqrt(counts)
        maximum = max(1.0, float(display_counts.max()))

        def polygon(channel: int) -> str:
            heights = display_counts[channel] / maximum * 96.0
            points = ["0,102"]
            points.extend(
                f"{index},{102.0 - height:.2f}"
                for index, height in enumerate(heights)
            )
            points.append("255,102")
            return " ".join(points)

        return (
            '<svg viewBox="0 0 256 104" preserveAspectRatio="none" '
            'role="img" aria-label="RGB-Histogramm">'
            '<rect width="256" height="104" rx="5" fill="#202124"/>'
            '<path d="M64 0V104 M128 0V104 M192 0V104" '
            'stroke="#555" stroke-width="0.5"/>'
            f'<polygon points="{polygon(0)}" fill="#ff5252" fill-opacity=".42"/>'
            f'<polygon points="{polygon(1)}" fill="#69f06f" fill-opacity=".42"/>'
            f'<polygon points="{polygon(2)}" fill="#5c8dff" fill-opacity=".48"/>'
            '<rect x=".5" y=".5" width="255" height="103" rx="5" '
            'fill="none" stroke="#777"/></svg>'
        )

    def _refresh_effect_visualizations(self) -> None:
        if self.project.image is None:
            return
        current = self.project.image.copy()
        current.thumbnail((420, 420), Image.Resampling.LANCZOS)
        try:
            for effect in self.project.effects:
                controls = self._effect_controls.get(id(effect))
                if controls is not None and controls.get("histogram") is not None:
                    controls["histogram"].set_content(self._histogram_svg(current))
                adjusted = apply_effect(current, effect)
                current.close()
                current = adjusted
        finally:
            current.close()

    def _build_ink_settings(self) -> None:
        ui.label("Farben").classes(
            "panel-sticky-header text-xl font-semibold w-full"
        )

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
                {
                    "library": "Papier",
                    "cmyk": "CMYK",
                    "lab": "LAB-Referenzfarbe",
                },
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
            with ui.column().classes("w-full gap-2 px-2 pb-2") as paper_cmyk_group:
                with ui.row().classes("w-full gap-2"):
                    self.paper_cmyk_inputs = []
                    for index, label in enumerate(("C %", "M %", "Y %", "K %")):
                        field = ui.number(
                            label,
                            value=round(self.project.settings.paper_cmyk[index], 1),
                            min=0,
                            max=100,
                            step=0.5,
                            on_change=lambda event, index=index: (
                                self._change_paper_cmyk(index, event.value)
                            ),
                        ).classes("grow min-w-[65px]")
                        self.paper_cmyk_inputs.append(field)
                with ui.row().classes("w-full items-center gap-2"):
                    self.paper_color_picker = ui.color_input(
                        "Farbwähler (Vorschau → CMYK)",
                        value=_rgb_to_hex(self.project.settings.paper),
                        preview=True,
                        on_change=self._change_paper_color_picker,
                    ).classes("grow")
                    self.paper_color_picker.button.props("icon=palette")
                    self._build_eyedropper_button(
                        "paper",
                        None,
                        "Papierfarbe aus dem Eingabebild aufnehmen",
                    )
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
            self.paper_cmyk_group = paper_cmyk_group
            self.paper_lab_group = paper_lab_group
        self._sync_paper_controls()

        with ui.row().classes("w-full items-center"):
            ui.label("Druckfarben").classes("text-lg font-semibold")
            ui.icon("info_outline").classes(
                "text-grey-6 text-base cursor-help shrink-0"
            ).tooltip(
                "Die Reihenfolge der Druckfarben kann per Drag & Drop geändert "
                "werden und bestimmt die Druckreihenfolge."
            )
            ui.element("div").classes("grow")
            ui.button(icon="add", on_click=self._add_ink).props(
                "flat dense round"
            ).tooltip("Druckfarbe hinzufügen")

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
            "warm-control ink-card w-full min-w-0 rounded-lg"
        )
        with (
            card.add_slot("header"),
            ui.row().classes("ink-card-header items-center gap-2 no-wrap"),
        ):
                ui.icon("drag_indicator").classes(
                    "ink-drag-handle text-grey-6 shrink-0"
                )
                number = ui.label().classes("font-semibold w-5 shrink-0")
                summary_swatch = ui.element("div").classes("shrink-0")
                summary = ui.label().classes("ink-card-summary text-sm grow")
                opacity_label = ui.label().classes("text-sm font-medium shrink-0")
                ui.button(
                    icon="delete_outline",
                    color="negative",
                    on_click=lambda ink=ink: self._delete_ink(ink),
                ).props("flat dense round size=sm").classes("shrink-0").tooltip(
                    "Druckfarbe löschen"
                )

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
                with ui.row().classes("w-full items-center gap-2"):
                    color_picker = ui.color_input(
                        "Farbwähler (Vorschau → CMYK)",
                        value=_rgb_to_hex(ink.rgb_preview),
                        preview=True,
                        on_change=lambda event, ink=ink: (
                            self._change_ink_color_picker(ink, event.value)
                        ),
                    ).classes("grow")
                    color_picker.button.props("icon=palette")
                    self._build_eyedropper_button(
                        "ink",
                        ink,
                        "Druckfarbe aus dem Eingabebild aufnehmen",
                    )

            with ui.column().classes("w-full gap-1 px-2") as library_group:
                palette_select = ui.select(
                    self._library_options(),
                    label="Pantone-Farbe",
                    value=ink.library_id or None,
                    with_input=True,
                    on_change=lambda event, ink=ink: self._change_library_color(
                        ink, event.value
                    ),
                ).classes("w-full")
                option_colors = [
                    _rgb_to_hex(entry.rgb)
                    for entry in self.color_library.entries
                    if entry.system != "paper"
                ]
                palette_select.add_slot(
                    "option",
                    f"""
                    <q-item v-bind="props.itemProps">
                      <q-item-section avatar>
                        <div :style="{{
                          width: '28px', height: '20px', borderRadius: '4px',
                          border: '1px solid rgba(0,0,0,.25)',
                          backgroundColor: {option_colors!r}[props.opt.value]
                        }}" />
                      </q-item-section>
                      <q-item-section>
                        <q-item-label>{{{{ props.opt.label }}}}</q-item-label>
                      </q-item-section>
                    </q-item>
                    """,
                )

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
                "cmyk_inputs": cmyk_inputs,
                "color_picker": color_picker,
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
            with ui.column().classes("w-full gap-1 px-2") as halftone_group:
                self._build_info_label(
                    "Rasterwinkel",
                    "Dreht das Raster dieser Druckfarbe. Unterschiedliche Winkel "
                    "reduzieren auffällige Überlagerungsmuster; das Ergebnis sollte "
                    "für das konkrete Gewebe und Motiv geprüft werden.",
                )
                ui.input(
                    "Winkel (°)",
                    value=f"{ink.screen_angle:g}".replace(".", ","),
                    on_change=lambda event, ink=ink: self._change_ink_screen_angle(
                        ink, event.value
                    ),
                ).props(
                    'inputmode=decimal suffix="°" debounce=300'
                ).classes("w-full")
            self._ink_controls[id(ink)]["halftone_group"] = halftone_group
            self._sync_ink_control_visibility(ink)
        self._update_ink_summary(ink)

    def _build_overprint_controls(self) -> None:
        ui.separator()
        ui.label("Überdruckfarben").classes("text-lg font-semibold")
        self._ensure_measured_overprints()
        palette = self._automatic_palette()
        for plate_count in range(2, len(self.project.inks) + 1):
            states = tuple(
                state
                for state in mixed_state_indices(len(self.project.inks))
                if int(palette.masks[state].sum()) == plate_count
            )
            ui.label(f"{plate_count} Druckfarben").classes(
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
        with ui.row().classes("panel-sticky-header w-full items-center"):
            ui.label("Simulation und Ausgabe").classes("text-xl font-semibold grow")
            with ui.element("div").classes(
                "w-5 h-5 shrink-0 flex items-center justify-center"
            ):
                self.preview_spinner = ui.spinner(size="sm")
                self.preview_spinner.set_visibility(False)

        self.status = ui.label("Noch kein Bild geladen").classes("text-grey-4")
        self.preview = ui.interactive_image(
            "",
            on_mouse=self._handle_crop_mouse,
            events=["mousedown", "mousemove", "mouseup", "mouseleave"],
        ).classes(
            "preview-image rounded-lg self-center"
        ).style("width: 100%; max-width: 100%")
        self.preview.set_visibility(False)
        with ui.row().classes("w-full h-8 items-center gap-1 no-wrap"):
            ui.label("Vorschau:").classes("text-xs text-grey-4 shrink-0")
            self.preview_width_input = ui.number(
                value=self.project.settings.preview_width,
                min=50,
                max=5000,
                step=1,
                on_change=lambda event: self._change_preview_dimension(
                    "width", event.value
                ),
            ).props(
                'dense outlined hide-bottom-space aria-label="Vorschau-Breite in Pixeln"'
            ).classes("preview-dimension-input")
            ui.label("×").classes("text-xs text-grey-4 shrink-0")
            self.preview_height_input = ui.number(
                value=self.project.settings.preview_height,
                min=50,
                max=5000,
                step=1,
                on_change=lambda event: self._change_preview_dimension(
                    "height", event.value
                ),
            ).props(
                'dense outlined hide-bottom-space aria-label="Vorschau-Höhe in Pixeln"'
            ).classes("preview-dimension-input")
            ui.label("px").classes("text-xs text-grey-4 shrink-0")
            self.preview_size = ui.label(
                "· Seitenverhältnis bleibt erhalten"
            ).classes("text-xs text-grey-4 grow truncate ml-1")
        self.preview_width_input.set_enabled(self.project.image is not None)
        self.preview_height_input.set_enabled(self.project.image is not None)

        self.palette_row = ui.row().classes("w-full gap-2 flex-wrap")

        with ui.expansion("Farbauszüge", icon="layers").classes("w-full"):
            self.plate_preview_row = ui.row().classes(
                "plate-preview-row w-full gap-3 flex-nowrap"
            )
            self._rebuild_plate_previews()

        self._build_export_settings()

    def _rebuild_plate_previews(self) -> None:
        self._plate_previews = []
        self.plate_preview_row.clear()
        with self.plate_preview_row:
            for index, ink in enumerate(self.project.inks, start=1):
                with ui.column().classes("plate-preview-item"):
                    ui.label(f"{index} · {ink.name}").classes(
                        "plate-preview-label font-medium"
                    )
                    plate = ui.image("").classes("plate-preview-image w-full rounded")
                    plate.set_visibility(False)
                    self._plate_previews.append((ink, plate))

    def _build_export_settings(self) -> None:
        ui.separator().classes("bg-grey-7")
        ui.label("Export").classes("text-lg font-semibold")
        with ui.expansion("Druckformat und Skalierung", icon="straighten").classes(
            "export-settings w-full text-white"
        ):
            with ui.row().classes("w-full"):
                self.print_width_input = ui.number(
                    "Breite (cm)",
                    value=self.project.settings.print_width_cm,
                    min=1,
                    on_change=lambda event: self._change_print_dimension(
                        "width", event.value
                    ),
                ).props("debounce=400").classes("grow")
                self.print_height_input = ui.number(
                    "Höhe (cm)",
                    value=self.project.settings.print_height_cm,
                    min=1,
                    on_change=lambda event: self._change_print_dimension(
                        "height", event.value
                    ),
                ).props("debounce=400").classes("grow")
            with ui.row().classes("w-full"):
                with ui.column().classes("grow gap-0"):
                    self._build_info_label(
                        "Arbeits-DPI",
                        "Auflösung, in der das Bild beim Export tatsächlich getrennt "
                        "und klassifiziert wird. Mehr DPI liefern feinere Details, "
                        "benötigen aber deutlich mehr Zeit und Speicher.",
                    )
                    self.work_dpi_input = ui.number(
                        value=self.project.settings.dpi,
                        min=72,
                        on_change=lambda event: self._change_export_setting(
                            "dpi", event.value, int
                        ),
                    ).props(
                        'aria-label="Arbeits-DPI" debounce=400'
                    ).classes("w-full")
                with ui.column().classes("grow gap-0"):
                    self._build_info_label(
                        "Ausgabe-DPI",
                        "Pixelauflösung der finalen 1-Bit-TIFF-Farbauszüge. Die exakt "
                        "klassifizierten Masken werden kachelweise und kantengeglättet "
                        "auf diese Größe skaliert. Eine höhere Ausgabe-DPI erzeugt "
                        "keine neuen Bilddetails, aber feinere Konturen für Belichter "
                        "und RIP.",
                    )
                    self.output_dpi_input = ui.number(
                        value=self.project.settings.output_dpi,
                        min=72,
                        on_change=lambda event: self._change_export_setting(
                            "output_dpi", event.value, int
                        ),
                    ).props(
                        'aria-label="Ausgabe-DPI" debounce=400'
                    ).classes("w-full")
            with ui.row().classes("w-full items-center gap-2"):
                self.preview_mode_select = ui.select(
                    {
                        "fit": "Format füllen",
                        "pad": "Einpassen mit Rand",
                        "free": "Freier Rahmen",
                    },
                    label="Separationsskalierung",
                    value=self.project.settings.resize_mode,
                    on_change=self._change_resize_mode,
                ).classes("grow")
                self.crop_reset_button = ui.button(
                    "Rahmen zurücksetzen",
                    icon="crop_free",
                    on_click=self._reset_crop_box,
                ).props("flat dense no-caps").classes("shrink-0")
            self.crop_reset_button.set_visibility(
                self.project.settings.resize_mode != "pad"
            )
        self.export_size_label = ui.label().classes("text-sm text-white")
        self._update_export_size_label()

        self.export_button = ui.button(
            "Simulation und Farbauszüge exportieren",
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

    def _change_halftone_mode(
        self, event: events.ValueChangeEventArguments
    ) -> None:
        if event.value not in {"solid", "halftone"}:
            return
        self.project.settings.halftone_mode = event.value
        self.halftone_parameters.set_visibility(event.value == "halftone")
        self.trapping_input.set_enabled(event.value != "halftone")
        for ink in self.project.inks:
            self._sync_ink_control_visibility(ink)
        self._update_export_size_label()
        self._save_session()
        self.schedule_preview()

    def _change_halftone_limit(self, name: str, percent: float | None) -> None:
        if percent is None:
            return
        setattr(
            self.project.settings,
            name,
            float(np.clip(float(percent) / 100.0, 0.0, 1.0)),
        )
        self._save_session()
        self.schedule_preview()

    def _change_export_setting(
        self,
        name: str,
        value: object,
        converter: type,
    ) -> None:
        if value is None:
            return
        normalized = str(value).strip().replace(",", ".")
        if normalized in {"", ".", "+", "-", "+.", "-."}:
            return
        try:
            numeric_value = float(normalized)
            converted = int(numeric_value) if converter is int else converter(
                numeric_value
            )
        except (TypeError, ValueError):
            return
        if name in {"dpi", "output_dpi"}:
            entered_dpi = int(converted)
            converted = max(72, entered_dpi)
            if converted != entered_dpi:
                dpi_input = (
                    self.work_dpi_input
                    if name == "dpi"
                    else self.output_dpi_input
                )
                dpi_input.set_value(converted)
        elif name == "trapping_mm":
            entered_trapping = float(converted)
            converted = min(2.0, max(0.0, entered_trapping))
            if converted != entered_trapping:
                self.trapping_input.set_value(str(converted))
        setattr(self.project.settings, name, converted)
        preview_changed = name == "trapping_mm"
        if (
            name in {"dpi", "output_dpi"}
            and self.project.settings.resize_mode == "fit"
        ):
            self._maximize_fit_crop()
            self._crop_active = False
            preview_changed = True
        if preview_changed:
            self._refresh_trapping_simulation()
            self._update_preview_display(force_source=True)
        self._update_export_size_label()
        if (
            name in {"dpi", "output_dpi"}
            and self.project.settings.resize_mode != "fit"
        ):
            self._refresh_plate_previews()
        self._save_session()

    def _change_resize_mode(self, event: events.ValueChangeEventArguments) -> None:
        if event.value not in {"fit", "pad", "free"}:
            return
        self.project.settings.resize_mode = event.value
        self._crop_active = False
        if event.value == "free":
            self.project.settings.crop_box = normalize_crop_box(
                self.project.settings.crop_box
            )
            self._sync_print_size_to_crop("width")
        elif event.value == "fit":
            self._maximize_fit_crop()
        self.crop_reset_button.set_visibility(event.value != "pad")
        self._update_export_size_label()
        self._refresh_trapping_simulation()
        self._update_preview_display(force_source=True)
        self._save_session()

    def _change_print_dimension(
        self,
        dimension: str,
        value: float | None,
    ) -> None:
        if self._syncing_print_size or value is None:
            return
        value = max(1.0, float(value))
        if dimension == "width":
            self.project.settings.print_width_cm = value
        else:
            self.project.settings.print_height_cm = value

        if self.project.settings.resize_mode == "free":
            self._sync_print_size_to_crop(dimension)
        elif self.project.settings.resize_mode == "fit":
            self._maximize_fit_crop()
            self._crop_active = False
        self._update_export_size_label()
        self._refresh_trapping_simulation()
        self._update_preview_display(force_source=True)
        self._save_session()

    def _sync_print_size_to_crop(self, anchor: str = "width") -> None:
        if self.project.image is None:
            return
        ratio = crop_aspect_ratio(
            self.project.settings.crop_box,
            self.project.image.size,
        )
        if ratio <= 0:
            return
        if anchor == "height":
            self.project.settings.print_width_cm = (
                self.project.settings.print_height_cm * ratio
            )
            if self.project.settings.print_width_cm < 1.0:
                self.project.settings.print_width_cm = 1.0
                self.project.settings.print_height_cm = 1.0 / ratio
        else:
            self.project.settings.print_height_cm = (
                self.project.settings.print_width_cm / ratio
            )
            if self.project.settings.print_height_cm < 1.0:
                self.project.settings.print_height_cm = 1.0
                self.project.settings.print_width_cm = ratio
        self._syncing_print_size = True
        try:
            self.print_width_input.set_value(
                round(self.project.settings.print_width_cm, 3)
            )
            self.print_height_input.set_value(
                round(self.project.settings.print_height_cm, 3)
            )
        finally:
            self._syncing_print_size = False

    def _reset_crop_box(self) -> None:
        if self.project.image is None:
            return
        if self.project.settings.resize_mode == "fit":
            self._maximize_fit_crop()
        else:
            self.project.settings.crop_box = (0.0, 0.0, 1.0, 1.0)
            self._sync_print_size_to_crop("width")
        self._crop_active = False
        self._update_export_size_label()
        self._refresh_trapping_simulation()
        self._update_preview_display(force_source=True)
        self._save_session()

    def _constrain_crop_to_print_aspect(self) -> None:
        if self.project.image is None:
            return
        self.project.settings.crop_box = constrain_crop_box_to_aspect(
            self.project.settings.crop_box,
            self.project.image.size,
            self._export_work_size(),
        )

    def _maximize_fit_crop(self) -> None:
        if self.project.image is None:
            return
        self.project.settings.crop_box = fit_crop_box(
            self.project.image.size,
            self._export_work_size(),
        )

    def _handle_crop_mouse(self, event: events.MouseEventArguments) -> None:
        if (
            self.project.settings.resize_mode not in {"fit", "free"}
            or self.project.preview_image is None
        ):
            return
        width, height = self.project.preview_image.size
        point = (
            float(np.clip(event.image_x / width, 0, 1)),
            float(np.clip(event.image_y / height, 0, 1)),
        )
        if event.type == "mousedown" and event.button == 0:
            action = self._crop_action_at(point)
            if action is None:
                self._crop_active = False
                self._set_crop_cursor(None)
                self._update_crop_overlay()
                return
            self._crop_active = True
            self._update_crop_overlay()
            self._crop_drag_start = point
            self._crop_drag_original = self.project.settings.crop_box
            self._crop_drag_action = action
            self._set_crop_cursor(action)
            return
        if self._crop_drag_start is None:
            if event.type == "mousemove":
                self._set_crop_cursor(self._crop_action_at(point))
            elif event.type == "mouseleave":
                self._crop_active = False
                self._set_crop_cursor(None)
                self._update_crop_overlay()
            return
        if event.type == "mousemove":
            if event.buttons & 1:
                self._apply_crop_drag(point)
            else:
                self._finish_crop_drag(point)
        elif event.type in {"mouseup", "mouseleave"}:
            self._finish_crop_drag(point)
            if event.type == "mouseleave":
                self._crop_active = False
                self._set_crop_cursor(None)
                self._update_crop_overlay()

    def _crop_action_at(self, point: tuple[float, float]) -> str | None:
        if self.project.preview_image is None:
            return None
        left, top, right, bottom = normalize_crop_box(
            self.project.settings.crop_box
        )
        width, height = self.project.preview_image.size
        tolerance_x = max(8.0 / width, 0.012)
        tolerance_y = max(8.0 / height, 0.012)
        x, y = point
        if not self._crop_active:
            return "move" if left < x < right and top < y < bottom else None
        near_left = abs(x - left) <= tolerance_x
        near_right = abs(x - right) <= tolerance_x
        near_top = abs(y - top) <= tolerance_y
        near_bottom = abs(y - bottom) <= tolerance_y
        within_x = left - tolerance_x <= x <= right + tolerance_x
        within_y = top - tolerance_y <= y <= bottom + tolerance_y

        if near_left and near_top:
            return "left_top"
        if near_right and near_top:
            return "right_top"
        if near_left and near_bottom:
            return "left_bottom"
        if near_right and near_bottom:
            return "right_bottom"
        if near_left and within_y:
            return "left"
        if near_right and within_y:
            return "right"
        if near_top and within_x:
            return "top"
        if near_bottom and within_x:
            return "bottom"
        if left < x < right and top < y < bottom:
            return "move"
        return None

    def _set_crop_cursor(self, action: str | None) -> None:
        if action == "move":
            cursor_class = "crop-cursor-move"
        elif action in {"left", "right"}:
            cursor_class = "crop-cursor-ew"
        elif action in {"top", "bottom"}:
            cursor_class = "crop-cursor-ns"
        elif action in {"left_top", "right_bottom"}:
            cursor_class = "crop-cursor-nwse"
        elif action in {"right_top", "left_bottom"}:
            cursor_class = "crop-cursor-nesw"
        else:
            cursor_class = None
        if cursor_class == self._crop_cursor_class:
            return
        if self._crop_cursor_class is not None:
            self.preview.classes(remove=self._crop_cursor_class)
        if cursor_class is not None:
            self.preview.classes(add=cursor_class)
        self._crop_cursor_class = cursor_class

    def _apply_crop_drag(self, point: tuple[float, float]) -> None:
        if (
            self._crop_drag_start is None
            or self._crop_drag_original is None
            or self._crop_drag_action is None
            or self.project.preview_image is None
        ):
            return
        if self._crop_drag_action == "move":
            self.project.settings.crop_box = move_crop_box(
                self._crop_drag_original,
                point[0] - self._crop_drag_start[0],
                point[1] - self._crop_drag_start[1],
            )
        else:
            width, height = self.project.preview_image.size
            if self.project.settings.resize_mode == "fit":
                output_width, output_height = self._export_work_size()
                image_width, image_height = (
                    self.project.image.size
                    if self.project.image is not None
                    else self.project.preview_image.size
                )
                aspect_ratio = (output_width / output_height) / (
                    image_width / image_height
                )
            else:
                aspect_ratio = None
            self.project.settings.crop_box = resize_crop_box(
                self._crop_drag_original,
                self._crop_drag_action,
                point,
                aspect_ratio=aspect_ratio,
                minimum_size=(12.0 / width, 12.0 / height),
            )
        self._update_crop_overlay()
        self._update_preview_size_label()

    def _finish_crop_drag(self, point: tuple[float, float]) -> None:
        if self._crop_drag_action is None:
            return
        self._apply_crop_drag(point)
        changed = self.project.settings.crop_box != self._crop_drag_original
        self._crop_drag_start = None
        self._crop_drag_original = None
        self._crop_drag_action = None
        self._set_crop_cursor(self._crop_action_at(point))
        if not changed:
            return
        if self.project.settings.resize_mode == "free":
            self._sync_print_size_to_crop("width")
        self._update_export_size_label()
        self._refresh_trapping_simulation()
        self._update_preview_display(force_source=True)
        self._save_session()

    def _update_export_size_label(self) -> None:
        settings = self.project.settings
        work_width, work_height = self._export_work_size()
        output_width = round(settings.print_width_cm / 2.54 * settings.output_dpi)
        output_height = round(settings.print_height_cm / 2.54 * settings.output_dpi)
        trapping_pixels = (
            0
            if settings.halftone_mode == "halftone"
            else round(settings.trapping_mm / 25.4 * settings.output_dpi)
        )
        scale_factor = settings.output_dpi / settings.dpi
        scale_text = f"{scale_factor:.1f}".replace(".", ",")
        warnings = []
        if scale_factor > 2.0:
            warnings.append("starke Hochskalierung")
        if settings.halftone_mode != "halftone" and settings.trapping_mm > 0.5:
            warnings.append("Trapping sehr hoch")
        warning_text = f" · ⚠ {' / '.join(warnings)}" if warnings else ""
        output_mode = (
            f"AM-Raster: {settings.halftone_frequency_lpi:g} lpi"
            if settings.halftone_mode == "halftone"
            else f"Trapping: {trapping_pixels} px"
        )
        self.export_size_label.set_text(
            f"Berechnung ({settings.dpi} DPI): {work_width} × {work_height} px · "
            f"TIFF ({settings.output_dpi} DPI): {output_width} × {output_height} px · "
            f"Skalierung: {scale_text}× · {output_mode}"
            f"{warning_text}"
        )

    def _export_work_size(self) -> tuple[int, int]:
        settings = self.project.settings
        return (
            max(1, round(settings.print_width_cm / 2.54 * settings.dpi)),
            max(1, round(settings.print_height_cm / 2.54 * settings.dpi)),
        )

    @staticmethod
    def _effect_target_rgb(effect: ImageEffect) -> tuple[int, int, int]:
        return tuple(
            int(np.clip(round(effect.value(parameter)), 0, 255))
            for parameter in ("target_r", "target_g", "target_b")
        )

    def _change_selective_target(
        self,
        effect: ImageEffect,
        value: str | None,
    ) -> None:
        if self._syncing_effect_controls or effect not in self.project.effects:
            return
        rgb = _hex_to_rgb(value)
        if rgb is None:
            return
        for parameter, component in zip(
            ("target_r", "target_g", "target_b"),
            rgb,
            strict=True,
        ):
            effect.parameters[parameter] = float(component)
        self._update_effect_summaries()
        self._refresh_input_preview()
        self.schedule_preview()

    def _change_effect_value(
        self,
        effect: ImageEffect,
        parameter: str,
        value: float | None,
    ) -> None:
        if value is None or effect not in self.project.effects:
            return
        effect.parameters[parameter] = float(value)
        if effect.kind == "levels":
            black_point = effect.value("black_point")
            white_point = effect.value("white_point")
            if parameter == "black_point" and black_point >= white_point:
                effect.parameters["black_point"] = white_point - 1.0
            elif parameter == "white_point" and white_point <= black_point:
                effect.parameters["white_point"] = black_point + 1.0
        self._update_effect_summaries()
        self._refresh_input_preview()
        self.schedule_preview()

    def _add_effect(self, kind: str) -> None:
        effect = ImageEffect.create(kind)
        self.project.effects.append(effect)
        with self.effect_cards:
            self._build_effect_controls(effect)
        self._update_effect_summaries()
        self._refresh_input_preview()
        self.schedule_preview()

    def _delete_effect(self, effect: ImageEffect) -> None:
        if effect not in self.project.effects:
            return
        self._cancel_eyedropper_for("effect", effect)
        self.project.effects.remove(effect)
        controls = self._effect_controls.pop(id(effect))
        controls["card"].delete()
        self._update_effect_summaries()
        self._refresh_input_preview()
        self.schedule_preview()

    def _drag_effect(self, event: events.SortableEventArguments) -> None:
        if event.old_index == event.new_index:
            return
        effect = self.project.effects.pop(event.old_index)
        self.project.effects.insert(event.new_index, effect)
        self._update_effect_summaries()
        self._refresh_input_preview()
        self.schedule_preview()

    def _update_effect_summaries(self) -> None:
        if not hasattr(self, "effect_empty_label"):
            return
        self.effect_empty_label.set_visibility(not self.project.effects)
        for index, effect in enumerate(self.project.effects, start=1):
            controls = self._effect_controls.get(id(effect))
            if controls is None:
                continue
            controls["number"].set_text(str(index))
            if effect.kind == "saturation_vibrance":
                summary = (
                    f"{EFFECT_NAMES[effect.kind]} · "
                    f"{effect.value('saturation'):.2f} / "
                    f"{effect.value('vibrance'):.2f}"
                )
            elif effect.kind == "brightness_contrast":
                summary = (
                    f"{EFFECT_NAMES[effect.kind]} · "
                    f"{effect.value('brightness'):.2f} / "
                    f"{effect.value('contrast'):.2f}"
                )
            elif effect.kind == "black_white":
                summary = (
                    f"{EFFECT_NAMES[effect.kind]} · "
                    f"{round(effect.value('amount') * 100)} %"
                )
            elif effect.kind == "levels":
                summary = (
                    f"{EFFECT_NAMES[effect.kind]} · "
                    f"{effect.value('black_point'):.0f} / "
                    f"{effect.value('gamma'):.2f} / "
                    f"{effect.value('white_point'):.0f}"
                )
            elif effect.kind == "hue":
                degrees = effect.value("degrees")
                summary = f"{EFFECT_NAMES[effect.kind]} · {degrees:+.0f}°"
            else:
                target = _rgb_to_hex(self._effect_target_rgb(effect)).upper()
                summary = (
                    f"{EFFECT_NAMES[effect.kind]} · {target} · "
                    f"{round(effect.value('amount') * 100)} %"
                )
            controls["summary"].set_text(summary)

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
        try:
            preview.thumbnail((800, 800), Image.Resampling.LANCZOS)
            adjusted = apply_effects(preview, self.project.effects)
            try:
                self._input_preview_size = adjusted.size
                _set_pil_source(self.input_preview, adjusted)
            finally:
                adjusted.close()
        finally:
            preview.close()
        self._refresh_effect_visualizations()

    def _build_eyedropper_button(
        self,
        kind: str,
        target: object | None,
        tooltip: str,
    ) -> None:
        button = ui.button(icon="colorize").props("flat dense round").classes(
            "shrink-0"
        )
        button.tooltip(tooltip)
        with button, ui.menu() as menu, ui.column().classes("w-60 gap-2 p-3"):
            ui.label("Pipette").classes("font-medium")
            radius_label = ui.label(
                self._eyedropper_radius_text()
            ).classes("text-xs text-grey-7")
            radius_slider = ui.slider(
                min=0,
                max=10,
                step=1,
                value=self._eyedropper_radius,
                on_change=lambda event, radius_label=radius_label: (
                    self._change_eyedropper_radius(event.value, radius_label)
                ),
            ).props("label-always")
            ui.label(
                "0 px nimmt exakt einen Pixel auf; größere Radien bilden "
                "den Median der Umgebung."
            ).classes("text-xs text-grey-6")
            ui.button(
                "Farbe aufnehmen",
                icon="colorize",
                on_click=lambda menu=menu, kind=kind, target=target,
                radius_slider=radius_slider: (
                    self._start_eyedropper(
                        menu, kind, target, radius_slider.value
                    )
                ),
            ).props("no-caps").classes("w-full")

    def _eyedropper_radius_text(self, radius: int | None = None) -> str:
        value = self._eyedropper_radius if radius is None else radius
        diameter = value * 2 + 1
        return f"Messradius: {value} px · {diameter} × {diameter} Pixel"

    def _change_eyedropper_radius(
        self,
        value: float | None,
        radius_label: object,
    ) -> None:
        if value is None:
            return
        self._eyedropper_radius = int(np.clip(round(value), 0, 10))
        radius_label.set_text(self._eyedropper_radius_text())

    def _start_eyedropper(
        self,
        menu: object,
        kind: str,
        target: object | None,
        radius: float | None,
    ) -> None:
        if radius is not None:
            self._eyedropper_radius = int(np.clip(round(radius), 0, 10))
        menu.close()
        self._activate_eyedropper(kind, target)

    def _activate_eyedropper(
        self,
        kind: str,
        target: object | None,
    ) -> None:
        if self.project.image is None or self._input_preview_size is None:
            ui.notify("Bitte zuerst ein Eingabebild laden.", type="warning")
            return
        active = self._eyedropper_target
        if active is not None and active[0] == kind and active[1] is target:
            self._cancel_eyedropper()
            ui.notify("Pipette abgebrochen.")
            return
        if active is not None:
            self._cancel_eyedropper()
        self._eyedropper_target = (kind, target)
        self._eyedropper_original_color = self._eyedropper_color(kind, target)
        self._eyedropper_hover_color = None
        self.input_preview.classes(add="eyedropper-active")
        ui.notify(
            "Pipette aktiv: Bewegen zeigt die Farbe, Klicken übernimmt sie."
        )

    def _cancel_eyedropper(self, *, restore: bool = True) -> None:
        if restore and self._eyedropper_original_color is not None:
            active = self._eyedropper_target
            if active is not None:
                self._set_eyedropper_field_color(
                    active[0],
                    active[1],
                    self._eyedropper_original_color,
                )
        self._eyedropper_target = None
        self._eyedropper_original_color = None
        self._eyedropper_hover_color = None
        self.input_preview.classes(remove="eyedropper-active")

    def _cancel_eyedropper_for(self, kind: str, target: object | None) -> None:
        active = self._eyedropper_target
        if active is not None and active[0] == kind and active[1] is target:
            self._cancel_eyedropper()

    def _handle_input_preview_mouse(
        self,
        event: events.MouseEventArguments,
    ) -> None:
        if self._eyedropper_target is None:
            return
        if event.type == "mouseleave":
            if self._eyedropper_original_color is not None:
                kind, target = self._eyedropper_target
                self._set_eyedropper_field_color(
                    kind,
                    target,
                    self._eyedropper_original_color,
                )
                self._eyedropper_hover_color = None
            return
        if event.type not in {"mousemove", "click"}:
            return
        rgb = self._sample_input_color(event.image_x, event.image_y)
        if rgb is None:
            return
        color = _rgb_to_hex(rgb)
        if event.type == "mousemove":
            if color != self._eyedropper_hover_color:
                kind, target = self._eyedropper_target
                self._set_eyedropper_field_color(kind, target, color)
                self._eyedropper_hover_color = color
            return

        kind, target = self._eyedropper_target
        self._cancel_eyedropper(restore=False)

        if kind == "ink" and isinstance(target, Ink) and target in self.project.inks:
            self._change_ink_color_picker(target, color)
        elif kind == "paper":
            self.project.settings.paper_cmyk = rgb_to_cmyk(rgb)
            self.project.settings.paper, self.project.settings.paper_lab = (
                color_from_cmyk(self.project.settings.paper_cmyk)
            )
            self.project.settings.paper_source = "cmyk"
            self._sync_paper_controls()
            self._sync_automatic_mixture_controls()
            self.schedule_preview()
        elif (
            kind == "effect"
            and isinstance(target, ImageEffect)
            and target in self.project.effects
        ):
            self._change_selective_target(target, color)
        else:
            ui.notify("Das Pipetten-Ziel ist nicht mehr verfügbar.", type="warning")
            return
        self._set_eyedropper_field_color(kind, target, color)
        ui.notify(f"Farbe {color.upper()} übernommen.", type="positive")

    def _eyedropper_color(
        self,
        kind: str,
        target: object | None,
    ) -> str | None:
        if kind == "ink" and isinstance(target, Ink):
            return _rgb_to_hex(target.rgb_preview)
        if kind == "paper":
            return _rgb_to_hex(self.project.settings.paper)
        if kind == "effect" and isinstance(target, ImageEffect):
            return _rgb_to_hex(self._effect_target_rgb(target))
        return None

    def _set_eyedropper_field_color(
        self,
        kind: str,
        target: object | None,
        color: str,
    ) -> None:
        if kind == "paper":
            field = self.paper_color_picker
            syncing_name = "_syncing_paper_controls"
        elif kind == "ink" and isinstance(target, Ink):
            controls = self._ink_controls.get(id(target))
            field = controls.get("color_picker") if controls is not None else None
            syncing_name = "_syncing_ink_controls"
        elif kind == "effect" and isinstance(target, ImageEffect):
            controls = self._effect_controls.get(id(target))
            field = controls.get("target_picker") if controls is not None else None
            syncing_name = "_syncing_effect_controls"
        else:
            return
        if field is None:
            return
        setattr(self, syncing_name, True)
        try:
            field.set_value(color)
        finally:
            setattr(self, syncing_name, False)

    def _sample_input_color(
        self,
        image_x: float,
        image_y: float,
    ) -> tuple[int, int, int] | None:
        if self.project.image is None or self._input_preview_size is None:
            return None
        preview_width, preview_height = self._input_preview_size
        source_width, source_height = self.project.image.size
        x = int(
            np.clip(
                image_x / max(1, preview_width) * source_width,
                0,
                source_width - 1,
            )
        )
        y = int(
            np.clip(
                image_y / max(1, preview_height) * source_height,
                0,
                source_height - 1,
            )
        )
        pixels = getattr(self.project, "rgb_array", None)
        if pixels is None:
            pixels = np.asarray(self.project.image, dtype=np.uint8)
        radius = self._eyedropper_radius
        sample = pixels[
            max(0, y - radius) : min(source_height, y + radius + 1),
            max(0, x - radius) : min(source_width, x + radius + 1),
        ]
        median = np.median(sample.reshape(-1, 3), axis=0)
        return tuple(round(component) for component in median)

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
        self.project.settings.paper_cmyk = rgb_to_cmyk(entry.rgb)
        self.project.settings.paper_lab = entry.lab
        self.project.settings.paper_source = "library"
        self._sync_paper_controls()
        self._sync_automatic_mixture_controls()
        self.schedule_preview()

    def _change_paper_source(self, event: events.ValueChangeEventArguments) -> None:
        if self._syncing_paper_controls or event.value not in {
            "library",
            "cmyk",
            "lab",
        }:
            return
        self.project.settings.paper_source = event.value
        if event.value == "library":
            entry = self.color_library.by_id.get(self.project.settings.paper_id)
            if entry is not None and entry.system == "paper":
                self.project.settings.paper = entry.rgb
                self.project.settings.paper_cmyk = rgb_to_cmyk(entry.rgb)
                self.project.settings.paper_lab = entry.lab
        elif event.value == "lab":
            rgb = ColorConverter.lab_to_rgb(
                np.asarray(self.project.settings.paper_lab, dtype=np.float32)
            )
            self.project.settings.paper = tuple(int(value) for value in rgb)
            self.project.settings.paper_cmyk = rgb_to_cmyk(
                self.project.settings.paper
            )
        else:
            self.project.settings.paper, self.project.settings.paper_lab = (
                color_from_cmyk(self.project.settings.paper_cmyk)
            )
        self._sync_paper_controls()
        self._sync_automatic_mixture_controls()
        self.schedule_preview()

    def _change_paper_cmyk(self, index: int, value: float | None) -> None:
        if self._syncing_paper_controls or value is None:
            return
        components = list(self.project.settings.paper_cmyk)
        components[index] = float(np.clip(value, 0.0, 100.0))
        self.project.settings.paper_cmyk = tuple(components)
        self.project.settings.paper, self.project.settings.paper_lab = (
            color_from_cmyk(self.project.settings.paper_cmyk)
        )
        self.project.settings.paper_source = "cmyk"
        self._sync_paper_controls()
        self._sync_automatic_mixture_controls()
        self.schedule_preview()

    def _change_paper_color_picker(self, event: events.ValueChangeEventArguments) -> None:
        if self._syncing_paper_controls:
            return
        rgb = _hex_to_rgb(event.value)
        if rgb is None:
            return
        self.project.settings.paper_cmyk = rgb_to_cmyk(rgb)
        self.project.settings.paper, self.project.settings.paper_lab = (
            color_from_cmyk(self.project.settings.paper_cmyk)
        )
        self.project.settings.paper_source = "cmyk"
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
        self.project.settings.paper_cmyk = rgb_to_cmyk(self.project.settings.paper)
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
                else (
                    "CMYK "
                    + "/".join(
                        f"{value:g}" for value in self.project.settings.paper_cmyk
                    )
                    if source == "cmyk"
                    else "LAB-Referenzfarbe"
                )
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
            self.paper_cmyk_group.set_visibility(source == "cmyk")
            self.paper_lab_group.set_visibility(source == "lab")
            self.paper_select.set_value(self.project.settings.paper_id)
            self.paper_color_picker.set_value(
                _rgb_to_hex(self.project.settings.paper)
            )
            for field, value in zip(
                self.paper_cmyk_inputs,
                self.project.settings.paper_cmyk,
                strict=True,
            ):
                field.set_value(round(float(value), 1))
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
        show_cmyk = ink.color_source == "cmyk"
        controls["cmyk_group"].set_visibility(show_cmyk)
        controls["library_group"].set_visibility(not show_cmyk)
        controls["palette"].set_visibility(not show_cmyk)
        if show_cmyk:
            controls["palette"].run_method("hidePopup")
        halftone_group = controls.get("halftone_group")
        if halftone_group is not None:
            halftone_group.set_visibility(
                self.project.settings.halftone_mode == "halftone"
            )

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
        if self._syncing_ink_controls:
            return
        if value not in {"cmyk", "library"}:
            return
        ink.color_source = value
        if value == "cmyk":
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
        if self._syncing_ink_controls:
            return
        if value is None:
            return
        components = list(ink.cmyk)
        components[index] = float(np.clip(value, 0.0, 100.0))
        ink.cmyk = tuple(components)
        ink.rgb_preview, ink.lab = color_from_cmyk(ink.cmyk)
        ink.color_source = "cmyk"
        ink.pantone = ""
        self._update_ink_swatch(ink)
        controls = self._ink_controls[id(ink)]
        self._syncing_ink_controls = True
        try:
            controls["color_picker"].set_value(_rgb_to_hex(ink.rgb_preview))
        finally:
            self._syncing_ink_controls = False
        self._update_ink_summary(ink)
        self._sync_automatic_mixture_controls()
        self.schedule_preview()

    def _change_ink_color_picker(self, ink: Ink, value: str | None) -> None:
        if self._syncing_ink_controls:
            return
        rgb = _hex_to_rgb(value)
        if rgb is None:
            return
        ink.cmyk = rgb_to_cmyk(rgb)
        ink.rgb_preview, ink.lab = color_from_cmyk(ink.cmyk)
        ink.color_source = "cmyk"
        ink.pantone = ""
        controls = self._ink_controls[id(ink)]
        self._syncing_ink_controls = True
        try:
            for field, component in zip(
                controls["cmyk_inputs"],
                ink.cmyk,
                strict=True,
            ):
                field.set_value(component)
        finally:
            self._syncing_ink_controls = False
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
        self._syncing_ink_controls = True
        try:
            controls["source"].set_value("library")
            controls["palette"].set_value(entry.id)
            controls["color_picker"].set_value(_rgb_to_hex(entry.rgb))
            if entry.cmyk is not None:
                for field, component in zip(
                    controls["cmyk_inputs"],
                    entry.cmyk,
                    strict=True,
                ):
                    field.set_value(component)
        finally:
            self._syncing_ink_controls = False
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
            self.project.settings.paper_cmyk = rgb_to_cmyk(paper_entry.rgb)
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

    def _change_ink_screen_angle(self, ink: Ink, value: str | None) -> None:
        if value is None:
            return
        try:
            angle = float(str(value).strip().replace(",", "."))
        except ValueError:
            return
        ink.screen_angle = round(float(np.clip(angle, 0.0, 179.5)) * 2.0) / 2.0
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
        ink.screen_angle = DEFAULT_SCREEN_ANGLES[len(self.project.inks)]
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
        self._cancel_eyedropper_for("ink", ink)
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
        self._crop_active = False
        self.project.settings.crop_box = (0.0, 0.0, 1.0, 1.0)
        if self.project.settings.resize_mode == "free":
            self._sync_print_size_to_crop("width")
            self._update_export_size_label()
        elif self.project.settings.resize_mode == "fit":
            self._maximize_fit_crop()
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
        self.project.settings.crop_box = normalize_crop_box(
            self.project.settings.crop_box
        )
        if self.project.settings.resize_mode == "free":
            self._sync_print_size_to_crop("width")
            self._update_export_size_label()
        elif self.project.settings.resize_mode == "fit":
            self._constrain_crop_to_print_aspect()
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
                deepcopy(self.project.effects),
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
            if self.project.settings.paper_source == "cmyk":
                components = "/".join(
                    f"{value:g}" for value in self.project.settings.paper_cmyk
                )
                return f"Papier · CMYK {components}"
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

    def _preview_trapping_radius(self) -> float:
        indices = self.project.class_indices
        trapping_mm = self.project.settings.trapping_mm
        if (
            indices is None
            or trapping_mm <= 0
            or self.project.settings.halftone_mode == "halftone"
        ):
            return 0.0

        image_height, image_width = indices.shape
        settings = self.project.settings
        if settings.resize_mode == "pad":
            image_ratio = image_width / image_height
            output_ratio = settings.print_width_cm / settings.print_height_cm
            if image_ratio > output_ratio:
                pixels_per_mm = image_width / (settings.print_width_cm * 10.0)
            else:
                pixels_per_mm = image_height / (settings.print_height_cm * 10.0)
        else:
            left, top, right, bottom = normalize_crop_box(settings.crop_box)
            crop_width = max(1.0, (right - left) * image_width)
            crop_height = max(1.0, (bottom - top) * image_height)
            horizontal_scale = crop_width / (settings.print_width_cm * 10.0)
            vertical_scale = crop_height / (settings.print_height_cm * 10.0)
            pixels_per_mm = (horizontal_scale + vertical_scale) / 2.0

        # Sub-pixel trapping cannot be represented by discrete preview classes.
        # Show at least one preview pixel whenever trapping is enabled.
        return max(1.0, trapping_mm * pixels_per_mm)

    def _refresh_trapping_simulation(self) -> None:
        base = self._base_preview_simulation
        palette = self._preview_palette
        indices = self.project.class_indices
        if base is None or palette is None or indices is None:
            return

        radius = self._preview_trapping_radius()
        if radius <= 0:
            simulation = base
        else:
            simulation = Simulation.render_trapped(indices, palette, radius)

        previous = self.project.preview_image
        self.project.preview_image = simulation
        self.project.preview_array = np.asarray(simulation)
        if previous is not None and previous is not base and previous is not simulation:
            previous.close()

    def _update_preview_display(self, *, force_source: bool = False) -> None:
        simulation = self.project.preview_image
        if simulation is None:
            return
        self._set_crop_cursor(None)
        mode = self.project.settings.resize_mode
        display_size = simulation.size

        if mode == "pad":
            image_width, image_height = simulation.size
            output_width, output_height = self._export_work_size()
            output_ratio = output_width / output_height
            image_ratio = image_width / image_height
            if image_ratio > output_ratio:
                display_size = (image_width, max(1, round(image_width / output_ratio)))
            else:
                display_size = (max(1, round(image_height * output_ratio)), image_height)
            scale = min(1.0, 1800 / max(display_size))
            display_size = (
                max(1, round(display_size[0] * scale)),
                max(1, round(display_size[1] * scale)),
            )
            display = ImageOps.pad(
                simulation,
                display_size,
                method=Image.Resampling.LANCZOS,
                color=self.project.settings.paper,
            )
            try:
                _set_pil_source(self.preview, display)
            finally:
                display.close()
            self.preview.set_content("")
            self.preview.classes(remove="crop-active")
            self.crop_reset_button.set_visibility(False)
        else:
            if force_source:
                _set_pil_source(self.preview, simulation)
            self.preview.classes(add="crop-active")
            self._update_crop_overlay()
            self.crop_reset_button.set_visibility(True)

        width, height = display_size
        aspect_ratio = width / height
        self.preview.style(
            replace=(
                f"width: min(100%, calc(70vh * {aspect_ratio:.8f})); "
                f"max-width: 100%; aspect-ratio: {width} / {height}"
            )
        )
        self._update_preview_size_label()
        self._refresh_plate_previews()

    def _plate_preview_size(self, source_size: tuple[int, int]) -> tuple[int, int]:
        output_width, output_height = self._export_work_size()
        output_ratio = output_width / output_height
        longest_edge = min(900, max(source_size))
        if output_ratio >= 1.0:
            return (longest_edge, max(1, round(longest_edge / output_ratio)))
        return (max(1, round(longest_edge * output_ratio)), longest_edge)

    def _refresh_plate_previews(self) -> None:
        if not self.project.channels:
            return
        for ink, image_element in self._plate_previews:
            channel = self.project.channels.get(ink.name)
            if channel is None:
                image_element.set_visibility(False)
                continue
            active = Image.fromarray(channel, mode="L")
            target_size = self._plate_preview_size(active.size)
            try:
                if self.project.settings.resize_mode == "pad":
                    framed = ImageOps.pad(
                        active,
                        target_size,
                        method=Image.Resampling.NEAREST,
                        color=0,
                    )
                else:
                    framed = active.resize(
                        target_size,
                        resample=Image.Resampling.NEAREST,
                        box=crop_box_pixels(
                            self.project.settings.crop_box,
                            active.size,
                        ),
                    )
                try:
                    plate = ImageOps.invert(framed)
                    try:
                        _set_pil_source(image_element, plate)
                    finally:
                        plate.close()
                finally:
                    framed.close()
            finally:
                active.close()
            image_element.set_visibility(True)

    def _update_crop_overlay(self) -> None:
        simulation = self.project.preview_image
        if simulation is None or self.project.settings.resize_mode == "pad":
            return
        width, height = simulation.size
        box = normalize_crop_box(self.project.settings.crop_box)

        left, top, right, bottom = box
        x0, y0 = left * width, top * height
        x1, y1 = right * width, bottom * height
        selection_width, selection_height = x1 - x0, y1 - y0
        stroke_width = max(1.5, min(width, height) / 350)
        overlay = [
            (
                f'<rect x="0" y="0" width="{width}" height="{y0:.3f}" '
                'fill="rgba(0,0,0,.48)"/>'
            ),
            (
                f'<rect x="0" y="{y1:.3f}" width="{width}" '
                f'height="{max(0.0, height - y1):.3f}" '
                'fill="rgba(0,0,0,.48)"/>'
            ),
            (
                f'<rect x="0" y="{y0:.3f}" width="{x0:.3f}" '
                f'height="{selection_height:.3f}" fill="rgba(0,0,0,.48)"/>'
            ),
            (
                f'<rect x="{x1:.3f}" y="{y0:.3f}" '
                f'width="{max(0.0, width - x1):.3f}" '
                f'height="{selection_height:.3f}" fill="rgba(0,0,0,.48)"/>'
            ),
        ]
        if self._crop_active:
            overlay.append(
                f'<rect x="{x0:.3f}" y="{y0:.3f}" '
                f'width="{selection_width:.3f}" '
                f'height="{selection_height:.3f}" fill="none" stroke="white" '
                f'stroke-width="{stroke_width:.3f}"/>'
            )
            grid_color = "rgba(255,255,255,.58)"
            for fraction in (1 / 3, 2 / 3):
                grid_x = x0 + selection_width * fraction
                grid_y = y0 + selection_height * fraction
                overlay.append(
                    f'<line x1="{grid_x:.3f}" y1="{y0:.3f}" '
                    f'x2="{grid_x:.3f}" y2="{y1:.3f}" stroke="{grid_color}" '
                    f'stroke-width="{stroke_width / 2:.3f}"/>'
                )
                overlay.append(
                    f'<line x1="{x0:.3f}" y1="{grid_y:.3f}" '
                    f'x2="{x1:.3f}" y2="{grid_y:.3f}" stroke="{grid_color}" '
                    f'stroke-width="{stroke_width / 2:.3f}"/>'
                )
            handle_size = max(8.0, min(width, height) / 70.0)
            half_handle = handle_size / 2.0
            handle_points = (
                (x0, y0),
                ((x0 + x1) / 2.0, y0),
                (x1, y0),
                (x0, (y0 + y1) / 2.0),
                (x1, (y0 + y1) / 2.0),
                (x0, y1),
                ((x0 + x1) / 2.0, y1),
                (x1, y1),
            )
            for handle_x, handle_y in handle_points:
                overlay.append(
                    f'<rect x="{handle_x - half_handle:.3f}" '
                    f'y="{handle_y - half_handle:.3f}" '
                    f'width="{handle_size:.3f}" height="{handle_size:.3f}" '
                    'fill="white" stroke="rgba(0,0,0,.75)" '
                    f'stroke-width="{stroke_width / 2:.3f}"/>'
                )
        self.preview.set_content("".join(overlay))

    def _update_preview_size_label(self) -> None:
        simulation = self.project.preview_image
        if simulation is None:
            return
        width, height = simulation.size
        mode = self.project.settings.resize_mode
        if mode == "free":
            left, top, right, bottom = normalize_crop_box(
                self.project.settings.crop_box
            )
            crop_width = max(1, round((right - left) * width))
            crop_height = max(1, round((bottom - top) * height))
            suffix = f" · Ausschnitt: {crop_width} × {crop_height} px"
        elif mode == "fit":
            left, top, right, bottom = normalize_crop_box(
                self.project.settings.crop_box
            )
            crop_width = max(1, round((right - left) * width))
            crop_height = max(1, round((bottom - top) * height))
            suffix = f" · Exportbereich: {crop_width} × {crop_height} px"
        else:
            suffix = " · vollständiges Bild mit Rand"
        self.preview_size.set_text(suffix)

    def _show_result(self, result: SeparationResult) -> None:
        previous_base = self._base_preview_simulation
        previous_preview = self.project.preview_image
        self._base_preview_simulation = result.simulation
        self._preview_palette = result.palette
        self.project.class_indices = result.class_indices
        self.project.channels = result.channels
        self._refresh_trapping_simulation()
        if (
            previous_base is not None
            and previous_base is not previous_preview
            and previous_base is not result.simulation
        ):
            previous_base.close()

        self.preview.set_visibility(True)
        self.status.set_text("Simulation aktuell")
        self._update_preview_display(force_source=True)

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
            archive = await self._export_runner()(
                export_project,
                self.project.image,
                deepcopy(self.project.settings),
                deepcopy(self.project.inks),
                deepcopy(self._active_measured_overprints()),
                deepcopy(self.project.effects),
            )
            if archive is not None:
                save_directly = self._save_export_directly()
                if save_directly:
                    destination = await run.io_bound(save_export_archive, archive)
                    notification.message = f"Export gespeichert: {destination}"
                else:
                    ui.download(archive, filename="screenprint_export.zip")
                    notification.message = "Export ist fertig."
                notification.spinner = False
                notification.type = "positive"
                notification.timeout = 8 if save_directly else 4
                notification.update()
        except (
            OSError,
            ValueError,
            RuntimeError,
            MemoryError,
            run.SubprocessException,
        ) as error:
            notification.dismiss()
            ui.notify(f"Export fehlgeschlagen: {error}", type="negative")
        finally:
            self.export_button.enable()

    @staticmethod
    def _export_runner() -> object:
        if getattr(sys, "frozen", False) and sys.platform == "win32":
            return run.io_bound
        return run.cpu_bound

    @staticmethod
    def _save_export_directly() -> bool:
        return getattr(sys, "frozen", False) and sys.platform == "darwin"
