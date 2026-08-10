from copy import deepcopy
from pathlib import Path

import numpy as np
from nicegui import events, run, ui
from PIL import Image, ImageOps

from screenprint_separator.export.plate_exporter import export_project
from screenprint_separator.models.ink import Ink
from screenprint_separator.models.project import Project
from screenprint_separator.models.settings import Settings
from screenprint_separator.processing.color_library import (
    ColorEntry,
    ColorLibrary,
    color_from_cmyk,
)
from screenprint_separator.processing.image_loader import ImageLoader
from screenprint_separator.processing.pipeline import SeparationResult, process_image


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#" + "".join(f"{value:02x}" for value in rgb)


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    color = value.strip().lstrip("#")
    if len(color) != 6:
        raise ValueError("Eine Farbe muss sechs Hex-Zeichen enthalten.")
    return tuple(int(color[index:index + 2], 16) for index in (0, 2, 4))


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
        self.project = Project(
            settings=settings,
            inks=default_inks,
        )
        self._preview_busy = False
        self._preview_dirty = False
        self._ink_titles: dict[int, object] = {}
        self._ink_controls: dict[int, dict[str, object]] = {}
        self._plate_previews: list[tuple[Ink, object]] = []

        ui.add_css(
            """
            body { background: #f4f1ea; }
            .settings-card { background: white; border: 1px solid #ded8cc; }
            .preview-card { background: #252525; color: white; min-height: 420px; }
            .preview-image > img { object-fit: contain !important; }
            """
        )

        with ui.column().classes("w-full max-w-[1500px] mx-auto p-4 gap-4"):
            ui.label("Screenprint Separator").classes("text-3xl font-bold")
            ui.label(
                "Drei Druckfarben · acht Überdruckzustände · LAB/ΔE00"
            ).classes("text-grey-7")

            with ui.row().classes("w-full items-start gap-4 flex-wrap lg:flex-nowrap"):
                with ui.column().classes(
                    "settings-card rounded-xl p-4 gap-4 w-full lg:w-[430px] shrink-0"
                ):
                    self._build_settings()

                with ui.column().classes(
                    "preview-card rounded-xl p-4 gap-3 w-full grow min-w-0"
                ):
                    self._build_preview()

        self.preview_timer = ui.timer(
            0.3,
            self._run_scheduled_preview,
            active=False,
            immediate=False,
        )

    def _build_settings(self) -> None:
        ui.label("Einstellungen").classes("text-xl font-semibold")
        self.upload = ui.upload(
            label="Bild laden",
            on_upload=self.load_image,
            auto_upload=True,
        ).props("accept=.png,.jpg,.jpeg,.tif,.tiff flat bordered").classes("w-full")

        with ui.expansion("Druckformat", icon="straighten").classes("w-full"):
            with ui.row().classes("w-full"):
                ui.number(
                    "Breite (cm)",
                    value=self.project.settings.print_width_cm,
                    min=1,
                    on_change=lambda event: self._change_setting(
                        "print_width_cm", event.value, float, False
                    ),
                ).classes("grow")
                ui.number(
                    "Höhe (cm)",
                    value=self.project.settings.print_height_cm,
                    min=1,
                    on_change=lambda event: self._change_setting(
                        "print_height_cm", event.value, float, False
                    ),
                ).classes("grow")
            with ui.row().classes("w-full"):
                ui.number(
                    "Arbeits-DPI",
                    value=self.project.settings.dpi,
                    min=72,
                    on_change=lambda event: self._change_setting(
                        "dpi", event.value, int, False
                    ),
                ).classes("grow")
                ui.number(
                    "Ausgabe-DPI",
                    value=self.project.settings.output_dpi,
                    min=72,
                    on_change=lambda event: self._change_setting(
                        "output_dpi", event.value, int, False
                    ),
                ).classes("grow")
            ui.select(
                {"fit": "Format füllen", "pad": "Einpassen mit Rand"},
                label="Skalierung",
                value=self.project.settings.resize_mode,
                on_change=lambda event: self._change_setting(
                    "resize_mode", event.value, str, False
                ),
            ).classes("w-full")

        with ui.expansion("Papier und Oberfläche", icon="texture").classes("w-full"):
            self.paper_select = ui.select(
                self._paper_options(),
                label="Bedruckstoff / Papierfarbe",
                value=self.project.settings.paper_id,
                on_change=self._change_paper_selection,
            ).classes("w-full")
            self.custom_paper_input = ui.color_input(
                "Eigene Papierfarbe",
                value=_rgb_to_hex(self.project.settings.paper),
                preview=True,
                on_change=self._change_custom_paper,
            ).classes("w-full")
            self.custom_paper_input.set_visibility(
                self.project.settings.paper_id == "custom"
            )
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
            ui.label("Glättungsradius").classes("text-sm")
            ui.slider(
                min=0.0,
                max=5.0,
                step=0.25,
                value=self.project.settings.texture_blur_radius,
                on_change=lambda event: self._change_setting(
                    "texture_blur_radius", event.value, float
                ),
            ).props("label-always")
            ui.select(
                [1, 3, 5, 7],
                label="Klassenglättung",
                value=self.project.settings.class_smooth_size,
                on_change=lambda event: self._change_setting(
                    "class_smooth_size", event.value, int
                ),
            ).classes("w-full")

        ui.separator()
        ui.label("Druckfarben").classes("text-lg font-semibold")
        ui.label(
            "CMYK direkt eingeben oder eine Farbe aus color_library.json wählen."
        ).classes("text-xs text-grey-7")

        for ink in list(self.project.inks):
            self._build_ink_controls(ink)

        self.order_label = ui.label().classes("text-sm font-medium")
        self._update_order_label()
        ui.button(
            "Farbbibliothek neu laden",
            icon="refresh",
            on_click=self._reload_color_library,
        ).props("flat dense")

        ui.separator()
        self.export_button = ui.button(
            "Simulation und Platten exportieren",
            icon="download",
            on_click=self.export,
        ).classes("w-full")
        self.export_button.disable()

    def _build_ink_controls(self, ink: Ink) -> None:
        with ui.card().classes("w-full p-3 gap-2"):
            with ui.row().classes("w-full items-center"):
                title = ui.label(ink.name).classes("font-semibold grow")
                self._ink_titles[id(ink)] = title
                ui.button(
                    icon="arrow_upward",
                    on_click=lambda ink=ink: self._move_ink(ink, -1),
                ).props("flat dense round")
                ui.button(
                    icon="arrow_downward",
                    on_click=lambda ink=ink: self._move_ink(ink, 1),
                ).props("flat dense round")

            source_select = ui.select(
                {
                    "cmyk": "CMYK eingeben",
                    "library": "Pantone / Farbbibliothek",
                },
                label="Farbquelle",
                value=ink.color_source,
                on_change=lambda event, ink=ink: self._change_ink_source(
                    ink, event.value
                ),
            ).classes("w-full")

            swatch = ui.element("div").style(
                self._swatch_style(ink.rgb_preview)
            ).classes("w-full")

            with ui.column().classes("w-full gap-2") as cmyk_group:
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

            with ui.column().classes("w-full gap-1") as library_group:
                palette_select = ui.select(
                    self._library_options(),
                    label="Pantone / gespeicherte Farbe",
                    value=ink.library_id or None,
                    with_input=True,
                    on_change=lambda event, ink=ink: self._change_library_color(
                        ink, event.value
                    ),
                ).classes("w-full")
                ui.label(
                    "Pantone-LAB-Werte können aus einer lizenzierten Quelle "
                    "in color_library.json eingetragen werden."
                ).classes("text-[11px] text-grey-7")

            self._ink_controls[id(ink)] = {
                "source": source_select,
                "swatch": swatch,
                "cmyk_group": cmyk_group,
                "library_group": library_group,
                "name": name_input,
                "cmyk_inputs": cmyk_inputs,
                "palette": palette_select,
            }
            self._sync_ink_control_visibility(ink)

            ui.label("Überdruckstärke").classes("text-xs")
            ui.slider(
                min=0.0,
                max=1.0,
                step=0.01,
                value=ink.opacity,
                on_change=lambda event, ink=ink: self._change_ink_value(
                    ink, "opacity", event.value
                ),
            ).props("label-always")
            ui.label("Klassifikations-Bias (ΔE)").classes("text-xs")
            ui.slider(
                min=-10.0,
                max=20.0,
                step=0.5,
                value=ink.bias,
                on_change=lambda event, ink=ink: self._change_ink_value(
                    ink, "bias", event.value
                ),
            ).props("label-always")

    def _build_preview(self) -> None:
        with ui.row().classes("w-full items-center"):
            ui.label("Simulation").classes("text-xl font-semibold grow")
            self.preview_spinner = ui.spinner(size="lg")
            self.preview_spinner.set_visibility(False)

        self.status = ui.label("Noch kein Bild geladen").classes("text-grey-4")
        self.preview = ui.interactive_image("").classes(
            "preview-image rounded-lg self-center"
        ).style("width: 100%; max-width: 100%")
        self.preview.set_visibility(False)

        self.palette_row = ui.row().classes("w-full gap-2 flex-wrap")

        with (
            ui.expansion("Plattenvorschau", icon="layers").classes("w-full"),
            ui.row().classes("w-full gap-3 flex-wrap"),
        ):
                for ink in self.project.inks:
                    with ui.column().classes("grow min-w-[180px]"):
                        ui.label(ink.name).classes("font-medium")
                        plate = ui.image("").classes("w-full rounded")
                        plate.set_visibility(False)
                        self._plate_previews.append((ink, plate))

        ui.separator().classes("bg-grey-7")
        self.filename = ui.label("Datei: –")
        self.size = ui.label("Originalgröße: –")
        self.preview_size = ui.label("Vorschaugröße: –")

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
        if refresh:
            self.schedule_preview()

    def _change_paper_selection(
        self,
        event: events.ValueChangeEventArguments,
    ) -> None:
        identifier = event.value
        if identifier == "custom":
            self.project.settings.paper_id = "custom"
            self.custom_paper_input.set_visibility(True)
            self.schedule_preview()
            return

        entry = self.color_library.by_id.get(identifier)
        if entry is None or entry.system != "paper":
            ui.notify("Unbekannte Papierfarbe", type="negative")
            return
        self.project.settings.paper_id = entry.id
        self.project.settings.paper = entry.rgb
        self.custom_paper_input.set_visibility(False)
        self.schedule_preview()

    def _change_custom_paper(self, event: events.ValueChangeEventArguments) -> None:
        if not event.value:
            return
        try:
            self.project.settings.paper = _hex_to_rgb(event.value)
        except ValueError as error:
            ui.notify(str(error), type="negative")
            return
        self.project.settings.paper_id = "custom"
        self.schedule_preview()

    def _change_ink_name(self, ink: Ink, value: str | None) -> None:
        if not value:
            return
        ink.name = value
        self._ink_titles[id(ink)].set_text(value)
        self._update_order_label()
        self.schedule_preview()

    @staticmethod
    def _swatch_style(rgb: tuple[int, int, int]) -> str:
        return (
            f"background:{_rgb_to_hex(rgb)};height:36px;border-radius:6px;"
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
        options["custom"] = "Eigene Papierfarbe …"
        return options

    def _sync_ink_control_visibility(self, ink: Ink) -> None:
        controls = self._ink_controls[id(ink)]
        controls["cmyk_group"].set_visibility(ink.color_source == "cmyk")
        controls["library_group"].set_visibility(ink.color_source == "library")

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
        self.schedule_preview()

    def _change_library_color(self, ink: Ink, identifier: str | None) -> None:
        if not identifier:
            return
        entry = self.color_library.by_id.get(identifier)
        if entry is None:
            ui.notify(f"Unbekannte Bibliotheksfarbe: {identifier}", type="negative")
            return
        self._apply_library_entry(ink, entry)
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
        self._ink_titles[id(ink)].set_text(ink.name)
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
        self._sync_ink_control_visibility(ink)
        self._update_order_label()

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

        paper_options = self._paper_options()
        paper_id = self.project.settings.paper_id
        if paper_id != "custom":
            paper_entry = self.color_library.by_id.get(paper_id)
            if paper_entry is None or paper_entry.system != "paper":
                paper_id = "custom"
                self.project.settings.paper_id = "custom"
            else:
                self.project.settings.paper = paper_entry.rgb
        self.paper_select.set_options(paper_options, value=paper_id)
        self.custom_paper_input.set_visibility(paper_id == "custom")

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
        self.schedule_preview()

    def _change_ink_value(self, ink: Ink, name: str, value: float | None) -> None:
        if value is None:
            return
        setattr(ink, name, float(value))
        self.schedule_preview()

    def _move_ink(self, ink: Ink, direction: int) -> None:
        index = self.project.inks.index(ink)
        target = index + direction
        if not 0 <= target < len(self.project.inks):
            return
        self.project.inks[index], self.project.inks[target] = (
            self.project.inks[target],
            self.project.inks[index],
        )
        self._update_order_label()
        self.schedule_preview()

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
        self.filename.set_text(f"Datei: {event.file.name}")
        self.size.set_text(f"Originalgröße: {width} × {height} px")
        self.export_button.enable()
        await self.refresh_preview()

    def schedule_preview(self) -> None:
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
        self.status.set_text("LAB-Klassifikation wird berechnet …")

        try:
            result = await run.io_bound(
                process_image,
                self.project.image.copy(),
                deepcopy(self.project.settings),
                deepcopy(self.project.inks),
                deepcopy(self.project.measured_overprints),
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

        self.palette_row.clear()
        with self.palette_row:
            for name, rgb, lab in zip(
                result.palette.names,
                result.palette.rgb,
                result.palette.lab,
                strict=True,
            ):
                color = _rgb_to_hex(tuple(int(value) for value in rgb))
                with ui.column().classes("gap-0 items-center"):
                    ui.element("div").style(
                        f"background:{color};width:48px;height:32px;border-radius:6px;"
                        "border:1px solid rgba(255,255,255,.35)"
                    ).tooltip(
                        f"{name}\nLAB {lab[0]:.1f}, {lab[1]:.1f}, {lab[2]:.1f}"
                    )
                    ui.label(name).classes("text-[10px] max-w-[90px] truncate")

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
                deepcopy(self.project.measured_overprints),
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
