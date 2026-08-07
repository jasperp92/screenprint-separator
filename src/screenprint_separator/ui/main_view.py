from copy import deepcopy

import numpy as np
from nicegui import events, run, ui
from PIL import Image, ImageOps

from screenprint_separator.export.plate_exporter import export_project
from screenprint_separator.models.ink import Ink
from screenprint_separator.models.project import Project
from screenprint_separator.models.settings import Settings
from screenprint_separator.processing.color_converter import ColorConverter
from screenprint_separator.processing.image_loader import ImageLoader
from screenprint_separator.processing.pipeline import SeparationResult, process_image


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#" + "".join(f"{value:02x}" for value in rgb)


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    color = value.strip().lstrip("#")
    if len(color) != 6:
        raise ValueError("Eine Farbe muss sechs Hex-Zeichen enthalten.")
    return tuple(int(color[index:index + 2], 16) for index in (0, 2, 4))


def _ink_from_rgb(name: str, rgb: tuple[int, int, int]) -> Ink:
    lab = ColorConverter.rgb_to_lab(np.asarray(rgb, dtype=np.uint8))
    return Ink(
        name=name,
        pantone=name,
        lab=tuple(float(value) for value in lab),
        rgb_preview=rgb,
        opacity=0.72,
    )


class MainView:
    def __init__(self) -> None:
        self.project = Project(
            settings=Settings(),
            inks=[
                _ink_from_rgb("Grün", (3, 48, 41)),
                _ink_from_rgb("Gelb", (255, 242, 155)),
                _ink_from_rgb("Blau", (198, 215, 255)),
            ],
        )
        self._preview_busy = False
        self._preview_dirty = False
        self._ink_titles: dict[int, object] = {}
        self._plate_previews: list[tuple[Ink, object]] = []

        ui.add_css(
            """
            body { background: #f4f1ea; }
            .settings-card { background: white; border: 1px solid #ded8cc; }
            .preview-card { background: #252525; color: white; min-height: 420px; }
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
            ui.color_input(
                "Papierfarbe",
                value=_rgb_to_hex(self.project.settings.paper),
                preview=True,
                on_change=self._change_paper,
            ).classes("w-full")
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
            "LAB steuert die Zuordnung; RGB steuert die Bildschirmdarstellung."
        ).classes("text-xs text-grey-7")

        for ink in list(self.project.inks):
            self._build_ink_controls(ink)

        self.order_label = ui.label().classes("text-sm font-medium")
        self._update_order_label()

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

            ui.input(
                "Pantone / Bezeichnung",
                value=ink.pantone,
                on_change=lambda event, ink=ink: self._change_ink_name(
                    ink, event.value
                ),
            ).classes("w-full")
            ui.color_input(
                "Vorschaufarbe",
                value=_rgb_to_hex(ink.rgb_preview),
                preview=True,
                on_change=lambda event, ink=ink: self._change_ink_rgb(
                    ink, event.value
                ),
            ).classes("w-full")

            with ui.row().classes("w-full gap-2"):
                for index, label in enumerate(("L*", "a*", "b*")):
                    ui.number(
                        label,
                        value=round(ink.lab[index], 2),
                        step=0.1,
                        on_change=lambda event, ink=ink, index=index: (
                            self._change_ink_lab(ink, index, event.value)
                        ),
                    ).classes("grow min-w-[80px]")

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
            "w-full max-h-[70vh] object-contain rounded-lg"
        )
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

    def _change_paper(self, event: events.ValueChangeEventArguments) -> None:
        if not event.value:
            return
        try:
            self.project.settings.paper = _hex_to_rgb(event.value)
        except ValueError as error:
            ui.notify(str(error), type="negative")
            return
        self.schedule_preview()

    def _change_ink_name(self, ink: Ink, value: str | None) -> None:
        if not value:
            return
        ink.pantone = value
        ink.name = value
        self._ink_titles[id(ink)].set_text(value)
        self._update_order_label()
        self.schedule_preview()

    def _change_ink_rgb(self, ink: Ink, value: str | None) -> None:
        if not value:
            return
        try:
            ink.rgb_preview = _hex_to_rgb(value)
        except ValueError as error:
            ui.notify(str(error), type="negative")
            return
        self.schedule_preview()

    def _change_ink_lab(self, ink: Ink, index: int, value: float | None) -> None:
        if value is None:
            return
        values = list(ink.lab)
        values[index] = float(value)
        ink.lab = tuple(values)
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
