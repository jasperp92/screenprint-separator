from nicegui import events, ui
from pathlib import Path


from screenprint_separator.models.project import Project
from screenprint_separator.models.settings import Settings
from screenprint_separator.processing.image_loader import ImageLoader


class MainView:

    def __init__(self) -> None:

        self.project = Project(
            settings=Settings(),
        )

        ui.label(
            "Screenprint Separator"
        ).classes("text-2xl font-medium")

        with ui.row():

            with ui.column():

                ui.label("Einstellungen")

                ui.separator()

                self.upload = ui.upload(
                    on_upload=self.load_image,
                    auto_upload=True,
                ).props(
                    "accept=.png,.jpg,.jpeg,.tif,.tiff"
                )

                ui.number(
                    label="DPI",
                    value=self.project.settings.dpi,
                )

                ui.slider(
                    min=0,
                    max=255,
                    value=self.project.settings.threshold,
                )

                ui.separator()

                ui.button("Export")

            with ui.column():

                ui.label("Vorschau")

                self.preview_label = ui.label(
                    "Noch kein Bild geladen"
                )

    def load_image(
        self,
        event: events.UploadEventArguments,
    ) -> None:

        path = Path(event.file._path)

        with open(path, "rb") as f:
            image_bytes = f.read()

        self.project.image, self.project.rgb_array = (
            ImageLoader.load(image_bytes)
        )

        self.preview_label.set_text(
            f"{event.file.name} geladen"
        )

        print(f"Datei: {event.file.name}")
        print(f"Größe: {self.project.image.size}")
        print(self.project.rgb_array.shape)