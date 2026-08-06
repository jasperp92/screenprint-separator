from nicegui import ui

from screenprint_separator.ui.main_view import MainView

MainView()

ui.run(
    title="Screenprint Separator",
    reload=False,
)