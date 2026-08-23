import multiprocessing
import sys

# PyInstaller workers re-enter the bundled executable. Divert multiprocessing
# worker invocations before importing NiceGUI or the application UI.
if __name__ == "__main__":
    multiprocessing.freeze_support()

from nicegui import native, ui

from screenprint_separator.ui.main_view import MainView


def main() -> None:
    is_packaged = getattr(sys, "frozen", False)
    use_native_window = is_packaged and sys.platform == "darwin"
    port = native.find_open_port() if is_packaged else 8080

    ui.run(
        root=MainView,
        title="Screenprint Separator",
        port=port,
        reload=not is_packaged,
        uvicorn_reload_dirs="src",
        native=use_native_window,
        window_size=(1400, 900) if use_native_window else None,
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
