import sys

from nicegui import ui

from screenprint_separator.ui.main_view import MainView


def main() -> None:
    MainView()
    is_packaged = getattr(sys, "frozen", False)
    use_native_window = is_packaged and sys.platform == "darwin"

    ui.run(
        title="Screenprint Separator",
        reload=not is_packaged,
        uvicorn_reload_dirs="src",
        native=use_native_window,
        window_size=(1400, 900) if use_native_window else None,
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
