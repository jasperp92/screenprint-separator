from nicegui import ui

from screenprint_separator.ui.main_view import MainView


def main() -> None:
    MainView()

    ui.run(
        title="Screenprint Separator",
        reload=True,
        uvicorn_reload_dirs="src",
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
