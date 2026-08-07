from nicegui import ui

from screenprint_separator.ui.main_view import MainView


def main() -> None:
    MainView()

    ui.run(
        title="Screenprint Separator",
        reload=False,
    )


if __name__ == "__main__":
    main()
