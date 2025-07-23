"""UI package initialization with custom CSS for widgets."""
from __future__ import annotations

from pathlib import Path

from IPython.display import HTML, display


def _load_custom_css() -> None:
    css_path = Path(__file__).with_name("widgets.css")
    if css_path.exists():
        with css_path.open("r", encoding="utf-8") as f:
            css = f.read()
        display(HTML(f"<style>{css}</style>"))


_load_custom_css()
