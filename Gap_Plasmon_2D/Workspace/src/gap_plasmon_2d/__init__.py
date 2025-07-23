from gap_plasmon_2d.utils.logging_setup import setup_subpackage_loggers
setup_subpackage_loggers()

# Ensure FloatText widgets do not fail when cleared
try:
    import ipywidgets as widgets
    from .utils.widgets import SafeFloatText
    widgets.FloatText = SafeFloatText
except Exception:
    pass
