import ipywidgets as widgets
import traitlets

class SafeFloatText(widgets.FloatText):
    """FloatText that treats ``None`` as 0.0 to avoid trait errors."""
    value = traitlets.CFloat(default_value=0.0, allow_none=True).tag(sync=True)

    @traitlets.validate('value')
    def _valid_value(self, proposal):
        return 0.0 if proposal['value'] is None else proposal['value']
