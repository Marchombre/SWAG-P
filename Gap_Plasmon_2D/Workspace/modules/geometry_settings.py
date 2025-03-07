# geometry_settings.py
import ipywidgets as widgets
from IPython.display import display, clear_output

# Default parameters for wave, geometry, and limits
wave = {"wavelength": 450, "angle": 0, "polarization": 1}

geometry_config = {
    "thick_super": 200,
    "width_reso": 30,
    "thick_reso": 30,
    "thick_gap": 3,
    "thick_func": 1,
    "thick_mol": 2,
    "thick_metalliclayer": 10,
    "thick_sub": 200,
    "thick_accroche": 1,
    "period": 100.2153
}

geometry_limits = {
    "thick_super": (0, 300),
    "width_reso": (0, 100),
    "thick_reso": (0, 100),
    "thick_gap": (0, 50),
    "thick_func": (0, 20),
    "thick_mol": (0, 20),
    "thick_metalliclayer": (0, 50),
    "thick_sub": (0, 300),
    "thick_accroche": (0, 20),
    "period": (50, 300)
}

def create_geometry_widget():
    """
    Creates and returns a widget (VBox) containing geometry sliders and a button
    to validate the configuration. The display is done only once.
    
    Returns:
        - widget: VBox containing the sliders, the button, and the output area.
    """
    geometry_sliders = {}
    slider_widgets = []
    
    # Create a slider for each parameter
    for key, default in geometry_config.items():
        min_val, max_val = geometry_limits.get(key, (0, 200))
        slider = widgets.FloatSlider(
            value=default, min=min_val, max=max_val, step=1,
            description=key,
            continuous_update=False,
            style={'description_width': 'initial'}
        )
        geometry_sliders[key] = slider
        slider_widgets.append(slider)
    
    # Create a validation button and an output widget to display the updated values
    button_update_geo = widgets.Button(description="Validate Geometry Configuration")
    output_geo = widgets.Output()
    
    def update_geometry(b):
        # Update geometry_config with the current slider values
        for key, slider in geometry_sliders.items():
            geometry_config[key] = slider.value
        with output_geo:
            clear_output()
            print("New geometry configuration:")
            for key, value in geometry_config.items():
                print(f"  {key}: {value}")
        # Optionally: display a global message in __main__
        import __main__
        __main__.GEOMETRY_CONFIG = geometry_config  # so that it is globally accessible
        print("GEOMETRY_CONFIG has been updated in the global (__main__) space.")

    button_update_geo.on_click(update_geometry)
    
    # Create and return a VBox containing the sliders, the button, and the output area
    widget = widgets.VBox(slider_widgets + [button_update_geo, output_geo])
    return widget
