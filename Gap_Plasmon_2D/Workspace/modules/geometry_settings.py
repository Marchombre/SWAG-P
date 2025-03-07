# geometry_settings.py
import ipywidgets as widgets
from IPython.display import display, clear_output

# Paramètres par défaut pour wave, géométrie et limites
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
    Crée et retourne un widget (VBox) contenant les sliders de géométrie et un bouton
    permettant de valider la configuration. L'affichage se fait une seule fois.
    
    Retourne:
        - widget: VBox contenant les sliders, le bouton et la zone de sortie.
    """
    geometry_sliders = {}
    slider_widgets = []
    
    # Créer un slider pour chaque paramètre
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
    
    # Créer le bouton de validation et un widget de sortie pour afficher les valeurs mises à jour
    button_update_geo = widgets.Button(description="Valider la configuration géométrique")
    output_geo = widgets.Output()
    
    def update_geometry(b):
        # Mettre à jour geometry_config avec les valeurs actuelles
        for key, slider in geometry_sliders.items():
            geometry_config[key] = slider.value
        with output_geo:
            clear_output()
            print("Nouvelle configuration géométrique:")
            for key, value in geometry_config.items():
                print(f"  {key}: {value}")
        # Optionnel : afficher un message global dans __main__
        import __main__
        __main__.GEOMETRY_CONFIG = geometry_config  # si vous souhaitez que ce soit accessible globalement
        print("GEOMETRY_CONFIG a été mis à jour dans l'espace global (__main__).")

    
    button_update_geo.on_click(update_geometry)
    
    # Créer et retourner une VBox contenant les sliders, le bouton et la zone de sortie
    widget = widgets.VBox(slider_widgets + [button_update_geo, output_geo])
    return widget
