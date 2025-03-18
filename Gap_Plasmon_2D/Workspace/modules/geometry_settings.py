import ipywidgets as widgets
from IPython.display import clear_output, display
import os
import json

# Paramètres par défaut pour la géométrie
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
    "thick_gap": (0, 30),
    "thick_func": (0, 20),
    "thick_mol": (0, 20),
    "thick_metalliclayer": (0, 50),
    "thick_sub": (0, 300),
    "thick_accroche": (0, 20),
    "period": (50, 300)
}

# Fichier de sauvegarde (dans le dossier Summary_Simulation)
def get_geometry_save_path():
    module_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    workspace_dir = os.path.dirname(module_dir)
    notebooks_dir = os.path.join(workspace_dir, "notebooks")
    summary_dir = os.path.join(notebooks_dir, "Summary_Simulation")
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    return os.path.join(summary_dir, "geometry_configurations.json")

# Liste globale pour stocker les configurations de géométrie enregistrées
GEOMETRY_CONFIGS = []

def load_geometry_configs():
    """Charge les configurations sauvegardées depuis le fichier JSON (si il existe)."""
    global GEOMETRY_CONFIGS
    save_path = get_geometry_save_path()
    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            GEOMETRY_CONFIGS = data.get("ALL_GEOMETRY_CONFIGS", [])
    else:
        GEOMETRY_CONFIGS = []

def save_geometry_configs():
    """Sauvegarde la liste GEOMETRY_CONFIGS dans le fichier JSON."""
    save_path = get_geometry_save_path()
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({"ALL_GEOMETRY_CONFIGS": GEOMETRY_CONFIGS}, f, indent=2)
    return save_path

def create_geometry_widget():
    """
    Crée et retourne un widget (VBox) contenant :
      - Des sliders pour modifier la géométrie.
      - Un champ de saisie pour le nom de la configuration.
      - Deux boutons : "Add Geometry Config" et "Save Geometry Configurations".
      - Un menu déroulant avec boutons "Load Config", "Update Config" et "Delete Config"
        pour modifier ou supprimer une configuration déjà enregistrée.
      - Une zone d'affichage pour présenter la configuration actuelle et la liste des configurations enregistrées.
    """
    geometry_sliders = {}
    slider_widgets = []
    
    # Création d'un slider pour chaque paramètre
    for key, default in geometry_config.items():
        min_val, max_val = geometry_limits.get(key, (0, 200))
        slider = widgets.FloatSlider(
            value=default, min=min_val, max=max_val, step=1,
            description=key,
            continuous_update=False,
            layout=widgets.Layout(width='300px'),
            style={'description_width': '120px'}
        )
        geometry_sliders[key] = slider
        slider_widgets.append(slider)
    
    # Champ de saisie pour le nom de la configuration
    config_name_text = widgets.Text(
        value='',
        placeholder='Nom de la configuration',
        description='Config Name:',
        layout=widgets.Layout(width='300px'),
        style={'description_width': '120px'}
    )
    
    # Boutons pour ajouter et sauvegarder les configurations
    button_add = widgets.Button(
        description="Add Geometry Config", 
        button_style='',
        layout=widgets.Layout(width='180px')
    )
    button_save = widgets.Button(
        description="Save & Quit", 
        button_style='success',
        layout=widgets.Layout(width='250px')
    )
    
    # Widgets pour modifier/supprimer une configuration existante
    config_dropdown = widgets.Dropdown(
        options=[],
        description="Saved Configs:",
        layout=widgets.Layout(width='300px'),
        style={'description_width': '120px'}
    )
    button_load = widgets.Button(
        description="Load Config", 
        button_style='',
        layout=widgets.Layout(width='120px')
    )
    button_update = widgets.Button(
        description="Update Config", 
        button_style='',
        layout=widgets.Layout(width='120px')
    )
    button_delete = widgets.Button(
        description="Delete Config", 
        button_style='danger',
        layout=widgets.Layout(width='120px')
    )
    
    output_area = widgets.Output(layout=widgets.Layout(border='1px solid gray', padding='10px'))
    
    # Charger les configurations existantes
    load_geometry_configs()
    
    def update_dropdown_options():
        # Met à jour le dropdown avec les config_name actuels
        options = [(cfg["config_name"], cfg) for cfg in GEOMETRY_CONFIGS]
        config_dropdown.options = options if options else [("None", None)]
    
    update_dropdown_options()
    
    def add_geometry_config(b):
        # Met à jour geometry_config avec les valeurs actuelles des sliders
        for key, slider in geometry_sliders.items():
            geometry_config[key] = slider.value
        # Récupération du nom de configuration
        config_name = config_name_text.value.strip()
        if not config_name:
            config_name = f"Geometry_{len(GEOMETRY_CONFIGS)+1}"
        # Création d'une copie de la configuration courante
        new_config = {"config_name": config_name, "geometry": geometry_config.copy()}
        GEOMETRY_CONFIGS.append(new_config)
        update_dropdown_options()
        save_geometry_configs()  # Sauvegarde automatique
        with output_area:
            clear_output()
            print("Configuration added:")
            print(new_config)
            print("\nList of configurations:")
            for cfg in GEOMETRY_CONFIGS:
                print(f"- {cfg['config_name']}: {cfg['geometry']}")
        config_name_text.value = ''
    
    def save_geometry_configs_btn(b):
        path = save_geometry_configs()
        with output_area:
            clear_output()
            print(f"Configurations saved in {path}")
    
    def load_config(b):
        # Charge la configuration sélectionnée dans les sliders et le champ de texte
        selected = config_dropdown.value
        if selected is None:
            with output_area:
                clear_output()
                print("No configuration selected to load.")
            return
        for key, value in selected["geometry"].items():
            if key in geometry_sliders:
                geometry_sliders[key].value = value
        config_name_text.value = selected["config_name"]
        with output_area:
            clear_output()
            print(f"Configuration '{selected['config_name']}' loaded into controls.")
    
    def update_config(b):
        # Met à jour la configuration sélectionnée avec les valeurs actuelles
        selected = config_dropdown.value
        if selected is None:
            with output_area:
                clear_output()
                print("No configuration selected for update.")
            return
        for key, slider in geometry_sliders.items():
            selected["geometry"][key] = slider.value
        new_name = config_name_text.value.strip() or selected["config_name"]
        selected["config_name"] = new_name
        update_dropdown_options()
        save_geometry_configs()  # Sauvegarde après update
        with output_area:
            clear_output()
            print(f"Configuration updated: {selected}")
    
    def delete_config(b):
        # Supprime la configuration sélectionnée de GEOMETRY_CONFIGS
        selected = config_dropdown.value
        if selected is None:
            with output_area:
                clear_output()
                print("No configuration selected for deletion.")
            return
        global GEOMETRY_CONFIGS
        GEOMETRY_CONFIGS = [cfg for cfg in GEOMETRY_CONFIGS if cfg["config_name"] != selected["config_name"]]
        update_dropdown_options()
        save_geometry_configs()  # Sauvegarde après suppression
        with output_area:
            clear_output()
            print(f"Configuration '{selected['config_name']}' deleted.")
            print("\nRemaining configurations:")
            for cfg in GEOMETRY_CONFIGS:
                print(f"- {cfg['config_name']}: {cfg['geometry']}")
    
    button_add.on_click(add_geometry_config)
    button_save.on_click(save_geometry_configs_btn)
    button_load.on_click(load_config)
    button_update.on_click(update_config)
    button_delete.on_click(delete_config)
    
    widget = widgets.VBox(
        slider_widgets +
        [config_name_text, widgets.HBox([button_add, button_save])] +
        [widgets.HBox([config_dropdown, button_load, button_update, button_delete])] +
        [output_area]
    )
    return widget

# Pour afficher le widget dans le notebook :
geometry_widget = create_geometry_widget()
