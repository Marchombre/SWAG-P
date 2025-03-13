# geometry_settings.py
import ipywidgets as widgets
from IPython.display import display, clear_output
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
    "thick_gap": (0, 50),
    "thick_func": (0, 20),
    "thick_mol": (0, 20),
    "thick_metalliclayer": (0, 50),
    "thick_sub": (0, 300),
    "thick_accroche": (0, 20),
    "period": (50, 300)
}

# Liste globale pour stocker les configurations de géométrie enregistrées
GEOMETRY_CONFIGS = []

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
    
    # Création d'un slider pour chaque paramètre de geometry_config
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
    
    # Champ de saisie pour le nom de la configuration
    config_name_text = widgets.Text(
        value='',
        placeholder='Nom de la configuration',
        description='Config Name:',
        style={'description_width': 'initial'}
    )
    
    # Boutons pour ajouter et sauvegarder les configurations
    button_add = widgets.Button(description="Add Geometry Config")
    button_save = widgets.Button(description="Save Geometry Configurations")
    
    # Widgets pour modifier/supprimer une configuration existante
    config_dropdown = widgets.Dropdown(
        options=[],
        description="Select Config:",
        style={'description_width': 'initial'}
    )
    button_load = widgets.Button(description="Load Config")
    button_update = widgets.Button(description="Update Config")
    button_delete = widgets.Button(description="Delete Config")
    
    output_area = widgets.Output()
    
    global GEOMETRY_CONFIGS
    GEOMETRY_CONFIGS = []
    
    def update_dropdown_options():
        # Met à jour le dropdown avec les config_name actuels
        options = [(cfg["config_name"], cfg) for cfg in GEOMETRY_CONFIGS]
        config_dropdown.options = options if options else [("None", None)]
    
    def add_geometry_config(b):
        # Mise à jour de geometry_config avec les valeurs actuelles des sliders
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
        with output_area:
            clear_output()
            print("Configuration ajoutée :")
            print(new_config)
            print("\nListe des configurations enregistrées :")
            for cfg in GEOMETRY_CONFIGS:
                print(f"- {cfg['config_name']}: {cfg['geometry']}")
        # Réinitialiser le champ de nom
        config_name_text.value = ''
    
    def save_geometry_configs(b):
        # Détermination du chemin Summary_Simulation
        module_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_dir = os.path.dirname(module_dir)
        notebooks_dir = os.path.join(workspace_dir, "notebooks")
        summary_dir = os.path.join(notebooks_dir, "Summary_Simulation")
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
        # Sauvegarde de GEOMETRY_CONFIGS dans un fichier JSON dans summary_dir
        save_path = os.path.join(summary_dir, "geometry_configurations.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({"ALL_GEOMETRY_CONFIGS": GEOMETRY_CONFIGS}, f, indent=2)
        with output_area:
            print(f"\nConfigurations sauvegardées dans {save_path}")
        # Rendre les configurations accessibles globalement
        import __main__
        __main__.GEOMETRY_CONFIGS = GEOMETRY_CONFIGS
    
    def load_config(b):
        # Charge la configuration sélectionnée dans les sliders et le champ de nom
        selected = config_dropdown.value
        if selected is None:
            with output_area:
                clear_output()
                print("Aucune configuration sélectionnée à charger.")
            return
        # Mise à jour des sliders et du champ de texte
        for key, value in selected["geometry"].items():
            if key in geometry_sliders:
                geometry_sliders[key].value = value
        config_name_text.value = selected["config_name"]
        with output_area:
            clear_output()
            print(f"Configuration '{selected['config_name']}' chargée dans les contrôles.")
    
    def update_config(b):
        # Met à jour la configuration sélectionnée avec les valeurs actuelles
        selected = config_dropdown.value
        if selected is None:
            with output_area:
                clear_output()
                print("Aucune configuration sélectionnée pour la mise à jour.")
            return
        # Mise à jour de la configuration dans GEOMETRY_CONFIGS
        for key, slider in geometry_sliders.items():
            selected["geometry"][key] = slider.value
        # Mise à jour du nom si le champ est différent
        new_name = config_name_text.value.strip() or selected["config_name"]
        selected["config_name"] = new_name
        update_dropdown_options()
        with output_area:
            clear_output()
            print(f"Configuration mise à jour : {selected}")
    
    def delete_config(b):
        # Supprime la configuration sélectionnée de GEOMETRY_CONFIGS
        selected = config_dropdown.value
        if selected is None:
            with output_area:
                clear_output()
                print("Aucune configuration sélectionnée pour la suppression.")
            return
        # Retirer la configuration
        global GEOMETRY_CONFIGS
        GEOMETRY_CONFIGS = [cfg for cfg in GEOMETRY_CONFIGS if cfg["config_name"] != selected["config_name"]]
        update_dropdown_options()
        with output_area:
            clear_output()
            print(f"Configuration '{selected['config_name']}' supprimée.")
            print("\nListe des configurations restantes :")
            for cfg in GEOMETRY_CONFIGS:
                print(f"- {cfg['config_name']}: {cfg['geometry']}")
    
    button_add.on_click(add_geometry_config)
    button_save.on_click(save_geometry_configs)
    button_load.on_click(load_config)
    button_update.on_click(update_config)
    button_delete.on_click(delete_config)
    
    # Assemblage du widget : sliders, champ de texte, boutons add & save, puis le dropdown et boutons modif/suppression, puis la zone d'output.
    widget = widgets.VBox(
        slider_widgets +
        [config_name_text, button_add, button_save] +
        [widgets.HBox([config_dropdown, button_load, button_update, button_delete])] +
        [output_area]
    )
    return widget

# Pour afficher le widget dans un Notebook, utilisez :
# widget = create_geometry_widget()
# display(widget)
