# geometry_material_widget.py
import ipywidgets as widgets
from IPython.display import display, clear_output
import os
import json

def load_json_config(file_name):
    """
    Charge un fichier JSON situé dans Summary_Simulation.
    """
    module_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(module_dir)
    notebooks_dir = os.path.join(workspace_dir, "notebooks")
    summary_dir = os.path.join(notebooks_dir, "Summary_Simulation")
    file_path = os.path.join(summary_dir, file_name)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def create_geometry_material_widget():
    """
    Crée un widget permettant de sélectionner des paires (géométrie, matériau) et de les
    enregistrer dans geom_mat_combinations.json.
    """
    # 1) Chargement des configurations existantes
    geom_data = load_json_config("geometry_configurations.json").get("ALL_GEOMETRY_CONFIGS", [])
    mat_data = load_json_config("material_config.json").get("ALL_CONFIGS", [])

    # 2) Nombre de lignes = max(nb config géométrie, nb config matériau)
    n_rows = max(len(geom_data), len(mat_data))

    # 3) Création des menus déroulants
    geom_dropdowns = []
    mat_dropdowns = []

    for i in range(n_rows):
        # Dropdown Géométrie
        geom_options = [(cfg["config_name"], cfg) for cfg in geom_data] if geom_data else [("None", None)]
        default_geom = geom_data[i] if i < len(geom_data) else None
        geom_dd = widgets.Dropdown(
            options=geom_options,
            value=default_geom,
            description=f"Geometry {i+1}:",
            style={'description_width': 'initial'}
        )
        geom_dropdowns.append(geom_dd)
        
        # Dropdown Matériaux
        mat_options = [(cfg["config_name"], cfg) for cfg in mat_data] if mat_data else [("None", None)]
        default_mat = mat_data[i] if i < len(mat_data) else None
        mat_dd = widgets.Dropdown(
            options=mat_options,
            value=default_mat,
            description=f"Material {i+1}:",
            style={'description_width': 'initial'}
        )
        mat_dropdowns.append(mat_dd)

    # 4) Construction des lignes
    rows = [widgets.HBox([geom_dropdowns[i], mat_dropdowns[i]]) for i in range(n_rows)]

    # 5) Bouton pour combiner et sauvegarder
    combine_button = widgets.Button(description="Combine & Save")
    output_area = widgets.Output()

    def on_combine_clicked(b):
        with output_area:
            clear_output()
            combined_configs = []
            # On parcourt chaque ligne
            for i in range(n_rows):
                geom_cfg = geom_dropdowns[i].value
                mat_cfg = mat_dropdowns[i].value
                if geom_cfg is not None and mat_cfg is not None:
                    # Nom combiné
                    combined_name = f"{geom_cfg['config_name']} - {mat_cfg['config_name']}"
                    combined = {
                        "config_name": combined_name,
                        "geometry": geom_cfg,      # dict: {"config_name":..., "geometry":{...}}
                        "material": mat_cfg        # dict: {"config_name":..., "MATERIALS_CONFIG":..., ...}
                    }
                    combined_configs.append(combined)

            if combined_configs:
                print("Combined Geometry-Material Configurations:")
                for idx, cfg in enumerate(combined_configs, start=1):
                    print(f"Row {idx} - {cfg['config_name']}:")
                    print("  geometry:", cfg["geometry"]["config_name"])
                    print("  material:", cfg["material"]["config_name"])
                    print("-" * 40)
                
                # On enregistre dans geom_mat_combinations.json
                module_dir = os.path.dirname(os.path.abspath(__file__))
                workspace_dir = os.path.dirname(module_dir)
                notebooks_dir = os.path.join(workspace_dir, "notebooks")
                summary_dir = os.path.join(notebooks_dir, "Summary_Simulation")
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                
                combos_file = os.path.join(summary_dir, "geom_mat_combinations.json")
                with open(combos_file, "w", encoding="utf-8") as f:
                    json.dump({"ALL_COMBINED_CONFIGS": combined_configs}, f, indent=2)
                
                print(f"\nCombinaisons sauvegardées dans {combos_file}")
            else:
                print("Aucune configuration combinée valide n'a été sélectionnée.")

    combine_button.on_click(on_combine_clicked)

    # 6) Assemblage final
    grid_widget = widgets.VBox(rows + [combine_button, output_area])
    return grid_widget
