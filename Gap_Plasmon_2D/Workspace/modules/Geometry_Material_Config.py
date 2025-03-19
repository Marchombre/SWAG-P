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
    Crée un widget permettant de sélectionner dynamiquement des paires (géométrie, matériau)
    et de les enregistrer dans geom_mat_combinations.json.
    
    Chaque ligne est affichée avec un bouton "Delete" qui permet de supprimer la ligne choisie.
    Seules les lignes valides (où les deux dropdowns sont renseignées) seront enregistrées.
    """
    # Chargement des configurations existantes
    geom_data = load_json_config("geometry_configurations.json").get("ALL_GEOMETRY_CONFIGS", [])
    mat_data = load_json_config("material_config.json").get("ALL_CONFIGS", [])
    
    # Fonctions utilitaires pour créer les dropdowns
    def get_geom_dropdown():
        options = [(cfg["config_name"], cfg) for cfg in geom_data] if geom_data else [("None", None)]
        return widgets.Dropdown(
            options=options,
            value=options[0][1] if options[0][1] is not None else None,
            description="Geometry:",
            layout=widgets.Layout(width='250px'),
            style={'description_width': '80px'}
        )
    
    def get_mat_dropdown():
        options = [(cfg["config_name"], cfg) for cfg in mat_data] if mat_data else [("None", None)]
        return widgets.Dropdown(
            options=options,
            value=options[0][1] if options[0][1] is not None else None,
            description="Material:",
            layout=widgets.Layout(width='250px'),
            style={'description_width': '80px'}
        )
    
    # Listes globales pour stocker les lignes
    row_widgets = []  # Chaque élément sera un dict: {'container': row, 'geom': geom_dd, 'mat': mat_dd, 'delete_btn': btn}
    
    # Container pour les lignes
    rows_container = widgets.VBox([], layout=widgets.Layout(margin='10px 0'))
    
    def update_rows_container():
        rows_container.children = [item['container'] for item in row_widgets]
    
    def create_new_row():
        geom_dd = get_geom_dropdown()
        mat_dd = get_mat_dropdown()
        delete_btn = widgets.Button(
            description="Delete",
            button_style="danger",
            layout=widgets.Layout(width='80px')
        )
        row = widgets.HBox([geom_dd, mat_dd, delete_btn],
                           layout=widgets.Layout(margin='5px 0', align_items='center'))
        # Stockage de la ligne
        row_widgets.append({'container': row, 'geom': geom_dd, 'mat': mat_dd, 'delete_btn': delete_btn})
        update_rows_container()
        
        def on_delete_clicked(b):
            # Supprime la ligne correspondante
            for item in row_widgets:
                if item['container'] == row:
                    row_widgets.remove(item)
                    break
            update_rows_container()
        delete_btn.on_click(on_delete_clicked)
    
    # Création d'une première ligne
    create_new_row()
    
    # Boutons pour ajouter ou supprimer des lignes
    add_row_btn = widgets.Button(
        description="Add Row",
        button_style="info",
        layout=widgets.Layout(width="120px")
    )
    add_row_btn.on_click(lambda b: create_new_row())
    
    # Bouton pour combiner et sauvegarder
    combine_button = widgets.Button(
        description="Combine & Save",
        button_style="success",
        layout=widgets.Layout(width="150px")
    )
    output_area = widgets.Output(layout=widgets.Layout(border='1px solid #ccc', padding='10px'))
    
    def on_combine_clicked(b):
        with output_area:
            clear_output()
            combined_configs = []
            # Parcourt chaque ligne
            for item in row_widgets:
                geom_cfg = item['geom'].value
                mat_cfg = item['mat'].value
                # Si l'une ou l'autre cellule est None, on ignore cette ligne
                if geom_cfg is None or mat_cfg is None:
                    continue
                combined_name = f"{geom_cfg['config_name']} - {mat_cfg['config_name']}"
                combined = {
                    "config_name": combined_name,
                    "geometry": geom_cfg,
                    "material": mat_cfg
                }
                combined_configs.append(combined)
            if combined_configs:
                print("Combined Geometry-Material Configurations:")
                for idx, cfg in enumerate(combined_configs, start=1):
                    print(f"Row {idx} - {cfg['config_name']}:")
                    print("  geometry:", cfg["geometry"]["config_name"])
                    print("  material:", cfg["material"]["config_name"])
                    print("-" * 40)
                # Enregistrement dans geom_mat_combinations.json
                module_dir = os.path.dirname(os.path.abspath(__file__))
                workspace_dir = os.path.dirname(module_dir)
                notebooks_dir = os.path.join(workspace_dir, "notebooks")
                summary_dir = os.path.join(notebooks_dir, "Summary_Simulation")
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                combos_file = os.path.join(summary_dir, "geom_mat_combinations.json")
                with open(combos_file, "w", encoding="utf-8") as f:
                    json.dump({"ALL_COMBINED_CONFIGS": combined_configs}, f, indent=2)
                print(f"\nCombinations saved in {combos_file}")
            else:
                print("No valid combination selected.")
    
    combine_button.on_click(on_combine_clicked)
    
    # Assemblage final du widget
    control_buttons = widgets.HBox([add_row_btn, combine_button],
                                   layout=widgets.Layout(justify_content='space-around', margin='10px 0'))
    
    main_widget = widgets.VBox(
        [rows_container, control_buttons, output_area],
        layout=widgets.Layout(border='2px solid #333', padding='10px', width='650px', background_color='#f9f9f9')
    )
    return main_widget

