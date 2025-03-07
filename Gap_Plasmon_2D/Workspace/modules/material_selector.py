# material_selector.py
import ipywidgets as widgets
import pandas as pd
from IPython.display import display
from material_list_provider import get_available_materials

# Liste par défaut des rôles utilisés dans votre simulation
DEFAULT_ROLES = [
    "perm_env", "perm_dielec", "perm_sub", "perm_reso",
    "perm_metalliclayer", "perm_accroche", "perm_func", "perm_mol"
]

def create_material_selector(json_path, roles=DEFAULT_ROLES):
    """
    Crée et affiche une interface interactive (dropdowns + zones Custom)
    pour la sélection des matériaux. La liste des matériaux est générée dynamiquement
    à partir du fichier JSON combiné.
    
    Paramètres
    ----------
    json_path : str
        Chemin vers le fichier JSON combiné.
    roles : list, optionnel
        Liste des rôles à utiliser (par défaut DEFAULT_ROLES).
    
    Retourne
    --------
    widget_container : VBox
        Conteneur (VBox) contenant l'ensemble des widgets,
        ainsi que le bouton de validation et une zone de sortie pour le récapitulatif.
        La configuration finale sera stockée dans la variable globale MATERIALS_CONFIG.
    """
    # Récupérer la liste dynamique des matériaux disponibles
    all_materials = get_available_materials(json_path)
    # Ajouter les options "None" et "Custom"
    all_materials_with_options = ["None", "Custom"] + all_materials

    dropdowns = {}
    text_inputs = {}
    widget_boxes = []
    
    for role in roles:
        dropdown = widgets.Dropdown(
            options=all_materials_with_options,
            description=role,
            style={'description_width': 'initial'}
        )
        text_input = widgets.Text(
            value="",
            description="Custom:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(visibility='hidden')
        )
        
        def on_dropdown_change(change, txt=text_input):
            if change['new'] == "Custom":
                txt.layout.visibility = 'visible'
            else:
                txt.layout.visibility = 'hidden'
                txt.value = ""
                
        dropdown.observe(lambda change, t=text_input: on_dropdown_change(change, t), names='value')
        
        dropdowns[role] = dropdown
        text_inputs[role] = text_input
        
        widget_boxes.append(widgets.HBox([dropdown, text_input]))
    
    button_create_df = widgets.Button(description="Valider la configuration matériaux")
    output_df = widgets.Output()
    
    def on_create_df(b):
        config = {"key": [], "material": []}
        for role in roles:
            if dropdowns[role].value == "Custom":
                mat_value = text_inputs[role].value.strip()
                if mat_value == "":
                    mat_value = "None"
            else:
                mat_value = dropdowns[role].value
            config["key"].append(role)
            config["material"].append(mat_value)
        df_config = pd.DataFrame(config)
        with output_df:
            output_df.clear_output()
            print("Configuration matériaux sélectionnée:")
            display(df_config)
        import __main__
        __main__.MATERIALS_CONFIG = df_config
        print("MATERIALS_CONFIG a été défini dans l'espace global __main__")

    
    button_create_df.on_click(on_create_df)
    
    widget_container = widgets.VBox(widget_boxes + [button_create_df, output_df])
    return widget_container
