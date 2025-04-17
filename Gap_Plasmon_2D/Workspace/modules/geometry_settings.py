#!/usr/bin/env python3
# geometry_settings.py

import os, json
import ipywidgets as widgets
from IPython.display import clear_output, display
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Configuration par défaut (les valeurs réelles utilisées pour les calculs de réflectance restent inchangées)
geometry_config = {
    "thick_super": 200,        # Épaisseur réelle du Superstrate
    "thick_reso": 30,          # Hauteur réelle du Nanocube
    "width_reso": 30,          # Largeur réelle du Nanocube
    "thick_gap": 3,            # Épaisseur réelle du Gap (polymère)
    "thick_mol": 0.5,          # Épaisseur réelle de la Molécule
    "thick_func": 1,           # Épaisseur réelle de la Fonctionnalisation
    "thick_diel": 1,           # Épaisseur réelle du Polymère
    "thick_metalliclayer": 10, # Épaisseur réelle du Métallique
    "thick_XIAOYI": 2,         # Épaisseur réelle du XIAOYI
    "thick_accroche": 1,       # Épaisseur réelle de l'Accroche
    "thick_sub": 200,          # Épaisseur réelle du Substrate
    "period": 100.2153         # Cellule RCWA (carrée)
}

geometry_limits = {
    "thick_super": (0, 300),
    "thick_reso": (0, 100),
    "width_reso": (0, 100),
    "thick_gap": (0, 30),
    "thick_mol": (0, 5),
    "thick_func": (0, 5),
    "thick_diel": (0, 30),
    "thick_metalliclayer": (0, 50),
    "thick_XIAOYI": (0, 10),
    "thick_accroche": (0, 20),
    "thick_sub": (0, 300),
    "period": (50, 300)
}

def displayed_thickness(t):
    """
    Retourne 1/10 de l'épaisseur réelle pour l'affichage du Substrate et du Superstrate.
    """
    return t / 10

def get_geometry_save_path():
    """
    Retourne le chemin vers le fichier JSON utilisé pour stocker les configurations géométriques.
    """
    module_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    workspace_dir = os.path.dirname(module_dir)
    CONFIG_DIR = os.path.join(workspace_dir, "CONFIGURATIONS")
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR)
    return os.path.join(CONFIG_DIR, "geometry_configurations.json")

GEOMETRY_CONFIGS = []

def load_geometry_configs():
    """
    Charge les configurations géométriques sauvegardées depuis le fichier JSON, s'il existe.
    """
    global GEOMETRY_CONFIGS
    save_path = get_geometry_save_path()
    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            GEOMETRY_CONFIGS = data.get("ALL_GEOMETRY_CONFIGS", [])
    else:
        GEOMETRY_CONFIGS = []

def save_geometry_configs():
    """
    Sauvegarde l'ensemble des configurations géométriques actuelles dans le fichier JSON.
    """
    save_path = get_geometry_save_path()
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({"ALL_GEOMETRY_CONFIGS": GEOMETRY_CONFIGS}, f, indent=2)
    return save_path

def draw_layer(ax, x, y, w, h, color, label, hatch=None):
    """
    Dessine une couche rectangulaire sur l'objet Axes 'ax',
    avec un libellé optionnel (affiché au centre) et éventuellement un motif de hachures.
    Si h <= 0, la couche n'est pas dessinée.
    """
    if h <= 0:
        return
    rect = patches.Rectangle((x, y), w, h, edgecolor=None, facecolor=color, hatch=hatch)
    ax.add_patch(rect)
    if label and h > 0:
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=9)

def create_geometry_widget():
    """
    Crée une interface basée sur ipywidgets pour ajuster les paramètres de la géométrie,
    sauvegarder/charger les configurations, et afficher le schéma.
    """
    ordered_params = [
        ("thick_super", "Superstrate"),
        ("thick_reso", "Nanocube height"),
        ("width_reso", "Nanocube width"),
        ("thick_gap", "Gap (polymer)"),
        ("thick_mol", "Molecule"),
        ("thick_func", "Functionalisation"),
        ("thick_diel", "Dielectric"),
        ("thick_metalliclayer", "Metallic"),
        ("thick_XIAOYI", "XIAOYI"),
        ("thick_accroche", "Accroche"),
        ("thick_sub", "Substrate"),
        ("period", "Period")
    ]
    
    geometry_sliders = {}
    slider_widgets = []
    
    for key, label in ordered_params:
        default = geometry_config.get(key, 0)
        min_val, max_val = geometry_limits.get(key, (0, 200))
        
        if key == "period":
            description_str = label + ":"
        else:
            description_str = label + " (nm):"
        # Création d'un slider avec un FloatText lié pour chaque paramètre
        slider = widgets.FloatSlider(
            value=default, min=min_val, max=max_val, step=0.1,
            description=description_str,
            continuous_update=False,
            layout=widgets.Layout(width='350px'),
            style={'description_width': '180px'}
        )
        float_text = widgets.FloatText(value=default, layout=widgets.Layout(width='100px'))
        widgets.jslink((slider, 'value'), (float_text, 'value'))
        
        # Empêche les valeurs négatives
        def validate_positive(change):
            if change['new'] < 0:
                change['owner'].value = 0
        float_text.observe(validate_positive, names='value')
        
        geometry_sliders[key] = slider
        slider_widgets.append(widgets.HBox([slider, float_text]))
    
    # Widgets de configuration
    config_name_text = widgets.Text(
        value='',
        placeholder='Configuration Name',
        description='Config Name :',
        layout=widgets.Layout(width='350px'),
        style={'description_width': '180px'}
    )
    # Widget pour saisir le nom du compartiment
    compartment_text = widgets.Text(
        value='',
        placeholder='Nom du compartiment',
        description='Compartiment :',
        layout=widgets.Layout(width='350px'),
        style={'description_width': '180px'}
    )
    
    button_add = widgets.Button(description="Add Config", layout=widgets.Layout(width='150px'))
    button_save = widgets.Button(description="Save & Quit", button_style='success', layout=widgets.Layout(width='200px'))
    config_dropdown = widgets.Dropdown(
        options=[],
        description="Saved Configs :",
        layout=widgets.Layout(width='350px'),
        style={'description_width': '180px'}
    )
    button_load = widgets.Button(description="Load", layout=widgets.Layout(width='100px'))
    button_update = widgets.Button(description="Update", layout=widgets.Layout(width='120px'))
    button_delete = widgets.Button(description="Delete", button_style='danger', layout=widgets.Layout(width='120px'))
    
    # Widget de filtrage par compartiment
    compartment_filter = widgets.Dropdown(
        options=["Tous"],
        value="Tous",
        description="Filtrer Compart.:",
        layout=widgets.Layout(width='350px'),
        style={'description_width': '180px'}
    )
    
    output_area = widgets.Output(layout=widgets.Layout(padding='10px'))
    load_geometry_configs()
    
    def update_dropdown_options(change=None):
        # Sélection du filtre et reconstruction de la liste en fonction du compartiment sélectionné
        filtre = compartment_filter.value
        if filtre == "Tous":
            filtered_configs = GEOMETRY_CONFIGS
        else:
            filtered_configs = [cfg for cfg in GEOMETRY_CONFIGS if cfg.get("compartment", "Défaut") == filtre]
        new_options = [(f"{cfg['config_name']} ({cfg.get('compartment', 'Défaut')})", cfg) 
                       for cfg in filtered_configs]
        if not new_options:
            new_options = [("None", None)]
        # Affectation directe pour forcer l'actualisation
        config_dropdown.options = new_options

        # Reconstruction et affectation de la liste des compartiments disponibles
        compartments = sorted({cfg.get("compartment", "Défaut") for cfg in GEOMETRY_CONFIGS})
        comp_options = ["Tous"] + compartments
        compartment_filter.options = comp_options

    # Mise à jour initiale du dropdown
    update_dropdown_options()
    # Observer le changement de sélection du compartiment pour rafraîchir la liste
    compartment_filter.observe(update_dropdown_options, names='value')
    
    def add_geometry_config(_):
        for key, slider in geometry_sliders.items():
            geometry_config[key] = slider.value
        config_name = config_name_text.value.strip() or f"Geometry_{len(GEOMETRY_CONFIGS)+1}"
        compartment = compartment_text.value.strip() or "Défaut"
        new_config = {"config_name": config_name, "compartment": compartment, "geometry": geometry_config.copy()}
        GEOMETRY_CONFIGS.append(new_config)
        update_dropdown_options()
        save_geometry_configs()
        with output_area:
            clear_output()
            print("Configuration ajoutée :")
            print(new_config)
        config_name_text.value = ''
        compartment_text.value = ''
    
    def save_geometry_configs_btn(_):
        path = save_geometry_configs()
        with output_area:
            clear_output()
            print(f"Configurations saved in {path}")
    
    def load_config(_):
        """
        Charge la configuration sélectionnée dans le widget et met à jour les sliders ainsi que
        les champs de nom et de compartiment.
        """
        selected = config_dropdown.value
        if selected is None:
            with output_area:
                clear_output()
                print("Aucune configuration sélectionnée pour le chargement.")
            return
        # Mise à jour des sliders avec les valeurs de la configuration chargée
        for key, value in selected["geometry"].items():
            if key in geometry_sliders:
                geometry_sliders[key].value = value
        # Mise à jour des champs de nom et du compartiment
        config_name_text.value = selected["config_name"]
        compartment_text.value = selected.get("compartment", "Défaut")
        with output_area:
            clear_output()
            print(f"Configuration '{selected['config_name']}' chargée.")

    def update_config(_):
        """
        Met à jour la configuration actuellement chargée en récupérant la nouvelle valeur
        des sliders, du nom et du compartiment, puis enregistre la configuration mise à jour.
        """
        selected = config_dropdown.value
        if selected is None:
            with output_area:
                clear_output()
                print("Aucune configuration sélectionnée pour la mise à jour.")
            return
        for key, slider in geometry_sliders.items():
            selected["geometry"][key] = slider.value
        new_name = config_name_text.value.strip() or selected["config_name"]
        selected["config_name"] = new_name
        selected["compartment"] = compartment_text.value.strip() or "Défaut"
        update_dropdown_options()
        save_geometry_configs()
        with output_area:
            clear_output()
            print(f"Configuration mise à jour : {selected}")

    def delete_config(_):
        selected = config_dropdown.value
        if selected is None:
            with output_area:
                clear_output()
                print("Aucune configuration sélectionnée pour la suppression.")
            return
        global GEOMETRY_CONFIGS
        GEOMETRY_CONFIGS = [cfg for cfg in GEOMETRY_CONFIGS if cfg["config_name"] != selected["config_name"]]
        update_dropdown_options()
        save_geometry_configs()
        with output_area:
            clear_output()
            print(f"Configuration '{selected['config_name']}' supprimée.")
    
    button_add.on_click(add_geometry_config)
    button_save.on_click(save_geometry_configs_btn)
    button_load.on_click(load_config)
    button_update.on_click(update_config)
    button_delete.on_click(delete_config)
    
    # Widget pour afficher le cas (Case A ou Case B) en dehors du dessin
    case_label_widget = widgets.Label(value="")
    
    # Zone de dessin (zone de travail réduite)
    fig_output = widgets.Output(layout=widgets.Layout(flex='1', height='700px', padding='10px'))
    
    def draw_structure(_=None):
        with fig_output:
            clear_output(wait=True)
            
            # Récupération des paramètres via les sliders
            p            = geometry_sliders["period"].value
            t_super      = geometry_sliders["thick_super"].value
            t_reso       = geometry_sliders["thick_reso"].value
            w_reso       = geometry_sliders["width_reso"].value
            t_gap        = geometry_sliders["thick_gap"].value
            t_mol        = geometry_sliders["thick_mol"].value
            t_func       = geometry_sliders["thick_func"].value
            t_diel       = geometry_sliders["thick_diel"].value
            t_metal      = geometry_sliders["thick_metalliclayer"].value
            t_XIAOYI     = geometry_sliders["thick_XIAOYI"].value
            t_acc        = geometry_sliders["thick_accroche"].value
            t_sub        = geometry_sliders["thick_sub"].value

            # Paramètres d'affichage pour les couches fixes
            disp_sub   = displayed_thickness(t_sub)   # Affichage réduit du Substrate
            disp_super = displayed_thickness(t_super)   # Affichage réduit du Superstrate

            # Hauteur totale du dessin (on utilise p comme dimension du carré)
            hauteur_totale = p

            # Hauteur disponible pour les couches intermédiaires
            hauteur_dispo = hauteur_totale - (disp_sub + disp_super)
            
            # --- Partie centrale (empilement vertical) ---
            somme_centrale = t_acc + t_XIAOYI + t_metal + t_gap + t_reso
            facteur_centrale = hauteur_dispo / somme_centrale if somme_centrale > 0 else 1
            disp_acc    = t_acc     * facteur_centrale
            disp_XIAOYI = t_XIAOYI  * facteur_centrale
            disp_metal  = t_metal   * facteur_centrale
            disp_gap    = t_gap     * facteur_centrale
            disp_reso   = t_reso    * facteur_centrale
            
            y_sub_bottom = 0
            y_sub_top    = y_sub_bottom + disp_sub
            
            y_acc_bottom = y_sub_top
            y_acc_top    = y_acc_bottom + disp_acc
            
            y_XIAOYI_bottom = y_acc_top
            y_XIAOYI_top    = y_XIAOYI_bottom + disp_XIAOYI
            
            y_metal_bottom = y_XIAOYI_top
            y_metal_top    = y_metal_bottom + disp_metal
            
            y_inter_bottom = y_metal_top  # Début de la zone centrale (Gap + Cube)
            y_gap_bottom   = y_inter_bottom
            y_gap_top      = y_gap_bottom + disp_gap
            
            y_cube_bottom  = y_gap_top
            y_cube_top     = y_cube_bottom + disp_reso

            y_super_bottom = y_cube_top
            y_super_top    = hauteur_totale

            # --- Partie latérale (colonnes gauche et droite) ---
            somme_laterale = t_diel + t_func + t_mol
            facteur_lateral = facteur_centrale
            disp_dielectric = t_diel * facteur_lateral
            disp_func       = t_func   * facteur_lateral
            disp_mol        = t_mol    * facteur_lateral
            hauteur_laterale = disp_dielectric + disp_func + disp_mol
            y_lat_start = y_inter_bottom
            y_lat_end   = y_lat_start + hauteur_laterale
            lateral_filler = max(0, y_super_bottom - y_lat_end)
            
            # Définition du cas (Case A ou Case B)
            if (t_diel + t_func + t_mol) < t_gap:
                case_str = "Case A"
            else:
                case_str = "Case B"
            case_label_widget.value = f"Case: {case_str}"
            
            # Coordonnées latérales
            central_x     = (p - w_reso) / 2
            lateral_width = (p - w_reso) / 2
            left_x  = 0
            right_x = central_x + w_reso
            
            # Dessin
            fig, ax = plt.subplots(figsize=(6,6))
            ax.set_title("Schematics (visualisation uniquement) - " + case_label_widget.value, fontsize=10, pad=5)
            
            # Substrate (zone inférieure)
            draw_layer(ax, 0, y_sub_bottom, p, disp_sub, "brown", "Substrate")
            bande_height = min(0.05 * p, disp_sub, disp_super)
            draw_layer(ax, 0, y_sub_bottom, p, bande_height, "none", "", hatch='///')
            
            # Partie centrale
            draw_layer(ax, 0, y_acc_bottom, p, disp_acc, "gold", "Accroche")
            draw_layer(ax, 0, y_XIAOYI_bottom, p, disp_XIAOYI, "purple", "XIAOYI")
            draw_layer(ax, 0, y_metal_bottom, p, disp_metal, "silver", "Metallic")
            draw_layer(ax, central_x, y_gap_bottom, w_reso, disp_gap, "lightgreen", "Gap (polymer)")
            draw_layer(ax, central_x, y_cube_bottom, w_reso, disp_reso, "orange", "Nanocube")
            
            # Colonnes latérales
            y_curr_left = y_lat_start
            draw_layer(ax, left_x,  y_curr_left, lateral_width, disp_dielectric, "green", "Dielectric")
            draw_layer(ax, right_x, y_curr_left, lateral_width, disp_dielectric, "green", "Dielectric")
            y_curr_left += disp_dielectric
            draw_layer(ax, left_x,  y_curr_left, lateral_width, disp_func, "pink", "Functionalisation")
            draw_layer(ax, right_x, y_curr_left, lateral_width, disp_func, "pink", "Functionalisation")
            y_curr_left += disp_func
            draw_layer(ax, left_x,  y_curr_left, lateral_width, disp_mol, "violet", "Molecule")
            draw_layer(ax, right_x, y_curr_left, lateral_width, disp_mol, "violet", "Molecule")
            y_curr_left += disp_mol
            if lateral_filler > 0:
                draw_layer(ax, left_x,  y_curr_left, lateral_width, lateral_filler, "lightblue", "Environnement")
                draw_layer(ax, right_x, y_curr_left, lateral_width, lateral_filler, "lightblue", "Environnement")
            
            # Superstrate (zone supérieure)
            draw_layer(ax, 0, y_super_bottom, p, hauteur_totale - y_super_bottom, "lightblue", "Superstrate\n(environnement)")
            draw_layer(ax, 0, hauteur_totale - bande_height, p, bande_height, "none", "", hatch='///')
            
            ax.set_xlim(0, p)
            ax.set_ylim(0, p)
            ax.set_aspect('equal', adjustable='box')
            ax.margins(0)
            
            plt.show()
    
    # Lancer le dessin à chaque modification des sliders
    for sld in geometry_sliders.values():
        sld.observe(draw_structure, names='value')
    
    draw_structure()  # Premier dessin
    
    config_controls = widgets.VBox([
        config_name_text,
        compartment_text,
        widgets.HBox([button_add, button_save]),
        compartment_filter,
        widgets.HBox([config_dropdown, button_load, button_update, button_delete]),
        output_area
    ])
    
    left_panel = widgets.VBox(slider_widgets + [config_controls], layout=widgets.Layout(width='600px'))
    right_panel = widgets.VBox([fig_output], layout=widgets.Layout(flex='1'))
    main_ui = widgets.HBox([left_panel, right_panel], layout=widgets.Layout(width='100%'))
    
    return main_ui

geometry_widget = create_geometry_widget()

