#!/usr/bin/env python3
# geometry_settings.py

import ipywidgets as widgets
from IPython.display import clear_output, display
import os, json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Default configuration (real values used for reflectance calculations remain unchanged)
geometry_config = {
    "thick_super": 200,        # Real Superstrate thickness
    "thick_reso": 30,          # Real Nanocube height
    "width_reso": 30,          # Real Nanocube width
    "thick_gap": 3,            # Real Gap (polymer) thickness
    "thick_mol": 0.5,          # Real Molecule thickness
    "thick_func": 1,           # Real Functionalisation thickness
    "thick_diel": 1,           # Real Polymer thickness
    "thick_metalliclayer": 10, # Real Metallic thickness
    "thick_accroche": 1,       # Real Accroche thickness
    "thick_sub": 200,          # Real Substrate thickness
    "period": 100.2153         # RCWA cell (square)
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
    "thick_accroche": (0, 20),
    "thick_sub": (0, 300),
    "period": (50, 300)
}

def displayed_thickness(t):
    """
    Return 1/10 of the real thickness for Substrate and Superstrate display.
    """
    return t / 10

def get_geometry_save_path():
    """
    Return the path to the JSON file used to store geometry configurations.
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
    Load all saved geometry configurations from the JSON file, if it exists.
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
    Save all current geometry configurations to the JSON file.
    """
    save_path = get_geometry_save_path()
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({"ALL_GEOMETRY_CONFIGS": GEOMETRY_CONFIGS}, f, indent=2)
    return save_path

def draw_layer(ax, x, y, w, h, color, label, hatch=None):
    """
    Draw a single rectangular layer on the given Axes 'ax',
    with an optional label (displayed at the center) and optional hatching.
    If h <= 0, the layer is not drawn.
    """
    if h <= 0:
        return
    # Ici, on peut conserver un contour fin si souhaité, ou le supprimer en définissant edgecolor à None.
    rect = patches.Rectangle((x, y), w, h, edgecolor=None, facecolor=color, hatch=hatch)
    ax.add_patch(rect)
    if label and h > 0:
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=9)

def create_geometry_widget():
    """
    Create an ipywidgets-based interface for adjusting the geometry parameters,
    saving/loading configurations, and displaying the schematic figure.
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
        ("thick_accroche", "Accroche"),
        ("thick_sub", "Substrate"),
        ("period", "Period")
    ]
    
    geometry_sliders = {}
    slider_widgets = []
    
    for key, label in ordered_params:
        default = geometry_config.get(key, 0)
        min_val, max_val = geometry_limits.get(key, (0, 200))
        slider = widgets.FloatSlider(
            value=default, min=min_val, max=max_val, step=0.1,
            description=label + " :",
            continuous_update=False,
            layout=widgets.Layout(width='350px'),
            style={'description_width': '180px'}
        )
        float_text = widgets.FloatText(value=default, layout=widgets.Layout(width='100px'))
        widgets.jslink((slider, 'value'), (float_text, 'value'))
        geometry_sliders[key] = slider
        slider_widgets.append(widgets.HBox([slider, float_text]))
    
    # Configuration controls
    config_name_text = widgets.Text(
        value='',
        placeholder='Configuration Name',
        description='Config Name :',
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
    
    # Suppression de la bordure pour output_area
    output_area = widgets.Output(layout=widgets.Layout(padding='10px'))
    
    load_geometry_configs()
    
    def update_dropdown_options():
        options = [(cfg["config_name"], cfg) for cfg in GEOMETRY_CONFIGS]
        config_dropdown.options = options if options else [("None", None)]
    update_dropdown_options()
    
    def add_geometry_config(_):
        for key, slider in geometry_sliders.items():
            geometry_config[key] = slider.value
        config_name = config_name_text.value.strip() or f"Geometry_{len(GEOMETRY_CONFIGS)+1}"
        new_config = {"config_name": config_name, "geometry": geometry_config.copy()}
        GEOMETRY_CONFIGS.append(new_config)
        update_dropdown_options()
        save_geometry_configs()
        with output_area:
            clear_output()
            print("Configuration added:")
            print(new_config)
        config_name_text.value = ''
    
    def save_geometry_configs_btn(_):
        path = save_geometry_configs()
        with output_area:
            clear_output()
            print(f"Configurations saved in {path}")
    
    def load_config(_):
        selected = config_dropdown.value
        if selected is None:
            with output_area:
                clear_output()
                print("No configuration selected for loading.")
            return
        for key, value in selected["geometry"].items():
            if key in geometry_sliders:
                geometry_sliders[key].value = value
        config_name_text.value = selected["config_name"]
        with output_area:
            clear_output()
            print(f"Configuration '{selected['config_name']}' loaded.")
    
    def update_config(_):
        selected = config_dropdown.value
        if selected is None:
            with output_area:
                clear_output()
                print("No configuration selected for updating.")
            return
        for key, slider in geometry_sliders.items():
            selected["geometry"][key] = slider.value
        new_name = config_name_text.value.strip() or selected["config_name"]
        selected["config_name"] = new_name
        update_dropdown_options()
        save_geometry_configs()
        with output_area:
            clear_output()
            print(f"Configuration updated: {selected}")
    
    def delete_config(_):
        selected = config_dropdown.value
        if selected is None:
            with output_area:
                clear_output()
                print("No configuration selected for deletion.")
            return
        global GEOMETRY_CONFIGS
        GEOMETRY_CONFIGS = [cfg for cfg in GEOMETRY_CONFIGS if cfg["config_name"] != selected["config_name"]]
        update_dropdown_options()
        save_geometry_configs()
        with output_area:
            clear_output()
            print(f"Configuration '{selected['config_name']}' deleted.")
    
    button_add.on_click(add_geometry_config)
    button_save.on_click(save_geometry_configs_btn)
    button_load.on_click(load_config)
    button_update.on_click(update_config)
    button_delete.on_click(delete_config)
    
    # Widget to display the case (A or B) outside the drawing
    case_label_widget = widgets.Label(value="")
    
    # Drawing zone (zone de travail réduite)
    # Suppression de la bordure ici en retirant "border='1px solid black'"
    fig_output = widgets.Output(layout=widgets.Layout(flex='1', height='700px', padding='10px'))
    
    def draw_structure(_=None):
        with fig_output:
            clear_output(wait=True)
            
            scale = 4  # scaling factor for display only
            
            # Retrieve geometry from sliders
            p            = geometry_sliders["period"].value
            t_super      = geometry_sliders["thick_super"].value
            t_reso       = geometry_sliders["thick_reso"].value
            w_reso       = geometry_sliders["width_reso"].value
            t_gap        = geometry_sliders["thick_gap"].value
            t_mol        = geometry_sliders["thick_mol"].value
            t_func       = geometry_sliders["thick_func"].value
            t_diel       = geometry_sliders["thick_diel"].value
            t_metal      = geometry_sliders["thick_metalliclayer"].value
            t_acc        = geometry_sliders["thick_accroche"].value
            t_sub        = geometry_sliders["thick_sub"].value

            # Displayed thickness for Substrate
            disp_sub   = displayed_thickness(t_sub)
            # For Superstrate, we force it to fill to the top (y = p)
            disp_super = displayed_thickness(t_super)
            
            # Compute vertical positions for lower layers:
            y_sub_bottom = 0
            y_sub_top    = y_sub_bottom + disp_sub
            band_height  = min(0.05 * p, disp_sub, disp_super)
            
            y_acc_bottom = y_sub_top
            disp_acc = 4 * t_acc  # Accroche scaled for display
            y_acc_top    = y_acc_bottom + disp_acc
            
            y_metal_bottom = y_acc_top
            y_metal_top    = y_metal_bottom + t_metal
            
            # Intermediate zone (central column): Gap (scaled) + Nanocube (real)
            central_height = scale * t_gap + t_reso
            y_inter_bottom = y_metal_top
            y_inter_top    = y_inter_bottom + central_height
            
            y_gap_bottom   = y_inter_bottom
            y_gap_top      = y_gap_bottom + scale * t_gap
            y_cube_bottom  = y_gap_top
            y_cube_top     = y_cube_bottom + t_reso
            
            # Lateral columns: order is Dielectric, then Functionalisation, then Molecule
            sum_lat = t_diel + t_func + t_mol
            lat_height_scaled = scale * sum_lat
            lat_filler = max(0, central_height - lat_height_scaled)
            
            central_x     = (p - w_reso) / 2
            lateral_width = (p - w_reso) / 2
            left_x  = 0
            right_x = central_x + w_reso
            
            # For Superstrate, force it to fill exactly up to the top of the cell
            y_super_top = p
            y_super_bottom = y_inter_top
            
            # Lateral filler calculation
            lateral_filler = max(0, y_super_bottom - (y_inter_bottom + lat_height_scaled))
            
            # Determine Case: if (t_diel+t_func+t_mol) < t_gap then Case A, else Case B
            if sum_lat < t_gap:
                case_str = "Case A"
            else:
                case_str = "Case B"
            case_label_widget.value = f"Case: {case_str}"
            
            # Figure with reduced overall size for better widget visibility
            fig, ax = plt.subplots(figsize=(6,6))
            ax.set_title("Schematics (not to scale) - " + case_label_widget.value, fontsize=10, pad=5)
            
            # Draw Substrate and its hatch band
            draw_layer(ax, 0, y_sub_bottom, p, disp_sub, "brown", "Substrate")
            draw_layer(ax, 0, y_sub_bottom, p, band_height, "none", "", hatch='///')
            
            # Draw Accroche and Metallic layers
            draw_layer(ax, 0, y_acc_bottom, p, disp_acc, "gold", "Accroche")
            draw_layer(ax, 0, y_metal_bottom, p, t_metal, "silver", "Metallic")
            
            # Draw Central column: Gap (polymer) then Nanocube
            draw_layer(ax, central_x, y_gap_bottom, w_reso, scale * t_gap, "lightgreen", "Gap (polymer)")
            draw_layer(ax, central_x, y_cube_bottom, w_reso, t_reso, "orange", "Nanocube")
            
            # Draw Lateral columns: Dielectric, Functionalisation, Molecule
            y_curr_left = y_inter_bottom
            thickness_poly_scaled = scale * t_diel
            draw_layer(ax, left_x,  y_curr_left, lateral_width, thickness_poly_scaled, "green", "Dielectric")
            draw_layer(ax, right_x, y_curr_left, lateral_width, thickness_poly_scaled, "green", "Dielectric")
            y_curr_left += thickness_poly_scaled

            thickness_func_scaled = scale * t_func
            draw_layer(ax, left_x,  y_curr_left, lateral_width, thickness_func_scaled, "pink", "Functionalisation")
            draw_layer(ax, right_x, y_curr_left, lateral_width, thickness_func_scaled, "pink", "Functionalisation")
            y_curr_left += thickness_func_scaled

            thickness_mol_scaled = scale * t_mol
            draw_layer(ax, left_x,  y_curr_left, lateral_width, thickness_mol_scaled, "violet", "Molecule")
            draw_layer(ax, right_x, y_curr_left, lateral_width, thickness_mol_scaled, "violet", "Molecule")
            y_curr_left += thickness_mol_scaled

            if lateral_filler > 0:
                draw_layer(ax, left_x,  y_curr_left, lateral_width, lateral_filler, "lightblue", "Superstrate \n (environement)")
                draw_layer(ax, right_x, y_curr_left, lateral_width, lateral_filler, "lightblue", "Superstrate \n (environement)")
            
            # Draw Superstrate in central column
            draw_layer(ax, 0, y_super_bottom, p, p - y_super_bottom, "lightblue", "Superstrate (environement)")
            draw_layer(ax, 0, p - band_height, p, band_height, "none", "", hatch='///')
            
            ax.set_xlim(0, p)
            ax.set_ylim(0, p)
            ax.set_aspect('equal', adjustable='box')
            ax.margins(0)
            
            plt.show()

    # Observe slider changes
    for sld in geometry_sliders.values():
        sld.observe(draw_structure, names='value')
    
    draw_structure()  # Initial draw
    
    config_controls = widgets.VBox([
        config_name_text,
        widgets.HBox([button_add, button_save]),
        widgets.HBox([config_dropdown, button_load, button_update, button_delete]),
        output_area
    ])
    
    left_panel = widgets.VBox(slider_widgets + [config_controls], layout=widgets.Layout(width='600px'))
    right_panel = widgets.VBox([fig_output], layout=widgets.Layout(flex='1'))
    main_ui = widgets.HBox([left_panel, right_panel], layout=widgets.Layout(width='100%'))
    
    return main_ui

geometry_widget = create_geometry_widget()
