#!/usr/bin/env python3
# simulate_and_plot.py

import os
import re
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from simulate_reflectance import simulate_reflectance_all_combos
from Geometry_Material_Config import load_json_config

# Liste ordonnée des paramètres géométriques avec noms conviviaux
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

def build_ordered_material_list(material_details):
    roles_order = [
        "perm_env",
        "perm_reso",
        "perm_gap",
        "perm_diel",
        "perm_func",
        "perm_mol",
        "perm_accroche",
        "perm_metalliclayer",
        "perm_sub"
    ]
    material_list = []
    if material_details and "MATERIALS_CONFIG" in material_details:
        for role in roles_order:
            val = ""
            for entry in material_details["MATERIALS_CONFIG"]:
                if entry.get("key", "").strip() == role:
                    mat_info = entry.get("material", {})
                    mtype = mat_info.get("type", "").strip().lower()
                    if mtype == "standard":
                        val = mat_info.get("material", "").strip()
                    elif mtype == "custom":
                        val = mat_info.get("expression", "").strip()
                    break
            if val:
                material_list.append(val)
    return material_list

def run_simulation_all_combos(lambda_range, wave, n_mod, json_combined_path, geom_mat_combinations_path=None):
    results = simulate_reflectance_all_combos(lambda_range, wave, n_mod, json_combined_path)
    
    if geom_mat_combinations_path is None:
        geom_mat_data = load_json_config("geom_mat_combinations.json")
    else:
        with open(geom_mat_combinations_path, "r", encoding="utf-8") as f:
            geom_mat_data = json.load(f)
    
    all_combined_configs = geom_mat_data.get("ALL_COMBINED_CONFIGS", [])
    
    config_labels = []
    geometry_summaries = []
    material_summaries = []
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for config in all_combined_configs:
        full_label = config.get("config_name", "UnknownConfig")
        label = full_label.replace(" - ", "\n")
        config_labels.append(label)
        
        geom_config = config.get("geometry", {}).get("geometry", {})
        geom_lines = []
        for key, disp_name in ordered_params:
            if key in geom_config:
                geom_lines.append(f"{disp_name}: {geom_config[key]}")
        geometry_summaries.append("\n".join(geom_lines))
        
        material_details = config.get("material", {})
        mat_lines = []
        if "MATERIALS_CONFIG" in material_details:
            for entry in material_details["MATERIALS_CONFIG"]:
                key = entry.get("key", "")
                disp_name = key
                for k, dname in ordered_params:
                    if k == key:
                        disp_name = dname
                        break
                mat_info = entry.get("material", {})
                mtype = mat_info.get("type", "").strip().lower()
                if mtype == "standard":
                    val = mat_info.get("material", "").strip()
                elif mtype == "custom":
                    val = mat_info.get("expression", "").strip()
                else:
                    val = ""
                if val:
                    mat_lines.append(f"{disp_name}: {val}")
        material_summaries.append("\n".join(mat_lines))
    
    title = "Simulation Reflectance"
    
    # Figure 
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(2, 1, height_ratios=[3, 2.5])
    
    # --- Graphique ---
    ax1 = fig.add_subplot(gs[0])
    for idx, (combo_name, (Rup, Rdown)) in enumerate(results.items()):
        color = colors[idx % len(colors)]
        lab = config_labels[idx] if idx < len(config_labels) else combo_name
        ax1.plot(lambda_range, Rup, label=lab, color=color)
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Reflectance")
    ax1.set_title(title)
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True)
    
    # --- Tableau récapitulatif ---
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')
    
    n_configs = len(all_combined_configs)
    col_labels = [config_labels[i] for i in range(n_configs)]
    row_labels = ["Geometry", "Material"]
    table_data = [geometry_summaries, material_summaries]
    
    table = ax2.table(
        cellText=table_data,
        colLabels=col_labels,
        rowLabels=row_labels,
        loc="center",
        cellLoc="left"
    )
    
    # Pas de table.scale(...) pour éviter l'uniformisation
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(col=list(range(n_configs)))
    
    # Mise en forme de base
    for (row, col), cell in table.get_celld().items():
        if row == -1:  # Ligne d’en-tête (colLabels)
            cell.set_facecolor("#40466e")
            cell.set_text_props(weight='bold', color='white', fontsize=10, ha='center')
        elif col == -1:  # Colonne d’étiquettes de ligne (rowLabels)
            cell.set_facecolor("#40466e")
            cell.set_text_props(weight='bold', color='white', fontsize=10)
        else:
            cell.set_facecolor("whitesmoke")
            cell.set_edgecolor("lightgray")
            cell.set_linewidth(0.5)
    
    # Colorer le texte des cellules de données
    for (row, col), cell in table.get_celld().items():
        if row >= 0 and col >= 0:
            cell.get_text().set_color(colors[col % len(colors)])
    
    # 1) Fixer la hauteur de la ligne d'en-tête (row = -1, col >= 0) à une valeur fixe
    #    (en évitant (-1, -1) qui n'existe pas)
    for (row, col), cell in table.get_celld().items():
        if row == -1 and col >= 0:
            cell.set_height(0.07)  # ajustez selon vos goûts
    
    # 2) Déterminer la hauteur nécessaire pour chaque ligne de données (row >= 0)
    #    en comptant le nombre de lignes dans le texte.
    row_heights = {}
    for (row, col), cell in table.get_celld().items():
        if row >= 0:  # ignore l'en-tête
            text = cell.get_text().get_text()
            nb_lines = text.count('\n') + 1
            # On mémorise le max de lignes pour la ligne donnée
            row_heights[row] = max(row_heights.get(row, 0), nb_lines)
    
    # 3) Appliquer la même hauteur à toutes les cellules d'une même ligne
    #    pour que la case d'étiquette (col = -1) soit alignée verticalement
    #    avec les cases de données (col >= 0).
    for (row, col), cell in table.get_celld().items():
        if row in row_heights:  # row >= 0
            # 0.04 est un facteur à ajuster selon la taille du texte
            cell.set_height(0.04 * row_heights[row])
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Sauvegarde
    module_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(module_dir)
    figures_dir = os.path.join(workspace_dir, "Figures")
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    fig_path = os.path.join(figures_dir, f"reflectance_{re.sub(r'[^A-Za-z0-9_]', '', title)}.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.show()
    print(f"Figure saved in: {fig_path}")
    
    return results
