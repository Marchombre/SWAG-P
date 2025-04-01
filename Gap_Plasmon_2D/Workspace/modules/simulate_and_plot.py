#!/usr/bin/env python3
"""
Module: simulate_and_plot.py

Ce module permet de simuler la reflectance pour différentes combinaisons géométriques et matérielles.
Il fournit également des fonctions pour construire des tableaux récapitulatifs des configurations.

La fonction build_summary_table(filter_labels=None, sim_files=None, exp_files=None) renvoie :
  - config_labels : liste des labels des configurations (une colonne par spectre)
  - geometry_summaries : liste des résumés de géométrie correspondants
  - material_summaries : liste des résumés de matériaux correspondants
  - colors : liste des couleurs utilisées pour la mise en forme du tableau
"""

import os
import re
import json
import ast
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from datetime import datetime

from simulate_reflectance import simulate_reflectance_all_combos
from Geometry_Material_Config import load_json_config
from Saving_Functions import get_material_str_clean, save_simulation_summary, save_figure
from data_readers import parse_simulation_summary, parse_experimental_data_summary

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

def run_simulation_all_combos(lambda_range, wave, n_mod, json_combined_path, geom_mat_combinations_path=None):
    results = simulate_reflectance_all_combos(lambda_range, wave, n_mod, json_combined_path)
    
    if geom_mat_combinations_path is None:
        geom_mat_data = load_json_config("geom_mat_combinations.json")
    else:
        with open(geom_mat_combinations_path, "r", encoding="utf-8") as f:
            geom_mat_data = json.load(f)
    
    all_configs = geom_mat_data.get("ALL_COMBINED_CONFIGS", [])
    
    config_labels = []
    geometry_summaries = []
    material_summaries = []
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    simulation_details = {}
    for config in all_configs:
        combo_name = config["config_name"]
        geometry_dict = config["geometry"]["geometry"]
        material_config_list = config["material"]["MATERIALS_CONFIG"]
        # Pour simplifier, nous ne transformons pas le df_config en dict ici, on garde la liste
        simulation_details[combo_name] = {
            "geometry": geometry_dict,
            "material_config": material_config_list,
            "ri_overrides": config["material"].get("RI_OVERRIDES", {}),
            # On ajoute Rup et Rdown pour la sauvegarde du résumé
            "Rup": results[combo_name][0],
            "Rdown": results[combo_name][1]
        }
        label = combo_name.replace(" - ", "\n")
        config_labels.append(label)
        geom_lines = []
        for key, disp_name in ordered_params:
            if key in geometry_dict:
                geom_lines.append(f"{disp_name}: {geometry_dict[key]}")
        geometry_summaries.append("\n".join(geom_lines))
        
        mat_lines = []
        for entry in material_config_list:
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
    
    title = "Simulation Reflectance spectra"
    
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 2.5])
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
    
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')
    n_configs = len(all_configs)
    col_labels = config_labels
    row_labels = ["Geometry", "Material"]
    table_data = [geometry_summaries, material_summaries]
    table = ax2.table(cellText=table_data, colLabels=col_labels, rowLabels=row_labels,
                      loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(col=list(range(n_configs)))
    for (row, col), cell in table.get_celld().items():
        if row == -1:
            cell.set_facecolor("#40466e")
            cell.set_text_props(weight='bold', color='white', fontsize=10, ha='center')
        elif col == -1:
            cell.set_facecolor("#40466e")
            cell.set_text_props(weight='bold', color='white', fontsize=10)
        else:
            cell.set_facecolor("whitesmoke")
            cell.set_edgecolor("lightgray")
            cell.set_linewidth(0.5)
    for (row, col), cell in table.get_celld().items():
        if row >= 0 and col >= 0:
            cell.get_text().set_color(colors[col % len(colors)])
    for (row, col), cell in table.get_celld().items():
        if row == -1 and col >= 0:
            cell.set_height(0.07)
    row_heights = {}
    for (row, col), cell in table.get_celld().items():
        if row >= 0:
            txt = cell.get_text().get_text()
            nb_lines = txt.count('\n') + 1
            row_heights[row] = max(row_heights.get(row, 0), nb_lines)
    for (row, col), cell in table.get_celld().items():
        if row in row_heights:
            cell.set_height(0.04 * row_heights[row])
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Sauvegarde des fichiers de résumé et de la figure via Saving_Functions
    
    # Obtenir le répertoire du fichier courant
    current_dir_Sim_and_plot = os.getcwd()
    # Chemin relatif vers le dossier "notebooks" (voisin de "modules")
    notebooks_path = os.path.join(current_dir_Sim_and_plot, '..', 'notebooks')
    figures_dir = os.path.join(current_dir_Sim_and_plot, '..', 'Figures')
    summary_dir = os.path.join(notebooks_path, "Summary_Simulation")
    
    
    save_simulation_summary(simulation_details, lambda_range, wave, n_mod, summary_dir)
    material_str_clean = get_material_str_clean(simulation_details)
    save_figure(fig, title, figures_dir, material_str_clean)
    
    return results


def build_summary_table(filter_labels=None, sim_files=None, exp_files=None):
    config_labels = []
    geometry_summaries = []
    material_summaries = []
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    if sim_files is not None and len(sim_files) > 0:
        for fpath in sim_files:
            sim_configs = parse_simulation_summary(fpath)
            for cfg in sim_configs:
                lbl = cfg.get("label", "Unknown")
                if filter_labels is not None and lbl not in filter_labels:
                    continue
                config_labels.append(lbl)
                geom = cfg.get("geometry", {})
                geom_lines = []
                for key, disp_name in ordered_params:
                    if key in geom:
                        geom_lines.append(f"{disp_name}: {geom[key]}")
                geometry_summaries.append("\n".join(geom_lines))
                mat = cfg.get("material", [])
                mat_lines = []
                if isinstance(mat, list):
                    for entry in mat:
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
    
    if exp_files is not None and len(exp_files) > 0:
        for fpath in exp_files:
            lbl = os.path.basename(fpath)
            if filter_labels is not None and lbl not in filter_labels:
                continue
            exp_data = parse_experimental_data_summary(fpath)
            config_labels.append(lbl)
            geometry_summaries.append(exp_data.get("geometry", ""))
            material_summaries.append(exp_data.get("material", ""))
    
    return config_labels, geometry_summaries, material_summaries, colors
