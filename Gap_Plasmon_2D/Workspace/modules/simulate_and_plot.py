#!/usr/bin/env python3
"""
Module: simulate_and_plot.py

Ce module simule la réflectance pour toutes les combinaisons géométrie/matériaux 
ou uniquement pour un sous-ensemble choisi et construit un graphique récapitulatif 
(composé du tracé des spectres et d'un tableau). Le résumé de simulation et la figure 
sont sauvegardés via les fonctions utilitaires de Saving_Functions.py.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime

# Importations internes
from simulate_reflectance import simulate_reflectance_all_combos, simulate_reflectance_single
from Saving_Functions import get_material_str_clean, save_simulation_summary, save_figure
# Liste ordonnée des paramètres géométriques avec leur libellé convivial
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

def format_geometry_summary(geometry):
    """Formate un résumé de la géométrie à partir d'un dictionnaire."""
    return "\n".join(f"{disp}: {geometry.get(key, 'NA')}" for key, disp in ordered_params if key in geometry)

def format_material_summary(material_config_list):
    """Formate un résumé des matériaux à partir d'une liste de configurations."""
    lines = []
    for entry in material_config_list:
        key = entry.get("key", "")
        disp_name = next((dname for k, dname in ordered_params if k == key), key)
        mat = entry.get("material", {})
        mtype = mat.get("type", "").strip().lower()
        if mtype == "standard":
            val = mat.get("material", "").strip()
        elif mtype == "custom":
            val = mat.get("expression", "").strip()
        else:
            val = ""
        if val:
            lines.append(f"{disp_name}: {val}")
    return "\n".join(lines)

def build_simulation_figure(simulation_details, lambda_range, title, all_configs):
    """
    Construit une figure composée du tracé des spectres (Rup) et d'un tableau récapitulatif.
    
    Parameters:
        simulation_details (dict): Détails de simulation (pour chaque configuration).
        lambda_range (array-like): Plage de longueurs d'onde.
        title (str): Titre de la figure.
        all_configs (list): Liste des configurations simulées (chaque configuration doit contenir "config_name").
    
    Returns:
        fig: Figure matplotlib construite.
    """
    config_labels = []
    geometry_summaries = []
    material_summaries = []
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for config in all_configs:
        combo_name = config["config_name"]
        config_labels.append(combo_name.replace(" - ", "\n"))
        details = simulation_details.get(combo_name, {})
        geometry_summaries.append(format_geometry_summary(details.get("geometry", {})))
        material_summaries.append(format_material_summary(details.get("material_config", [])))
    
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(2, 1, height_ratios=[3, 2.5])
    
    # Tracé des spectres de reflectance
    ax1 = fig.add_subplot(gs[0])
    for idx, config in enumerate(all_configs):
        combo_name = config["config_name"]
        Rup = simulation_details.get(combo_name, {}).get("Rup")
        if Rup is not None:
            color = colors[idx % len(colors)]
            ax1.plot(lambda_range, Rup, label=config_labels[idx], color=color)
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Reflectance")
    ax1.set_title(title)
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True)
    
    # Construction du tableau récapitulatif
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')
    if config_labels:
        table_data = [geometry_summaries, material_summaries]
        table = ax2.table(cellText=table_data, colLabels=config_labels, rowLabels=["Geometry", "Material"],
                          loc="center", cellLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.auto_set_column_width(col=list(range(len(config_labels))))
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
        # Ajustement dynamique de la hauteur des cellules
        row_heights = {}
        for (row, col), cell in table.get_celld().items():
            if row >= 0:
                nb_lines = cell.get_text().get_text().count('\n') + 1
                row_heights[row] = max(row_heights.get(row, 0), nb_lines)
        for (row, col), cell in table.get_celld().items():
            if row in row_heights:
                cell.set_height(0.04 * row_heights[row])
    else:
        ax2.text(0.5, 0.5, "Aucune configuration simulée", horizontalalignment='center')
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def run_simulation_one_combo(lam_range, wave, n_mod, combo, json_combined_path):
    """
    Simule la réflectance pour une configuration unique.
    
    Utilise les informations de géométrie et de matériau contenues dans combo.
    
    Returns:
        Rup, simulation_details (dict)
    """
    # Extraction de la géométrie et de la configuration matière
    geometry = combo["geometry"]["geometry"]
    material_config_list = combo["material"]["MATERIALS_CONFIG"]
    df_config = pd.DataFrame(material_config_list)
    ri_overrides = combo["material"].get("RI_OVERRIDES", {})
    Rup, Rdown = simulate_reflectance_single(lam_range, geometry, wave, df_config, json_combined_path, n_mod, ri_overrides)
    simulation_details = {
        "geometry": geometry,
        "material_config": df_config.to_dict(orient="records"),
        "ri_overrides": ri_overrides,
        "Rup": Rup,
        "Rdown": Rdown
    }
    
    return Rup, simulation_details



def run_simulation_all_combos(lambda_range, wave, n_mod, json_combined_path, geom_mat_combinations_path=None, selected_configs=None):
    """
    Lance la simulation pour toutes les combinaisons ou pour un sous-ensemble sélectionné.
    
    Si selected_configs est fourni (ensemble de noms), la simulation se fait configuration par configuration
    en appelant run_simulation_one_combo. Sinon, la simulation globale via simulate_reflectance_all_combos est utilisée.
    
    Ensuite, le résumé de simulation et la figure sont sauvegardés via les fonctions de Saving_Functions.
    
    Returns:
        results (dict), simulation_details (dict), all_configs (list), fig (Figure)
    """
    if selected_configs is None:
        # Simulation globale sur toutes les configurations
        results, simulation_details, all_configs = simulate_reflectance_all_combos(lambda_range, wave, n_mod, json_combined_path)
        # Si un fichier de combinaisons est fourni, on le recharge pour obtenir la liste complète
        if geom_mat_combinations_path:
            with open(geom_mat_combinations_path, "r", encoding="utf-8") as f:
                all_configs = json.load(f).get("ALL_COMBINED_CONFIGS", [])
    else:
        # Chargement complet des configurations depuis le fichier de combinaisons ou du fichier json_combined_path
        if geom_mat_combinations_path:
            with open(geom_mat_combinations_path, "r", encoding="utf-8") as f:
                all_configs = json.load(f).get("ALL_COMBINED_CONFIGS", [])
        else:
            with open(json_combined_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            all_configs = data.get("ALL_COMBINED_CONFIGS", [])
        # Filtrer uniquement les configurations dont le nom est dans selected_configs
        all_configs = [cfg for cfg in all_configs if cfg["config_name"] in selected_configs]
        simulation_details = {}
        results = {}
        for config in all_configs:
            config_name = config["config_name"]
            try:
                Rup, details = run_simulation_one_combo(lambda_range, wave, n_mod, config, json_combined_path)
            except Exception as e:
                print(f"Erreur lors de la simulation de {config_name}: {e}")
                continue
            simulation_details[config_name] = details
            results[config_name] = (details["Rup"], details["Rdown"])
    
    # Construction du graphique récapitulatif
    title = "Simulation Reflectance Spectra"
    fig = build_simulation_figure(simulation_details, lambda_range, title, all_configs)
    
    # Sauvegarde du résumé de simulation et de la figure grâce aux fonctions de Saving_Functions.py
    import os
    current_dir = os.getcwd()
    notebooks_path = os.path.join(current_dir, '..', 'notebooks')
    figures_dir = os.path.join(current_dir, '..', 'Figures')
    summary_dir = os.path.join(notebooks_path, "Summary_Simulation")
    
    # Enregistrement du résumé
    save_simulation_summary(simulation_details, lambda_range, wave, n_mod, summary_dir)
    
    # Génération de la chaîne descriptive à partir de la config matérielle (le premier combo)
    material_str_clean = get_material_str_clean(simulation_details)
    
    # Sauvegarde de la figure avec un nom incluant material_str_clean
    save_figure(fig, title, figures_dir, material_str_clean)
    
    return results, simulation_details, all_configs, fig




if __name__ == "__main__":
    # Exemple d'exécution
    lambda_range = list(range(400, 701, 10))
    wave = {"wavelength": 550, "angle": 0, "polarization": 1, "parameter": "exemple"}
    n_mod = 3
    json_combined_path = "chemin/vers/json_combined.json"
    # Simulation globale
    run_simulation_all_combos(lambda_range, wave, n_mod, json_combined_path)
    # Pour simuler uniquement un sous-ensemble, par exemple {"Config 1", "Config 3"}
    # run_simulation_all_combos(lambda_range, wave, n_mod, json_combined_path, selected_configs={"Config 1", "Config 3"})
