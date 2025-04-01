#!/usr/bin/env python3
"""
Module: simulate_and_plot.py

Ce module permet de simuler la reflectance pour différentes combinaisons géométriques et matérielles.
Il fournit également des fonctions pour construire des tableaux récapitulatifs des configurations.

Deux modes de lecture sont supportés pour la construction du tableau :
  - Mode JSON : Lecture du fichier de configuration (geom_mat_combinations.json) (utilisé dans l'onglet Simulation).
  - Mode Simulation Summary : Lecture des fichiers texte de simulation afin d'extraire dynamiquement
    toutes les combinaisons (même lorsqu'un fichier contient plusieurs spectres).

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

def run_simulation_all_combos(lambda_range, wave, n_mod, json_combined_path, geom_mat_combinations_path=None):
    """
    Exécute la simulation de reflectance pour toutes les combinaisons et trace le résultat.
    Retourne un dictionnaire des résultats.
    
    Ce mode est utilisé pour l'onglet Simulation.
    Les configurations sont lues depuis le fichier JSON (ou un fichier alternatif) et un tableau complet
    (pour toutes les configurations) est affiché sous le graphique.
    """
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
    
    for config in all_configs:
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
    
    module_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(module_dir)
    figures_dir = os.path.join(workspace_dir, "Figures")
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    fig_path = os.path.join(figures_dir, f"{re.sub(r'[^A-Za-z0-9_]', '', title)}.png")
    plt.savefig(fig_path, bbox_inches="tight")
    print(f"Figure saved in: {fig_path}")
    
    return results

def parse_simulation_summary(file_path):
    """
    Extrait toutes les combinaisons d'un fichier de simulation texte.
    
    Pour chaque bloc, recherche :
      - "Combo name:" suivi du label,
      - "Geometry:" suivi d'un dictionnaire Python,
      - "Material config (df_config):" suivie d'une liste.
    
    Retourne une liste de dictionnaires avec les clés : 'label', 'geometry', 'material'.
    """
    combos = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Erreur lors de la lecture de {file_path}: {e}")
        return combos
    pattern = re.compile(
        r"Combo name:\s*(?P<label>.*?)\s*\n"
        r"(?:.*?\n)*?"
        r"Geometry:\s*(?P<geometry>\{.*?\})\s*\n"
        r"(?:.*?\n)*?"
        r"Material config \(df_config\):\s*(?P<material>\[.*?\])",
        re.DOTALL
    )
    for match in pattern.finditer(content):
        label = match.group("label").strip().replace(" - ", "\n")
        geom_str = match.group("geometry").strip()
        mat_str = match.group("material").strip()
        try:
            geometry = ast.literal_eval(geom_str)
        except Exception:
            geometry = {}
        try:
            material = ast.literal_eval(mat_str)
        except Exception:
            material = []
        combos.append({"label": label, "geometry": geometry, "material": material})
    return combos

def parse_experimental_data_summary(file_path):
    """
    Extrait un résumé structuré à partir d'un fichier expérimental.

    Le fichier expérimental contient des lignes telles que :
       Environnement : Air / n=1 
       Cube : Argent / n(lambda) / 30 nm
       Gap diélectrique / n = 1.45 / 2 nm
       Fonctionnalisation diélectrique / n = 1.45 / 1 nm
       Couche métallique : Or / n(lambda) / 10 nm
       Substrat : ITO / n(lambda) / 200 nm

    Pour chaque ligne, si le séparateur "/" est présent, on filtre les tokens pour supprimer ceux contenant "n(" ou "n=".
    Pour "Cube", le premier token est utilisé comme Material et le token contenant "nm" (s'il existe) comme Geometry.
    Sinon, toute la valeur est assignée à Geometry.
    
    Retourne un dictionnaire avec les clés "geometry" et "material" contenant les résumés sous forme de chaînes.
    """
    expected_keys = [
        "Environnement",
        "Cube",
        "Gap diélectrique / n =",
        "Fonctionnalisation diélectrique / n =",
        "Couche métallique",
        "Substrat"
    ]
    geom_lines = []
    mat_lines = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                for key in expected_keys:
                    if line.startswith(key):
                        parts = line.split(":", 1)
                        if len(parts) < 2:
                            continue
                        value = parts[1].strip()
                        if "/" in value:
                            tokens = [tok.strip() for tok in value.split("/")]
                            # Supprimer les tokens contenant "n(" ou "n="
                            tokens = [t for t in tokens if not re.search(r"n\(|n=", t)]
                            if key == "Cube":
                                mat_val = tokens[0] if tokens else ""
                                geom_val = ""
                                for t in tokens:
                                    if "nm" in t:
                                        geom_val = t
                                        break
                            else:
                                mat_val = tokens[0] if tokens else ""
                                geom_val = tokens[-1] if tokens and "nm" in tokens[-1] else ""
                        else:
                            mat_val = ""
                            geom_val = value
                        geom_lines.append(f"{key}: {geom_val}".strip())
                        mat_lines.append(f"{key}: {mat_val}".strip())
                        break
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier expérimental {file_path}: {e}")
    return {"geometry": "\n".join(geom_lines), "material": "\n".join(mat_lines)}

def build_summary_table(filter_labels=None, sim_files=None, exp_files=None):
    """
    Construit le tableau récapitulatif adaptatif pour les configurations de simulation et expérimentales.
    
    Paramètres :
      - filter_labels : (optionnel) liste de labels de simulation à inclure (après transformation).
      - sim_files : (optionnel) liste de chemins vers des fichiers de simulation (txt). Si fourni, on extrait dynamiquement.
      - exp_files : (optionnel) liste de chemins vers des fichiers expérimentaux (txt). On extrait les données via parse_experimental_data_summary.
    
    Retourne :
      - config_labels : liste des labels pour chaque spectre (colonne)
      - geometry_summaries : liste des résumés de géométrie correspondants
      - material_summaries : liste des résumés de matériaux correspondants
      - colors : liste des couleurs utilisées pour la mise en forme.
    """
    config_labels = []
    geometry_summaries = []
    material_summaries = []
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Traitement des fichiers de simulation
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
    
    # Traitement des fichiers expérimentaux
    if exp_files is not None and len(exp_files) > 0:
        for fpath in exp_files:
            # On filtre également les expérimentaux si filter_labels est défini
            lbl = os.path.basename(fpath)
            if filter_labels is not None and lbl not in filter_labels:
                continue
            exp_data = parse_experimental_data_summary(fpath)
            config_labels.append(lbl)
            geometry_summaries.append(exp_data.get("geometry", ""))
            material_summaries.append(exp_data.get("material", ""))
    
    return config_labels, geometry_summaries, material_summaries, colors
