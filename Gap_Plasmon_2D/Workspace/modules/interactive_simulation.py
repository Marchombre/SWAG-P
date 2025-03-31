#!/usr/bin/env python3
"""
Module: interactive_simulation.py

Cette application propose deux onglets :
  1. "Simulation" : Permet de lancer une simulation de spectres (avec réglage de λ min, λ max, nb points, nb modes)
     et d'afficher la figure de reflectance. La figure affiche un tableau complet issu du JSON.
     
  2. "Plot" : Expose directement, via une liste déroulante unique, l'ensemble des spectres disponibles (simulés et expérimentaux).
     L'utilisateur peut sélectionner les spectres à tracer, le graphique et le tableau récapitulatif se mettent à jour en fonction.
"""

import os
import glob
import re
import ast
import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

from simulate_and_plot import run_simulation_all_combos, ordered_params, load_json_config
from data_readers import read_all_combos, read_experimental_data

# --- Fonctions utilitaires pour lister les fichiers ---
def list_sim_summary_files(summary_dir):
    pattern = os.path.join(summary_dir, "simulation_summary*.txt")
    files = glob.glob(pattern)
    files.sort()
    return files

def list_exp_data_files(exp_data_dir):
    pattern = os.path.join(exp_data_dir, "Data_structure*.txt")
    files = glob.glob(pattern)
    files.sort()
    return files

# --- Fonctions de parsing (identiques à vos fonctions existantes) ---
def parse_simulation_summary(file_path):
    """
    Extrait toutes les combinaisons d'un fichier de simulation texte.
    
    Pour chaque bloc, recherche :
      - "Combo name:" suivi du label,
      - "Geometry:" suivi d'un dictionnaire Python,
      - "Material config (df_config):" suivi d'une liste.
    
    Utilise une expression régulière non-gourmande avec DOTALL pour capturer même avec des lignes intermédiaires.
    
    Retourne une liste de dictionnaires avec les clés : 'label', 'geometry', 'material'.
    """
    combos = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Erreur de lecture de {file_path}: {e}")
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
        combos.append({
            "label": label,
            "geometry": geometry,
            "material": material
        })
    return combos

def parse_experimental_data_summary(file_path):
    """
    Extrait un résumé structuré à partir d'un fichier expérimental.
    
    Le fichier exp contient des lignes telles que :
       Environnement : Air / n=1 
       Cube : Argent / n(lambda) / 30 nm
       Gap diélectrique / n = 1.45 / 2 nm
       Fonctionnalisation diélectrique / n = 1.45 / 1 nm
       Couche métallique : Or / n(lambda) / 10 nm
       Substrat : ITO / n(lambda) / 200 nm
    
    Pour chaque ligne, si le séparateur "/" est présent, le premier token est pris pour Material
    et le dernier (contenant "nm") pour Geometry. Sinon, toute la valeur est affectée à Geometry.
    
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
                            # Pour "Cube", on prend le premier token pour Material et le dernier pour Geometry
                            if key == "Cube":
                                mat_val = tokens[0]
                                geom_val = tokens[-1] if "nm" in tokens[-1] else ""
                            else:
                                mat_val = tokens[0]
                                geom_val = tokens[-1] if "nm" in tokens[-1] else ""
                        else:
                            mat_val = ""
                            geom_val = value
                        geom_lines.append(f"{key}: {geom_val}".strip())
                        mat_lines.append(f"{key}: {mat_val}".strip())
                        break
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier expérimental {file_path}: {e}")
    return {"geometry": "\n".join(geom_lines), "material": "\n".join(mat_lines)}

# --- Fonction de construction du tableau récapitulatif ---
def build_combined_summary_table(filter_labels=None, sim_files=None, exp_files=None):
    """
    Construit le tableau récapitulatif combiné pour les spectres affichés.
    
    Pour chaque fichier de simulation, les configurations sont extraites via parse_simulation_summary.
    Pour chaque fichier expérimental, le résumé est extrait via parse_experimental_data_summary.
    
    Retourne :
      - config_labels : liste des labels pour chaque spectre (colonne),
      - geometry_summaries : liste des résumés de géométrie,
      - material_summaries : liste des résumés de matériaux,
      - colors : liste des couleurs pour la mise en forme.
    (filter_labels s'applique ici uniquement pour les simulations.)
    """
    config_labels = []
    geometry_summaries = []
    material_summaries = []
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Traitement des fichiers de simulation
    if sim_files is not None and len(sim_files) > 0:
        sim_seen = set()  # pour éviter les doublons
        for fpath in sim_files:
            sim_configs = parse_simulation_summary(fpath)
            for cfg in sim_configs:
                lbl = cfg.get("label", "Unknown")
                if filter_labels is not None and lbl not in filter_labels:
                    continue
                if lbl in sim_seen:
                    continue
                sim_seen.add(lbl)
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
    
    # Traitement des fichiers expérimentaux (toujours ajoutés)
    if exp_files is not None and len(exp_files) > 0:
        exp_seen = set()
        for fpath in exp_files:
            if fpath in exp_seen:
                continue
            exp_seen.add(fpath)
            exp_data = parse_experimental_data_summary(fpath)
            base_lbl = os.path.basename(fpath)
            lbl = base_lbl
            count = 1
            while lbl in config_labels:
                lbl = f"{base_lbl} ({count})"
                count += 1
            config_labels.append(lbl)
            geometry_summaries.append(exp_data.get("geometry", ""))
            material_summaries.append(exp_data.get("material", ""))
    
    return config_labels, geometry_summaries, material_summaries, colors

# --- Fonction pour rassembler l'ensemble des spectres et leurs résumés ---
def get_all_spectra_and_summaries(summary_dir, exp_data_dir):
    """
    Parcourt l'ensemble des fichiers de simulation et expérimentaux pour construire :
      - spectra: dictionnaire {label: (wl, R)} pour chaque spectre,
      - summaries: dictionnaire {label: (geometry_summary, material_summary)}.
    
    Pour les simulations, on utilise read_all_combos et parse_simulation_summary.
    Pour les expérimentaux, on utilise read_experimental_data et parse_experimental_data_summary.
    """
    spectra = {}
    summaries = {}
    # Simulations
    sim_files = list_sim_summary_files(summary_dir)
    for fpath in sim_files:
        combos = read_all_combos(fpath)  # {label: (wl, R)}
        sim_configs = parse_simulation_summary(fpath)
        for combo_label, (wl, R) in combos.items():
            label_key = combo_label.replace(" - ", "\n")
            if label_key in spectra:
                continue
            spectra[label_key] = (wl, R)
            found = False
            for cfg in sim_configs:
                cfg_label = cfg.get("label", "Unknown").replace(" - ", "\n")
                if cfg_label == label_key:
                    geom = cfg.get("geometry", {})
                    geom_lines = []
                    for key, disp_name in ordered_params:
                        if key in geom:
                            geom_lines.append(f"{disp_name}: {geom[key]}")
                    geom_summary = "\n".join(geom_lines)
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
                    mat_summary = "\n".join(mat_lines)
                    summaries[label_key] = (geom_summary, mat_summary)
                    found = True
                    break
            if not found:
                summaries[label_key] = ("", "")
    # Expérimentaux
    exp_files = list_exp_data_files(exp_data_dir)
    for fpath in exp_files:
        data = read_experimental_data(fpath)
        if data:
            base_lbl = os.path.basename(fpath)
            label = base_lbl
            count = 1
            while label in spectra:
                label = f"{base_lbl} ({count})"
                count += 1
            spectra[label] = data
            exp_data = parse_experimental_data_summary(fpath)
            geom_summary = exp_data.get("geometry", "")
            mat_summary = exp_data.get("material", "")
            summaries[label] = (geom_summary, mat_summary)
    return spectra, summaries

# --- Création de l'interface interactive ---
def create_advanced_app(json_combined_path, summary_dir, exp_data_dir):
    """
    Crée l'interface interactive avec deux onglets :
      - Onglet "Simulation" : Lance la simulation via JSON et affiche la figure complète (graphique + tableau complet).
      - Onglet "Plot" : Expose directement, via une liste déroulante unique, l'ensemble des spectres disponibles (simulés et expérimentaux).
         L'utilisateur peut sélectionner les spectres à tracer, et le graphique ainsi que le tableau récapitulatif se mettent à jour en fonction.
         La liste déroulante s'actualise dynamiquement lors du passage à cet onglet.
    """
    # ===============================
    # Onglet 1 : Simulation
    # ===============================
    sim_lambda_min = widgets.FloatText(value=450.0, description="λ min (nm):",
                                       layout=widgets.Layout(width='150px'),
                                       style={'description_width': 'initial'})
    sim_lambda_max = widgets.FloatText(value=1000.0, description="λ max (nm):",
                                       layout=widgets.Layout(width='150px'),
                                       style={'description_width': 'initial'})
    sim_n_points = widgets.IntSlider(value=200, min=50, max=1000, step=10, description="Nb points:",
                                     layout=widgets.Layout(width='200px'),
                                     style={'description_width': 'initial'})
    sim_n_mod = widgets.IntSlider(value=10, min=1, max=100, step=1, description="Modes:",
                                  layout=widgets.Layout(width='200px'),
                                  style={'description_width': 'initial'})
    sim_run_button = widgets.Button(description="Lancer la simulation", button_style="success",
                                    tooltip="Lance la simulation")
    sim_mode_radio = widgets.RadioButtons(options=["Nouvelle figure", "Même figure"],
                                          value="Nouvelle figure",
                                          description="Plot mode:",
                                          style={'description_width': 'initial'})
    
    sim_controls = widgets.VBox([
        widgets.HTML("<h3>Simulation</h3>"),
        widgets.HBox([sim_lambda_min, sim_lambda_max]),
        widgets.HBox([sim_n_points, sim_n_mod]),
        widgets.HBox([sim_mode_radio, sim_run_button])
    ])
    
    sim_output = widgets.Output(layout=widgets.Layout(border="2px solid #ccc", padding="10px", min_height="400px"))
    
    def on_sim_run_clicked(b):
        with sim_output:
            sim_output.clear_output(wait=True)
            spinner = widgets.HTML(
                "<div style='text-align: center;'><img src='https://i.gifer.com/ZZ5H.gif' width='50px'/><br><em>Simulation en cours...</em></div>"
            )
            display(spinner)
            lam_min = sim_lambda_min.value
            lam_max = sim_lambda_max.value
            n_points = sim_n_points.value
            n_mod = sim_n_mod.value
            wave = {"angle": 0, "polarization": 1}
            lam_range = np.linspace(lam_min, lam_max, n_points)
            old_show = plt.show
            plt.show = lambda *args, **kwargs: None
            run_simulation_all_combos(lam_range, wave, n_mod, json_combined_path)
            plt.show = old_show
            if plt.get_fignums():
                fig = plt.gcf()
            else:
                fig = plt.figure()
            if not fig.get_axes():
                fig.add_subplot(111)
            sim_output.clear_output(wait=True)
            display(fig)
            plt.close(fig)
    
    sim_run_button.on_click(on_sim_run_clicked)
    sim_tab = widgets.VBox([sim_controls, sim_output])
    
    # ===============================
    # Onglet 2 : Plot (liste déroulante unique)
    # ===============================
    # Widget unique pour sélectionner les spectres disponibles
    spectra_select = widgets.SelectMultiple(
        options=[], 
        description="Spectres disponibles:",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='80%', height='150px')
    )
    
    # Bouton pour tracer les spectres sélectionnés
    plot_button = widgets.Button(
        description="Tracer", button_style="info",
        tooltip="Tracer les spectres sélectionnés"
    )
    
    plot_output = widgets.Output(layout=widgets.Layout(border="2px solid #ccc", padding="10px", min_height="400px"))
    
    # Dictionnaires globaux pour conserver les données extraites
    plotted_lines = {}   # {label: (wl, R)}
    summaries = {}       # {label: (geom_summary, mat_summary)}
    
    # Fonction de mise à jour des spectres disponibles
    def update_spectra():
        all_spectra, all_summaries = get_all_spectra_and_summaries(summary_dir, exp_data_dir)
        spectra_select.options = list(all_spectra.keys())
        nonlocal plotted_lines, summaries
        plotted_lines = all_spectra
        summaries = all_summaries
    
    # Au démarrage, on met à jour la liste (mais elle sera rafraîchie dynamiquement lors du changement d'onglet)
    update_spectra()
    
    def on_plot_button_clicked(b):
        selected_labels = list(spectra_select.value)
        if not selected_labels:
            selected_labels = list(plotted_lines.keys())
            spectra_select.value = tuple(selected_labels)
        
        if len(plotted_lines) > 0:
            fig = plt.figure(figsize=(10, 10))
            gs = fig.add_gridspec(2, 1, height_ratios=[3, 3])
            ax_plot = fig.add_subplot(gs[0])
            ax_table = fig.add_subplot(gs[1])
        else:
            fig, ax_plot = plt.subplots(figsize=(10, 6))
            ax_table = None
        
        for label in selected_labels:
            x, y = plotted_lines[label]
            ax_plot.plot(x, y, label=label)
        ax_plot.set_xlabel("Wavelength (nm)")
        ax_plot.set_ylabel("Reflectance")
        ax_plot.set_title("Spectres combinés")
        ax_plot.legend()
        ax_plot.grid(True)
        
        if ax_table is not None:
            config_labels = []
            geom_summaries = []
            mat_summaries = []
            for label in selected_labels:
                config_labels.append(label)
                geom, mat = summaries.get(label, ("", ""))
                geom_summaries.append(geom)
                mat_summaries.append(mat)
            ax_table.axis("off")
            if config_labels:
                n_configs = len(config_labels)
                row_labels = ["Geometry", "Material"]
                table_data = [geom_summaries, mat_summaries]
                table = ax_table.table(
                    cellText=table_data,
                    colLabels=config_labels,
                    rowLabels=row_labels,
                    loc="center",
                    cellLoc="left"
                )
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.auto_set_column_width(col=list(range(n_configs)))
                for (row, col), cell in table.get_celld().items():
                    if row == -1:
                        cell.set_facecolor("#40466e")
                        cell.set_text_props(weight="bold", color="white", fontsize=10, ha="center")
                    elif col == -1:
                        cell.set_facecolor("#40466e")
                        cell.set_text_props(weight="bold", color="white", fontsize=10)
                    else:
                        cell.set_facecolor("whitesmoke")
                        cell.set_edgecolor("lightgray")
                        cell.set_linewidth(0.5)
                for (row, col), cell in table.get_celld().items():
                    if row >= 0 and col >= 0:
                        cell.get_text().set_color(plt.rcParams['axes.prop_cycle'].by_key()['color'][col % len(plt.rcParams['axes.prop_cycle'].by_key()['color'])])
                for (row, col), cell in table.get_celld().items():
                    if row == -1 and col >= 0:
                        cell.set_height(0.07)
                row_heights = {}
                for (row, col), cell in table.get_celld().items():
                    if row >= 0:
                        txt = cell.get_text().get_text()
                        nb_lines = txt.count("\n") + 1
                        row_heights[row] = max(row_heights.get(row, 0), nb_lines)
                for (row, col), cell in table.get_celld().items():
                    if row in row_heights:
                        cell.set_height(0.04 * row_heights[row])
                fig.tight_layout(rect=[0, 0, 1, 0.95])
        
        with plot_output:
            plot_output.clear_output(wait=True)
            display(fig)
            plt.close(fig)
    
    plot_button.on_click(on_plot_button_clicked)
    
    plot_controls = widgets.VBox([
        widgets.HTML("<h3>Plot</h3>"),
        spectra_select,
        plot_button
    ])
    
    plot_tab = widgets.VBox([plot_controls, plot_output])
    
    tabs = widgets.Tab()
    tabs.children = [sim_tab, plot_tab]
    tabs.set_title(0, "Simulation")
    tabs.set_title(1, "Plot")
    
    # Actualisation dynamique de la liste déroulante lorsque l'onglet Plot est sélectionné
    def on_tab_change(change):
        if change['new'] == 1:  # Si l'onglet Plot devient actif
            update_spectra()
    
    tabs.observe(on_tab_change, names='selected_index')
    
    app_layout = widgets.VBox([tabs])
    return app_layout

# Exemple d'utilisation dans un notebook :
# from interactive_simulation import create_advanced_app
# app = create_advanced_app(json_combined_path, summary_dir, exp_data_dir)
# display(app)
