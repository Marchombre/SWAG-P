#!/usr/bin/env python3
"""
Module: interactive_simulation.py

Cette application propose trois onglets :
  1. "Simulation" : Permet de lancer une simulation de spectres (avec réglage de λ min, λ max, nb points, nb modes)
     et d'afficher la figure de reflectance. La figure affiche un tableau complet issu du JSON.
     
  2. "Plot" : Expose directement, via une liste déroulante unique, l'ensemble des spectres disponibles (simulés et expérimentaux).
     L'utilisateur peut sélectionner les spectres à tracer, et le graphique ainsi que le tableau récapitulatif se mettent à jour en fonction.
     
  3. "Difference" : Permet de sélectionner deux spectres (de référence et à comparer) et d'afficher la différence
     (spectre cible moins spectre de référence) dans un affichage moderne et sophistiqué.
"""

import os
import glob
import re
import ast
import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from datetime import datetime

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

# --- Gestion des labels pour les fichiers de simulation ---
def get_simulation_label(base_label, file_path, label_to_tag):
    """
    Calcule un nouveau label pour un spectre issu d'un fichier de simulation.
    
    Le nom du fichier est supposé suivre le format:
      simulation_summary_RCWA_<version>_<material_str_clean>.txt
      
    Seule la partie <version> (le premier mot après 'simulation_summary_RCWA_') est utilisée pour
    distinguer les spectres issus de fichiers différents ayant le même base_label.
    
    Si plusieurs fichiers ont la même version, un indice numérique supplémentaire est ajouté.
    """
    fname = os.path.basename(file_path)
    tag = os.path.splitext(fname)[0]  # Retire l'extension
    prefix = "simulation_summary_RCWA_"
    version = ""
    if tag.startswith(prefix):
        remainder = tag[len(prefix):]  # Par exemple "V1_materialStrClean"
        parts = remainder.split("_", 1)  # On ne conserve que le premier élément
        if parts:
            version = parts[0]  # Par exemple "V1" ou "V2"
    # On utilise label_to_tag comme dictionnaire de suivi par base_label et par version
    if base_label not in label_to_tag:
        label_to_tag[base_label] = {}
    if version not in label_to_tag[base_label]:
        label_to_tag[base_label][version] = 1
        return f"{base_label} ({version})" if version else base_label
    else:
        label_to_tag[base_label][version] += 1
        count = label_to_tag[base_label][version]
        return f"{base_label} ({version} {count})"


# --- Fonction de construction du tableau récapitulatif ---
def build_combined_summary_table(filter_labels=None, sim_files=None, exp_files=None):
    config_labels = []
    geometry_summaries = []
    material_summaries = []
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    label_to_tag = {}
    # Traitement des fichiers de simulation
    if sim_files is not None and len(sim_files) > 0:
        for fpath in sim_files:
            sim_configs = parse_simulation_summary(fpath)
            for cfg in sim_configs:
                base_label = cfg.get("label", "Unknown")
                if filter_labels is not None and base_label not in filter_labels:
                    continue
                new_label = get_simulation_label(base_label, fpath, label_to_tag)
                config_labels.append(new_label)
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
    # Traitement des fichiers expérimentaux (sans indice de provenance)
    if exp_files is not None and len(exp_files) > 0:
        for fpath in exp_files:
            exp_data = parse_experimental_data_summary(fpath)
            base_lbl = os.path.basename(fpath)
            config_labels.append(base_lbl)
            geometry_summaries.append(exp_data.get("geometry", ""))
            material_summaries.append(exp_data.get("material", ""))
    return config_labels, geometry_summaries, material_summaries, colors

# --- Fonction pour rassembler l'ensemble des spectres et leurs résumés ---
def get_all_spectra_and_summaries(summary_dir, exp_data_dir):
    spectra = {}
    summaries = {}
    label_to_tag = {}
    # Simulations
    sim_files = list_sim_summary_files(summary_dir)
    for fpath in sim_files:
        combos = read_all_combos(fpath)  # {label: (wl, R)}
        sim_configs = parse_simulation_summary(fpath)
        for combo_label, (wl, R) in combos.items():
            base_label = combo_label.replace(" - ", "\n")
            if base_label in spectra:
                new_label = get_simulation_label(base_label, fpath, label_to_tag)
            else:
                fname = os.path.basename(fpath)
                tag = os.path.splitext(fname)[0]
                prefix = "simulation_summary_RCWA_"
                if tag.startswith(prefix):
                    remainder = tag[len(prefix):]
                    parts = remainder.split("_", 1)
                    tag = parts[0] if parts else ""
                label_to_tag[base_label] = {tag: 1}
                new_label = f"{base_label} ({tag})" if tag else base_label
            spectra[new_label] = (wl, R)
            found = False
            for cfg in sim_configs:
                cfg_label = cfg.get("label", "Unknown").replace(" - ", "\n")
                if cfg_label == base_label:
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
                    summaries[new_label] = (geom_summary, mat_summary)
                    found = True
                    break
            if not found:
                summaries[new_label] = ("", "")
    # Expérimentaux
    exp_files = list_exp_data_files(exp_data_dir)
    for fpath in exp_files:
        data = read_experimental_data(fpath)
        if data:
            base_lbl = os.path.basename(fpath)
            spectra[base_lbl] = data
            exp_data = parse_experimental_data_summary(fpath)
            geom_summary = exp_data.get("geometry", "")
            mat_summary = exp_data.get("material", "")
            summaries[base_lbl] = (geom_summary, mat_summary)
    return spectra, summaries

# --- Création de l'interface interactive ---
def create_advanced_app(json_combined_path, summary_dir, exp_data_dir):
    """
    Crée l'interface interactive avec trois onglets :
      - Onglet "Simulation" : Lance la simulation via JSON et affiche la figure complète (graphique + tableau complet).
      - Onglet "Plot" : Expose directement, via une liste déroulante unique, l'ensemble des spectres disponibles (simulés et expérimentaux).
         L'utilisateur peut sélectionner les spectres à tracer, et le graphique ainsi que le tableau récapitulatif se mettent à jour en fonction.
      - Onglet "Difference" : Permet de sélectionner deux spectres (de référence et à comparer) et d'afficher la différence
         (spectre cible moins spectre de référence) dans un affichage moderne et sophistiqué.
         La liste déroulante s'actualise dynamiquement lors du passage à l'onglet.
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
    sim_n_points = widgets.IntText(value=200, description="Nb points:",
                                layout=widgets.Layout(width='200px'),
                                style={'description_width': 'initial'})
    sim_n_mod = widgets.IntText(value=10, description="Modes:",
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
    spectra_select = widgets.SelectMultiple(
        options=[], 
        description="Spectres disponibles:",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='80%', height='150px')
    )
    
    plot_button = widgets.Button(
        description="Tracer", button_style="info",
        tooltip="Tracer les spectres sélectionnés"
    )
    
    plot_output = widgets.Output(layout=widgets.Layout(border="2px solid #ccc", padding="10px", min_height="400px"))
    
    plotted_lines = {}   # {label: (wl, R)}
    summaries = {}       # {label: (geom_summary, mat_summary)}
    
    def update_spectra():
        all_spectra, all_summaries = get_all_spectra_and_summaries(summary_dir, exp_data_dir)
        spectra_select.options = list(all_spectra.keys())
        nonlocal plotted_lines, summaries
        plotted_lines = all_spectra
        summaries = all_summaries
    
    update_spectra()
    
    def on_plot_button_clicked(b):
        selected_labels = list(spectra_select.value)
        if not selected_labels:
            selected_labels = list(plotted_lines.keys())
            spectra_select.value = tuple(selected_labels)
        
        fig = plt.figure(figsize=(10, 10))
        ax_plot = fig.add_axes([0.1, 0.55, 0.8, 0.4])
        ax_table = fig.add_axes([0.05, 0.05, 0.9, 0.4])
        
        for label in selected_labels:
            x, y = plotted_lines[label]
            ax_plot.plot(x, y, label=label)
        ax_plot.set_xlabel("Wavelength (nm)")
        ax_plot.set_ylabel("Reflectance")
        ax_plot.set_title("Spectres combinés")
        ax_plot.legend()
        ax_plot.grid(True)
        
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
            fontsize = 10 if n_configs <= 5 else max(10 - (n_configs - 5), 4)
            table = ax_table.table(
                cellText=[geom_summaries, mat_summaries],
                colLabels=config_labels,
                rowLabels=["Geometry", "Material"],
                loc="center",
                cellLoc="left"
            )
            table.auto_set_font_size(False)
            table.set_fontsize(fontsize)
            table.auto_set_column_width(col=list(range(len(config_labels))))
            for (row, col), cell in table.get_celld().items():
                if row == -1:
                    cell.set_facecolor("#40466e")
                    cell.set_text_props(weight="bold", color="white", fontsize=fontsize, ha="center")
                elif col == -1:
                    cell.set_facecolor("#40466e")
                    cell.set_text_props(weight="bold", color="white", fontsize=fontsize)
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
    
    # ===============================
    # Onglet 3 : Difference
    # ===============================
    diff_ref_dropdown = widgets.Dropdown(
        options=[],
        description="Référence:",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='80%')
    )
    diff_target_dropdown = widgets.Dropdown(
        options=[],
        description="À comparer:",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='80%')
    )
    diff_button = widgets.Button(
        description="Tracer la différence", button_style="warning",
        tooltip="Tracer la différence entre les deux spectres sélectionnés"
    )
    diff_output = widgets.Output(layout=widgets.Layout(border="2px solid #ccc", padding="10px", min_height="400px"))
    
    def update_diff_options():
        spectra_all, _ = get_all_spectra_and_summaries(summary_dir, exp_data_dir)
        options = list(spectra_all.keys())
        diff_ref_dropdown.options = options
        diff_target_dropdown.options = options
    update_diff_options()
    
    def on_diff_button_clicked(b):
        ref_label = diff_ref_dropdown.value
        target_label = diff_target_dropdown.value
        if not ref_label or not target_label:
            with diff_output:
                diff_output.clear_output()
                print("Veuillez sélectionner les deux spectres.")
            return
        spectra_all, _ = get_all_spectra_and_summaries(summary_dir, exp_data_dir)
        ref_data = spectra_all.get(ref_label)
        target_data = spectra_all.get(target_label)
        if ref_data is None or target_data is None:
            with diff_output:
                diff_output.clear_output()
                print("Données introuvables pour l'un des spectres.")
            return
        wl1, R1 = ref_data
        wl2, R2 = target_data
        if np.array_equal(wl1, wl2):
            common_wl = wl1
            diff_R = np.array(R2) - np.array(R1)
        else:
            common_wl = wl1
            diff_R = np.array(np.interp(wl1, wl2, R2)) - np.array(R1)
        
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_axes([0.1, 0.15, 0.8, 0.75])
        ax.plot(common_wl, diff_R, label=f"Diff: {target_label} - {ref_label}", color="blue")
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Différence de reflectance")
        ax.set_title(f"Différence: {target_label} - {ref_label}")
        ax.legend()
        ax.grid(True)
        
        with diff_output:
            diff_output.clear_output(wait=True)
            display(fig)
            plt.close(fig)
    
    diff_button.on_click(on_diff_button_clicked)
    
    diff_controls = widgets.VBox([
        widgets.HTML("<h3>Difference</h3>"),
        diff_ref_dropdown,
        diff_target_dropdown,
        diff_button
    ])
    
    diff_tab = widgets.VBox([diff_controls, diff_output])
    
    tabs = widgets.Tab()
    tabs.children = [sim_tab, plot_tab, diff_tab]
    tabs.set_title(0, "Simulation")
    tabs.set_title(1, "Plot")
    tabs.set_title(2, "Difference")
    
    def on_tab_change(change):
        if change['new'] == 1:
            update_spectra()
        elif change['new'] == 2:
            update_diff_options()
    
    tabs.observe(on_tab_change, names='selected_index')
    
    app_layout = widgets.VBox([tabs])
    return app_layout

# Exemple d'utilisation dans un notebook :
# from interactive_simulation import create_advanced_app
# app = create_advanced_app(json_combined_path, summary_dir, exp_data_dir)
# display(app)
