#!/usr/bin/env python3
"""
Module: interactive_simulation.py

Cette application propose trois onglets :
  1. "Simulation" : Permet de lancer une simulation de spectres, d'afficher la convergence et de télécharger l'image.
     L'onglet Simulation est découpé en trois parties :
       - En haut à gauche : les sélecteurs relatifs à la simulation (fichiers, paramètres et configurations).
       - En haut à droite : les sélecteurs et l'affichage de la convergence, avec des champs répartis sur deux lignes et le bouton associé.
       - En bas : la figure de simulation est affichée et, plus bas, un tableau récapitulatif (avec les mêmes couleurs que le tracé).
  2. "Plot" : Liste et trace les spectres disponibles (simulés et expérimentaux) avec un tableau récapitulatif.
  3. "Difference" : Compare deux spectres et affiche leur différence.
"""

import os
import glob
import re
import ast
import io, base64
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import HTML, display, clear_output
from datetime import datetime

# Construction des chemins
module_dir       = os.path.dirname(os.path.abspath(__file__))
workspace_dir    = os.path.dirname(module_dir)
notebooks_dir    = os.path.join(workspace_dir, "notebooks")
summary_dir      = os.path.join(notebooks_dir, "Summary_Simulation")
exp_data_dir     = os.path.join(notebooks_dir, "Experimental_Data")
configurations_dir = os.path.join(workspace_dir, "CONFIGURATIONS")
data_dir         = os.path.join(workspace_dir, "data")
json_combined_path = os.path.join(data_dir, "combined_materials.json")

# Importations internes
from simulate_and_plot import run_simulation_all_combos, ordered_params, run_simulation_one_combo
from data_readers import (
    read_all_combos,
    read_experimental_data,
    parse_simulation_summary,
    parse_experimental_data_summary,
    list_sim_summary_files,
    list_exp_data_files,
    get_simulation_label,
    get_all_spectra_and_summaries
)
from convergence_analysis import compute_convergence, create_multi_convergence_widget
from simulate_reflectance import simulate_reflectance_single

# --- Téléchargement de la figure ---
def create_download_link(fig, filename="figure.png"):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    href = f'<a download="{filename}" href="data:image/png;base64,{b64}" target="_blank">Télécharger l\'image</a>'
    return HTML(href)


# --- Interface interactive ---
def create_advanced_app(json_combined_path, summary_dir, exp_data_dir):
    """
    Crée l'interface interactive avec trois onglets :
      - Onglet "Simulation" : La partie haute est divisée en deux colonnes (à gauche : contrôles de simulation ; à droite : contrôles de convergence).
        La partie basse affiche la figure de simulation et, sous celle‑ci, un tableau récapitulatif.
      - Onglet "Plot" : Affiche les spectres disponibles et leur tableau récapitulatif.
      - Onglet "Difference" : Permet de comparer deux spectres.
    """
    # Lecture des configurations
    combos_file = os.path.join(configurations_dir, "geom_mat_combinations.json")
    if os.path.exists(combos_file):
        with open(combos_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_configs = data.get("ALL_COMBINED_CONFIGS", [])
    else:
        all_configs = []
    
    # --- Partie Simulation - Haut (deux colonnes) ---
    # Gauche : Contrôles de simulation
    sim_lambda_min = widgets.FloatText(value=450.0, description="λ min (nm):",
                                        layout=widgets.Layout(width='150px'),
                                        style={'description_width': 'initial'})
    sim_lambda_max = widgets.FloatText(value=1000.0, description="λ max (nm):",
                                        layout=widgets.Layout(width='150px'),
                                        style={'description_width': 'initial'})
    sim_n_points = widgets.IntText(value=200, description="Points:",
                                    layout=widgets.Layout(width='200px'),
                                    style={'description_width': 'initial'})
    sim_n_mod = widgets.IntText(value=70, description="Modes:",
                                 layout=widgets.Layout(width='200px'),
                                 style={'description_width': 'initial'})
    sim_run_button = widgets.Button(description="Run simulation", button_style="success",
                                    tooltip="Lancer la simulation")
    
    # Empêche les valeurs négatives pour les épaisseurs
    def validate_positive(change):
        if change['new'] < 0:
            change['owner'].value = 0  
            
    # On applique la validation pour chacun des widgets.
    sim_lambda_min.observe(validate_positive, names='value')
    sim_lambda_max.observe(validate_positive, names='value')
    sim_n_points.observe(validate_positive, names='value')
    sim_n_mod.observe(validate_positive, names='value')                      
            
    
    sim_files_dropdown = widgets.Dropdown(
        options=list_sim_summary_files(summary_dir),
        description="Simulation files:",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='500px')
    )
    sim_refresh_button = widgets.Button(description="Refresh", button_style="info",
                                        tooltip="Rafraîchir les fichiers de simulation")
    sim_refresh_button.on_click(lambda b: sim_files_dropdown.set_trait("options", list_sim_summary_files(summary_dir)))
    
    sim_download_button = widgets.Button(description="Download selected file", button_style="info",
                                         tooltip="Télécharger le fichier sim sélectionné")
    sim_download_output = widgets.Output()
    def create_file_download_link(file_path, link_text=None):
        with open(file_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode("utf-8")
        if link_text is None:
            link_text = os.path.basename(file_path)
        return HTML(f'<a download="{os.path.basename(file_path)}" href="data:application/octet-stream;base64,{b64}" target="_blank">{link_text}</a>')
    def on_download_clicked(b):
        with sim_download_output:
            sim_download_output.clear_output(wait=True)
            selected_file = sim_files_dropdown.value
            if selected_file:
                dl_link = create_file_download_link(selected_file, link_text=f"Download {os.path.basename(selected_file)}")
                display(dl_link)
            else:
                print("Aucun fichier sélectionné.")
    sim_download_button.on_click(on_download_clicked)
    # Définition d'un conteneur horizontal pour regrouper plusieurs widgets sur la même ligne.
    # Ici, nous plaçons le dropdown des fichiers, le bouton de rafraîchissement,
    # le bouton de téléchargement et la zone de sortie pour le téléchargement.
    # Vous pouvez ajuster les marges et l'alignement pour positionner ces éléments précisément.
    sim_files_box = widgets.HBox(
        [
            sim_files_dropdown,    # Liste déroulante pour sélectionner les fichiers
            sim_download_output    # Zone d'affichage des liens de téléchargement (peut être cachée ou affichée)
        ],
        layout=widgets.Layout(
            width='100%',                     # Occupe toute la largeur disponible
            justify_content='flex-start',     # Alignement horizontal (flex-start, center, flex-end, space-between, ...)
            margin='10px 0px 10px 0px'          # Marges (haut, droite, bas, gauche). Ici, 10px en haut et en bas.
        )
    )
    
    Download_Refresh_box = widgets.HBox(    
        [
            sim_refresh_button,    # Bouton pour rafraîchir la liste des fichiers
            sim_download_button,   # Bouton pour télécharger le fichier sélectionné
        ], 
        layout=widgets.Layout(
            width='50%',                     # Occupe 1/2 de l'espace disponible
            justify_content='flex-start',     # Alignement horizontal (flex-start, center, flex-end, space-between, ...)
            margin='10px 0px 10px 0px'          # Marges (haut, droite, bas, gauche). Ici, 10px en haut et en bas.
        )
    )                               

    # Création d'un widget de sélection multiple pour choisir une ou plusieurs configurations de simulation.
    # La largeur est fixée à 350px.
    sim_config_selector = widgets.SelectMultiple(
        options=[(cfg["config_name"], cfg) for cfg in all_configs],
        description="Config simulation:",
        layout=widgets.Layout(width='500px', height='150px'),
        style={'description_width': 'initial'}  # Permet de conserver la largeur par défaut pour la description
    )

    # Regroupement vertical des contrôles de simulation dans un conteneur VBox.
    # Chaque ligne est soit un HBox (pour afficher plusieurs éléments horizontalement),
    # soit un widget unique.
    sim_controls = widgets.VBox(
        [
            widgets.HTML("<h3>Simulation - Paramètres</h3>"),
            sim_files_box,  # La première ligne : fichier(s) sélection et boutons associés.
            widgets.HBox(
                [Download_Refresh_box],
                layout=widgets.Layout(justify_content='space-around', margin='5px 0px')
            ),  # Ligne pour les boutons de téléchargement et de rafraîchissement                
            widgets.HBox(
                [sim_lambda_min, sim_lambda_max],
                layout=widgets.Layout(justify_content='space-around', margin='5px 0px')
            ),  # Ligne pour les valeurs lambda min et max
            widgets.HBox(
                [sim_n_points, sim_n_mod],
                layout=widgets.Layout(justify_content='space-around', margin='5px 0px')
            ),  # Ligne pour le nombre de points et le nombre de modes
            # Pour le bouton Run Simulation, vous pouvez le centrer ou l'aligner à gauche/droite selon vos préférences.
            widgets.HBox(
                [sim_run_button],
                layout=widgets.Layout(justify_content='center', margin='5px 0px')
            ),  # Ligne contenant le bouton de lancement de la simulation.
            sim_config_selector                              # Affichage du sélecteur multiple pour les configurations de simulation.
        ],
        layout=widgets.Layout(
            padding='10px',         # Espace interne autour de ces éléments
            border='solid 1px lightgray'
        )
    )

    
    # Droite : Contrôles et tracé de convergence
    conv_widget = create_multi_convergence_widget(json_combined_path, all_configs)
    
    top_section = widgets.HBox([sim_controls, conv_widget])
    
    # --- Partie Simulation - Bas : Zone d'affichage de la simulation ---
    sim_output = widgets.Output(
        layout=widgets.Layout(
            border="2px solid #ccc",
            padding="10px",
            min_height="400px",
            margin='40px 0 0 0'  # 20px de marge en haut, 0 en droite, 0 en bas, 0 en gauche
        )
    )
    
    
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
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            # Création d'une figure globale qui contiendra le graphique et le tableau
            fig = plt.figure(figsize=(12, 10))

            # Ajout d'un axe pour le tracé du graphique de simulation
            ax_plot = fig.add_axes([0.1, 0.50, 0.9, 0.49])
            # Ajout d'un axe pour le tableau récapitulatif
            ax_table = fig.add_axes([0.1, 0.10, 0.9, 0.40])

            config_labels = []
            geom_summaries = []
            mat_summaries = []

            for idx, cfg in enumerate(sim_config_selector.value):
                Rup, _ = run_simulation_one_combo(lam_range, wave, n_mod, cfg, json_combined_path)
                ax_plot.plot(lam_range, Rup, label=cfg["config_name"], color=colors[idx % len(colors)])
                config_labels.append(cfg["config_name"])
                # Construction du résumé de géométrie
                geom = cfg.get("geometry", {}).get("geometry", {})
                geom_lines = []
                for key, disp_name in ordered_params:
                    if key in geom:
                        geom_lines.append(f"{disp_name}: {geom[key]}")
                geom_summaries.append("\n".join(geom_lines))
                # Construction du résumé matière
                mat_list = cfg.get("material", {}).get("MATERIALS_CONFIG", [])
                mat_lines = []
                if isinstance(mat_list, list):
                    for entry in mat_list:
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
                mat_summaries.append("\n".join(mat_lines))

            ax_plot.set_xlabel("Wavelength (nm)")
            ax_plot.set_ylabel("Reflectance")
            ax_plot.set_title("Simulation")
            ax_plot.legend()
            ax_plot.grid(True)
            # Désactive l'affichage des axes sur l'axe qui contiendra le tableau
            ax_table.axis("off")

            # Modification pour insérer un saut de ligne dans les noms de spectres.
            # Pour chaque nom, dès qu'on trouve "Mat_", on le remplace par "\nMat_"
            config_labels = [label.replace("Mat_", "\nMat_") for label in config_labels]

            if config_labels:
                n_configs = len(config_labels)
                # Détermine la taille de la police
                fontsize = 8 if n_configs <= 5 else max(8 - (n_configs - 5), 3)
                # Création du tableau dans l'axe ax_table
                table = ax_table.table(
                    cellText=[geom_summaries, mat_summaries],
                    colLabels=config_labels,
                    rowLabels=["Geometry", "Material"],
                    loc="center",
                    cellLoc="left"
                )
                # Désactive l'ajustement automatique de la taille de police et définit la taille manuellement
                table.auto_set_font_size(False)
                table.set_fontsize(fontsize)
                # Ajuste la largeur des colonnes en fonction du contenu
                table.auto_set_column_width(col=list(range(len(config_labels))))
                
                # Personnalisation des cellules d'en-tête
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
                
                # Applique les couleurs du cycle aux textes des cellules du corps
                for (row, col), cell in table.get_celld().items():
                    if row >= 0 and col >= 0:  # Exclut les en-têtes
                        cell.get_text().set_color(colors[col % len(colors)])
                
                # Mise à jour dynamique de la hauteur des cellules
                row_heights = {}
                # Parcourt chaque cellule du corps pour déterminer le nombre de lignes de texte
                for (row, col), cell in table.get_celld().items():
                    if row >= 0:
                        txt = cell.get_text().get_text()
                        nb_lines = txt.count("\n") + 1  # Le nombre de lignes est le nombre de sauts de ligne + 1
                        row_heights[row] = max(row_heights.get(row, 0), nb_lines)
                # Applique la hauteur à chaque cellule en fonction du nombre de lignes détectées
                for (row, col), cell in table.get_celld().items():
                    if row in row_heights:
                        cell.set_height(0.04 * row_heights[row])
            
            ax_table.figure.canvas.draw_idle()  # Mise à jour immédiate du canvas
            
            sim_output.clear_output(wait=True)
            display(fig)
            download_link = create_download_link(fig, filename=f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            display(download_link)
            plt.close(fig)

            
            
    sim_run_button.on_click(on_sim_run_clicked)
    
    # Assemblage de l'onglet Simulation : partie haute (deux colonnes) + partie basse (figure simulation et tableau)
    sim_tab = widgets.VBox([widgets.HBox([sim_controls, conv_widget]), sim_output])
    
    # --- Onglet Plot ---
    spectra_select = widgets.SelectMultiple(
        options=[], 
        description="Available spectra:",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='80%', height='150px')
    )
    plot_button = widgets.Button(
        description="Draw", button_style="info",
        tooltip="Drawing selecting spectra"
    )
    plot_output = widgets.Output(layout=widgets.Layout(border="2px solid #ccc", padding="10px", min_height="400px"))
    plotted_lines = {}   # {label: (wl, R)}
    summaries = {}       # {label: (geom_summary, mat_summary)}
    def update_spectra():
        all_spectra, all_summaries = get_all_spectra_and_summaries(summary_dir, exp_data_dir, ordered_params)
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
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            for (row, col), cell in table.get_celld().items():
                if row >= 0 and col >= 0:
                    cell.get_text().set_color(colors[col % len(colors)])
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
            download_link = create_download_link(fig, filename=f"plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            display(download_link)
            plt.close(fig)
            
            
    plot_button.on_click(on_plot_button_clicked)
    plot_controls = widgets.VBox([
        widgets.HTML("<h3>Plot</h3>"),
        spectra_select,
        plot_button
    ])
    plot_tab = widgets.VBox([plot_controls, plot_output])
    
    # --- Onglet Difference ---
    diff_ref_dropdown = widgets.Dropdown(
        options=[],
        description="Base:",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='500px')
    )
    diff_target_dropdown = widgets.Dropdown(
        options=[],
        description="Comparing to:",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='500px')
    )
    diff_button = widgets.Button(
        description="Draw difference", button_style="warning",
        tooltip="Drawing the difference between two spectra"
    )
    diff_output = widgets.Output(layout=widgets.Layout(border="2px solid #ccc", padding="10px", min_height="400px"))
    
    def update_diff_options():
        spectra_all, _ = get_all_spectra_and_summaries(summary_dir, exp_data_dir, ordered_params)
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
        spectra_all, _ = get_all_spectra_and_summaries(summary_dir, exp_data_dir, ordered_params)
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
            download_link = create_download_link(fig, filename=f"difference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            display(download_link)
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

if __name__ == "__main__":
    app = create_advanced_app(json_combined_path, summary_dir, exp_data_dir)
    display(app)
