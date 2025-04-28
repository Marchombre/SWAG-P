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
import io, base64
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import HTML, display, Javascript
from datetime import datetime
import textwrap


# Construction des chemins
module_dir       = os.path.dirname(os.path.abspath(__file__))
workspace_dir    = os.path.dirname(module_dir)
notebooks_dir    = os.path.join(workspace_dir, "notebooks")
summary_dir      = os.path.join(notebooks_dir, "Summary_Simulation")
exp_data_dir     = os.path.join(notebooks_dir, "Experimental_Data")
configurations_dir = os.path.join(workspace_dir, "CONFIGURATIONS")
data_dir         = os.path.join(workspace_dir, "data")
json_combined_path = os.path.join(data_dir, "combined_materials.json")

# chemin du CSV
auto_modes_path = os.path.join(workspace_dir, 'Convergence', 'optimal_n_modes.csv')
try:
    auto_modes_df = pd.read_csv(auto_modes_path, index_col='config_name')
except Exception:
    auto_modes_df = None


# Importations internes
from simulate_and_plot import ordered_params, run_simulation_one_combo
from data_readers import (
    list_sim_summary_files,
    get_all_spectra_and_summaries
)

from convergence_analysis import create_multi_convergence_widget
from Saving_Functions import save_simulation_summary, save_figure, get_material_str_clean
from Characterization import find_best_dip_fwhm, minmax




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
    sim_lambda_min = widgets.FloatText(value=600.0, description="lam_ min (nm):",
                                        layout=widgets.Layout(width='150px'),
                                        style={'description_width': 'initial'})
    sim_lambda_max = widgets.FloatText(value=1000.0, description="lam_ max (nm):",
                                        layout=widgets.Layout(width='150px'),
                                        style={'description_width': 'initial'})
    sim_n_points = widgets.IntText(value=200, description="Points:",
                                    layout=widgets.Layout(width='200px'),
                                    style={'description_width': 'initial'})
    sim_n_mod = widgets.IntText(value=10, description="Modes:",
                                 layout=widgets.Layout(width='200px'),
                                 style={'description_width': 'initial'})
    sim_run_button = widgets.Button(description="Run simulation", button_style="success",
                                    tooltip="Lancer la simulation")
    
    # mode de calcul des modes
    mode_selection = widgets.RadioButtons(
        options=[('Fixe', 'fixed'),
                 ('Personnalisé', 'custom'),
                 ('Automatique', 'auto')],
        value='fixed',
        description='Modes:',
        style={'description_width': 'initial'}
    )
    # boîte qui contiendra, en mode custom, un IntText par config sélectionnée
    custom_modes_box = widgets.VBox()
    
    
    
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
    
    sim_refresh_button = widgets.Button(description="Refresh files", button_style="info",
                                        tooltip="Rafraîchir les fichiers de simulation")
    sim_refresh_button.on_click(lambda b: sim_files_dropdown.set_trait("options", list_sim_summary_files(summary_dir)))
        
    
    sim_download_button = widgets.Button(description="Download", button_style="danger",
                                         tooltip="Download the selected simulation file")
    sim_download_output = widgets.Output()
    def create_file_download_link(file_path, link_text=None):
        with open(file_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode("utf-8")
        if link_text is None:
            link_text = os.path.basename(file_path)
        return HTML(f'<a download="{os.path.basename(file_path)}" href="data:application/octet-stream;base64,{b64}" target="_blank">{link_text}</a>')
    

    def on_download_clicked(b):
        selected_file = sim_files_dropdown.value
        if selected_file:
            with open(selected_file, "rb") as f:
                data = f.read()
            b64 = base64.b64encode(data).decode("utf-8")
            file_name = os.path.basename(selected_file)
            js_code = f"""
            var a = document.createElement('a');
            a.href = "data:application/octet-stream;base64,{b64}";
            a.download = "{file_name}";
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            """
            display(Javascript(js_code))
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
    
    # Nouveau widget pour saisir le nom de la simulation
    sim_name_widget = widgets.Text(
        value="",
        placeholder="Nom de simulation (auto si vide)",
        description="Sim Name:",
        layout=widgets.Layout(width='500px'),
        style={'description_width': 'initial'}
    )                      

    # Création d'un widget de sélection multiple pour choisir une ou plusieurs configurations de simulation.
    # La largeur est fixée à 350px.
    sim_config_selector = widgets.SelectMultiple(
        options=[(cfg["config_name"], cfg) for cfg in all_configs],
        description="Config simulation:",
        layout=widgets.Layout(width='500px', height='150px'),
        style={'description_width': 'initial'}  # Permet de conserver la largeur par défaut pour la description
    )
    
    custom_n_mod_inputs = {}
    def refresh_custom_modes(*args):
        if mode_selection.value == 'custom':
            inputs = []
            for cfg in sim_config_selector.value:
                name = cfg['config_name']
                inp = widgets.IntText(
                    value=sim_n_mod.value,
                    description=name,
                    layout=widgets.Layout(width='300px'),
                    style={'description_width': 'initial'}
                )
                custom_n_mod_inputs[name] = inp
                inputs.append(inp)
            custom_modes_box.children = inputs
        else:
            custom_modes_box.children = []

    mode_selection.observe(refresh_custom_modes, names='value')
    sim_config_selector.observe(refresh_custom_modes, names='value')
    
    
    
    # Checkbox pour activer/désactiver le verbose
    verbose_toggle = widgets.Checkbox(
        value=True,
        description="Verbose",
        indent=False,
        layout=widgets.Layout(width='150px'),
        style={'description_width': 'initial'}
    )


    sim_debug = widgets.Output(
        layout=widgets.Layout(
            width='100%',
            height='200px',
            overflow_y='auto',
            border='1px solid darkred',
            display='block' if verbose_toggle.value else 'none'
        )
    )
    # Masque/affiche et vide le contenu quand on décoche/coché verbose
    def toggle_sim_debug(change):
        sim_debug.layout.display = 'block' if change['new'] else 'none'
        if not change['new']:
            sim_debug.clear_output()
    verbose_toggle.observe(toggle_sim_debug, names='value')

    
    def load_configs():
        combos_file = os.path.join(configurations_dir, "geom_mat_combinations.json")
        if os.path.exists(combos_file):
            with open(combos_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("ALL_COMBINED_CONFIGS", [])
        return []

    def refresh_configs(b):
        global all_configs
        # Recharge le fichier JSON avec les dernières modifications
        all_configs = load_configs()
        # Met à jour les options du sélecteur avec la nouvelle liste de configurations
        sim_config_selector.options = [(cfg["config_name"], cfg) for cfg in all_configs]

    # Création du bouton de rafraîchissement
    config_refresh_button = widgets.Button(
        description="Refresh Configs", 
        button_style="info",
        tooltip="Rafraîchir les fichiers de configurations"
    )
    # Attache le callback au clic sur le bouton
    config_refresh_button.on_click(refresh_configs)

   

    # Regroupement vertical des contrôles de simulation dans un conteneur VBox.
    # Chaque ligne est soit un HBox (pour afficher plusieurs éléments horizontalement),
    # soit un widget unique.
    sim_controls = widgets.VBox(
        [
            widgets.HTML("<h3>Simulation - Paramètres</h3>"),
            sim_name_widget, sim_files_box,  # La première ligne : fichier(s) sélection et boutons associés.
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
            
            widgets.VBox(
                [sim_run_button, config_refresh_button],
                layout=widgets.Layout(justify_content='center', margin='5px 0px')
            ),  # Ligne contenant le bouton de lancement de la simulation.
            sim_config_selector,            # Affichage du sélecteur multiple pour les configurations de simulation.
            mode_selection, custom_modes_box,
            verbose_toggle
        ],
        layout=widgets.Layout(
            padding='10px',         # Espace interne autour de ces éléments
            border='solid 1px lightgray'
        )
    )

    
    # Droite : Contrôles et tracé de convergence
    conv_widget = create_multi_convergence_widget(json_combined_path, all_configs)
    
    
        
        
        
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
            # Affichage d'un spinner pendant la simulation
            spinner = widgets.HTML(
                "<div style='text-align: center;'><img src='https://i.gifer.com/ZZ5H.gif' width='50px'/><br><em>Simulation en cours...</em></div>"
            )
            display(spinner)
            
            # Récupération des paramètres
            lam_min = sim_lambda_min.value
            lam_max = sim_lambda_max.value
            n_points = sim_n_points.value
                        
            wave = {"angle": 0, "polarization": 1}
            lam_range = np.linspace(lam_min, lam_max, n_points)
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            n_colors = len(colors)

            
            # détermine n_mod pour chaque config
            mode = mode_selection.value
            mode_by_cfg = {}
            if mode == 'fixed':
                # même n_mod pour tous
                for cfg in sim_config_selector.value:
                    mode_by_cfg[cfg['config_name']] = sim_n_mod.value
            elif mode == 'custom':
                # récupère la saisie personnalisée
                for name, inp in custom_n_mod_inputs.items():
                    mode_by_cfg[name] = inp.value
            else:  # auto
                if auto_modes_df is None:
                    raise FileNotFoundError(f'Modes auto introuvables: {auto_modes_path}')
                for cfg in sim_config_selector.value:
                    name = cfg['config_name']
                    mode_by_cfg[name] = int(auto_modes_df.loc[name, 'optimal_n_mode'])
            
            
            
            # Configuration de l'axe du tracé
            # 1) Définition des marges pour 80 % de la largeur
            left_marges, width_marges = 0.10, 0.80
            fig      = plt.figure(figsize=(13, 9))
            ax_plot = fig.add_axes([left_marges, 0.50, width_marges, 0.35])
            ax_table = fig.add_axes([left_marges, 0.05, width_marges, 0.35])
            ax_table.axis('off')
                        
            # Initialisation des listes pour le tracé et le tableau ainsi que du dictionnaire des détails de simulation
            config_labels = []
            geom_summaries = []
            mat_summaries = []
            simulation_details = {}
            fwhm_summaries = []
            lam_summaries = []
            
            S_lam_summaries  = []   
            # listes numériques pour S_lam
            S_lam_min_vals = []
            S_lam_sym_vals = []
            
            Q_factor = []
            raw_score_summaries = []   # depth*slope/width
            debug_lines = []
            
            # Pour chaque configuration sélectionnée, lancer la simulation et collecter les courbes et résumés
            verbose = verbose_toggle.value
            
            
            for idx, cfg in enumerate(sim_config_selector.value):
                name    = cfg['config_name']
                n_modes = mode_by_cfg[name]
                Rup, _, details = run_simulation_one_combo(
                    lam_range, wave, n_modes, cfg, json_combined_path
                )

                lam = np.array(lam_range)
                Rup = np.array(Rup)

                # Enregistrer les détails
                simulation_details[cfg["config_name"]] = details

                lam_left, lam_right, width, lam_dip, Rdip, ylev, lam_m_l, Rm_l, \
                lam_m_r, Rm_r, lam_sym, R_sym, slope, depth, raw_score, dips, scores_list, depths, slopes, widths, \
                lam_max_ls, R_max_ls, lam_max_rs, R_max_rs, lam_syms, R_syms = \
                    find_best_dip_fwhm(lam, Rup,
                                    smooth_win=0, # odd integer ≥ 3: window length for Savitzky–Golay smoothing
                                    polyorder=0,   # polynomial order for the filter (must be < smooth_win)
                                    dip_prom=0.01, # min “prominence” (depth) to qualify as a dip
                                    dip_dist=5,   # min separation (in points) between dips
                                    peak_dist=5,
                                    verbose = True)  # min separation (in points) between maxima


                # On choisit le max de plus petite amplitude 
                if Rm_l < Rm_r:
                    lam_min  = lam_m_l
                    lam_middle = lam_left
                else:
                    lam_min  = lam_m_r
                    lam_middle = lam_right
                
                #  on ajoute S_lam
                S_lam_min_abs = abs((lam_dip   - lam_min) / lam_middle)
                S_lam_sym_abs = abs((lam_dip   - lam_sym  ) / lam_middle)
                # Ajout pour mémoriser les valeurs absolues
                S_lam_min_vals.append(S_lam_min_abs)
                S_lam_sym_vals.append(S_lam_sym_abs)                
                
                
                color = colors[idx % n_colors]
                ax_plot.plot(lam_range, Rup, color=color)

                if verbose:
                    # barre horizontale au niveau ylev
                    ax_plot.hlines(ylev, xmin=lam_left, xmax=lam_right,
                                linewidth=2, color=color,
                                label='_nolegend_')

                    # croix de dips détectés
                    ax_plot.scatter(
                        lam[dips], Rup[dips],
                        marker='x', s=40,
                        color=color,
                        label='_nolegend_'
                    )
                    # croix des maxima initiaux
                    ax_plot.scatter(
                        lam_max_ls, R_max_ls,
                        marker='x', s=30,
                        color=color, label='_nolegend_'
                    )
                    ax_plot.scatter(
                        lam_max_rs, R_max_rs,
                        marker='x', s=30,
                        color=color, label='_nolegend_'
                    )
                    # croix des points symétriques
                    ax_plot.scatter(
                        lam_syms, R_syms,
                        marker='x', s=30,
                        color=color, label='_nolegend_'
                    )
                    # annotation du dip retenu
                    ax_plot.scatter([lam_dip], [Rdip],
                                    marker='o', s=70,
                                    facecolor='none',
                                    edgecolor=color,
                                    linewidths=2,
                                    label='_nolegend_')


    
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
                
                
                
                #  on ajoute la FWHM 
                fwhm_summaries.append(f"{width:.1f} nm")
                #  on ajoute lambda
                lam_summaries.append(f"{lam_dip:.1f} nm")    
                # on ajoute S_lam
                S_lam_summaries.append(f"{S_lam_min_abs:.3f} & {S_lam_sym_abs:.3f}")                # on ajoute le Q-factor
                Q_factor.append(f"{(lam_dip / width):.1f}")            
                # on ajoute le score interne
                raw_score_summaries.append(f"{raw_score:.2e}")
                
                            
                if verbose:
                    dips_nm  = ", ".join(f"{l:.1f}" for l in lam[dips])
                    scores_str = ", ".join(f"{s:.3e}" for s in scores_list)
                    depths_str = ", ".join(f"{d:.3f}"  for d in depths)
                    slopes_str = ", ".join(f"{s:.3e}" for s in slopes)
                    widths_str = ", ".join(f"{w:.3f}" for w in widths)
                
                    # Ligne unique résumé pour ce spectre
                    debug_lines.append(
                        f"{cfg['config_name']}: dips[{dips_nm}]  "
                        f"dip{lam_dip:.1f}nm  "
                        f"depths=[{depths_str}]  "
                        f"depth={depth:.3f}  "
                        f"slopes=[{slopes_str}]  "
                        f"slope={slope:.3e}  "
                        f"FWHMs=[{widths_str}]  "
                        f"FWHM={width:.1f}  "
                        f"scores=[{scores_str}]  "
                        f"score={raw_score:.3e}  "
                    )   
                
                    # sauvegarde **configuration par configuration**
                    # on utilise [-1] pour ne prendre que la dernière entrée de chaque liste (celle de la config en cours).
                    save_simulation_summary(
                        { name: simulation_details[name] },  # un dict à une seule entrée
                        lam_range,
                        wave,
                        n_modes,                             # le n_modes spécifique
                        summary_dir,
                        custom_name=name,                   # nom du fichier = nom de la config
                        fwhm_summaries=[fwhm_summaries[-1]],
                        lam_summaries=[lam_summaries    [-1]],
                        S_lam_summaries=[S_lam_summaries[-1]],
                        Q_factor=[Q_factor               [-1]],
                        raw_score_summaries=[raw_score_summaries[-1]],
                        #comp_summaries=[comp_summaries   [-1]]
                    )
                    
            # calculer la configuration optimale 
            if S_lam_min_vals:
                # calcul de la distance euclidienne (norme 2) pour chaque couple
                norms = [np.hypot(a, b) for a, b in zip(S_lam_min_vals, S_lam_sym_vals)]
                best_idx = int(np.argmin(norms))
                # on récupère le nom de la config correspondante
                best_cfg = [cfg['config_name'] for cfg in sim_config_selector.value][best_idx]
                # on peut aussi afficher les valeurs exactes
                best_min = S_lam_min_vals[best_idx]
                best_sym = S_lam_sym_vals[best_idx]
                debug_lines.append(
                    f"→ BEST_CONFIG: {best_cfg}  "
                    f"(S_lam_min={best_min:.3f}, S_lam_sym={best_sym:.3f})"
                )


            # On fusionne toutes les lignes en un texte multi-lignes
            debug_txt = "\n".join(debug_lines)           


            # === début nouveau bloc de wrapping ===
            wrapper = textwrap.TextWrapper(
                width=150,             # nombre max de caractères par ligne
                break_long_words=True,
                replace_whitespace=False
            )
            wrapped = []
            for line in debug_txt.splitlines():
                # wrapper.wrap renvoie [] si line == ""
                wrapped.extend(wrapper.wrap(line) or [""])
            # on écrase debug_txt par sa version "coupée"
            debug_txt = "\n".join(wrapped)
            # === fin bloc de wrapping ===

            # Affiche le debug dans le widget sim_debug
            sim_debug.clear_output()
            if verbose:
                with sim_debug:
                    # debug_txt contient déjà le texte “wrapped”
                    display(widgets.Textarea(
                        value=debug_txt,
                        layout=widgets.Layout(
                            width='100%',
                            height='200px',
                            overflow_y='auto'
                        )
                    ))


            Rn = minmax(raw_score_summaries)
            Qn = minmax([float(q) for q in Q_factor])

            comp = (Rn + Qn ) / 3.0
            comp_summaries = [f"{c:.3f}" for c in comp]
            
            
            
            # FINALISATION DU TRACÉ
            ax_plot.set_xlabel("Wavelength (nm)")
            ax_plot.set_ylabel("Reflectance")
            ax_plot.set_title("Simulation")
            ax_plot.grid(True)
            
            # Désactivation de l'affichage des axes dans l'axe qui contiendra le tableau
            ax_table.axis("off")
            
            # Ajustement des noms pour le tableau (ajout d'un saut de ligne si nécessaire)
            config_labels = [label.replace("Mat_", "\nMat_") for label in config_labels]
            
            if config_labels:
                # 1) nombre de colonnes
                n_configs = len(config_labels)
                # 2) taille de police dynamique
                fontsize = 8 if n_configs <= 5 else max(8 - (n_configs - 5), 3)
                table = ax_table.table(
                    cellText=[
                        geom_summaries,
                        mat_summaries,
                        fwhm_summaries,
                        lam_summaries,
                        S_lam_summaries,
                        Q_factor,
                        raw_score_summaries,
                        #comp_summaries
                    ],
                    colLabels=config_labels,
                    rowLabels=[
                        "Geometry", "Material",
                        "FWHM", r"$\lambda_0$",
                        "S_lam L & R", "Q-factor",
                        "Score interne"#, "Score total"
                    ],
                    loc="center", cellLoc="left"
                )
                table.auto_set_font_size(False)
                table.set_fontsize(fontsize)
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
                # Appliquer les couleurs aux textes des cellules du corps
                for (row, col), cell in table.get_celld().items():
                    if row >= 0 and col >= 0:
                        cell.get_text().set_color(colors[col % len(colors)])
                # Ajustement dynamique de la hauteur des cellules
                row_heights = {}
                for (row, col), cell in table.get_celld().items():
                    if row >= 0:
                        txt = cell.get_text().get_text()
                        nb_lines = txt.count("\n") + 1
                        row_heights[row] = max(row_heights.get(row, 0), nb_lines)
                for (row, col), cell in table.get_celld().items():
                    if row in row_heights:
                        cell.set_height(0.04 * row_heights[row])
            
            ax_table.figure.canvas.draw_idle()  # Mise à jour du tableau
            

            # transforme mode_by_cfg en liste alignée sur simulation_details:
            n_mod_list = [ mode_by_cfg[name] for name in simulation_details.keys() ]

            save_simulation_summary(
                simulation_details,
                lam_range,
                wave,
                n_mod_list,              # liste de n_modes
                summary_dir,
                custom_name=sim_name_widget.value,
                fwhm_summaries=fwhm_summaries,
                lam_summaries=lam_summaries,
                S_lam_summaries=S_lam_summaries,
                Q_factor=Q_factor,
                raw_score_summaries=raw_score_summaries,
                #comp_summaries=comp_summaries
            )

            
            figures_dir = os.path.join(workspace_dir, "Figures")
            material_str_clean = get_material_str_clean(simulation_details)
            save_figure(fig, "Simulation Reflectance Spectra", figures_dir, material_str_clean)
            
            # Affichage final de la figure et du lien de téléchargement
            sim_output.clear_output(wait=True)
            display(fig)
            download_link = create_download_link(fig, filename=f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            display(download_link)
            plt.close(fig)

            
            
    sim_run_button.on_click(on_sim_run_clicked)
    
    # Assemblage de l'onglet Simulation : partie haute (deux colonnes) + partie basse (figure simulation et tableau)
    sim_tab = widgets.VBox([
        widgets.HBox([sim_controls, conv_widget]),
        sim_debug,     
        sim_output
    ])    
    
    
    
        # --- Onglet Plot ---

    # 1) Widgets
    spectra_select = widgets.SelectMultiple(
        options=[],
        description="Available spectra:",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='80%', height='150px')
    )
    plot_verbose_toggle = widgets.Checkbox(
        value=True,
        description="Verbose",
        indent=False,
        layout=widgets.Layout(width='150px'),
        style={'description_width': 'initial'}
    )
    
    
    plot_debug = widgets.Output(
        layout=widgets.Layout(
            width='100%',
            height='200px',
            overflow_y='auto',
            border='1px solid darkred',
            display='block' if plot_verbose_toggle.value else 'none'
        )
    )
    def toggle_plot_debug(change):
        plot_debug.layout.display = 'block' if change['new'] else 'none'
        if not change['new']:
            plot_debug.clear_output()
    plot_verbose_toggle.observe(toggle_plot_debug, names='value')    
    
    
    plot_button = widgets.Button(
        description="Draw", button_style="info",
        tooltip="Draw selected spectra"
    )
    plot_output = widgets.Output(
        layout=widgets.Layout(border="2px solid #ccc", padding="10px", min_height="400px")
    )
    plot_controls = widgets.VBox([
        widgets.HTML("<h3>Plot</h3>"),
        spectra_select,
        plot_verbose_toggle,
        plot_button
    ])

    # 2) Variables partagées
    plotted_lines = {}    # {label: (wavelength_array, reflectance_array)}
    summaries     = {}    # {label: (geom_summary, mat_summary)}
    metrics_all   = {}    # {label: metrics_dict}

    # 3) Fonction de mise à jour des spectres disponibles

    def update_spectra():
        nonlocal plotted_lines, summaries, metrics_all
        spectra, sums, mets = get_all_spectra_and_summaries(
            summary_dir, exp_data_dir, ordered_params
        )
        spectra_select.options = list(spectra.keys())
        plotted_lines = spectra
        summaries     = sums
        metrics_all   = mets

    # appel initial
    update_spectra()

    # 4) Callback de tracé

    def on_plot_button_clicked(b):
        # a) rafraîchir les données
        update_spectra()
        verbose = plot_verbose_toggle.value
        
        # b) marges et création de la figure + axes
        left_marges, width_marges = 0.10, 0.80
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        n_colors = len(colors)

        fig = plt.figure(figsize=(13, 9))
        
        ax_plot  = fig.add_axes([left_marges, 0.50, width_marges, 0.35])
        ax_table = fig.add_axes([left_marges, 0.05, width_marges, 0.35])
        ax_table.axis('off')
        
        # c) préparation des listes
        config_labels      = []
        geom_summaries     = []
        mat_summaries      = []
        fwhm_summaries     = []
        lam_summaries      = []
        S_lam_min_vals     = []
        S_lam_sym_vals     = []
        S_lam_summaries    = []
        Q_factor_list      = []
        raw_score_list     = []
        debug_lines        = []
        
        # d) détermination des labels à tracer
        labels = list(spectra_select.value) or list(plotted_lines.keys())
        
        # e) boucle de tracé et calcul des métriques
        for idx, label in enumerate(labels):
            # données
            wl, R  = plotted_lines[label]
            lam = np.array(wl)
            Rup = np.array(R)
            
            # calcul dip/FWHM
            (lam_left, lam_right, width_fwhm, lam_dip, Rdip, ylev,
            lam_m_l, Rm_l, lam_m_r, Rm_r, lam_sym, R_sym, slope,
            depth, raw_score, dips, scores_list, depths, slopes,
            widths, lam_max_ls, R_max_ls, lam_max_rs, R_max_rs,
            lam_syms, R_syms) = find_best_dip_fwhm(
                lam, Rup,
                smooth_win=0, polyorder=0,
                dip_prom=0.01, dip_dist=5,
                peak_dist=5, verbose=True
            )
            # On choisit le max de plus petite amplitude 
            if Rm_l < Rm_r:
                lam_min  = lam_m_l
                lam_middle = lam_left
            else:
                lam_min  = lam_m_r
                lam_middle = lam_right
            
            #  on ajoute S_lam
            S_lam_min_abs = abs((lam_dip - lam_min)   / lam_middle)
            S_lam_sym_abs = abs((lam_dip - lam_sym)   / lam_middle)
            # Ajout pour mémoriser les valeurs absolues
            S_lam_min_vals.append(S_lam_min_abs)
            S_lam_sym_vals.append(S_lam_sym_abs)
            
            
            color = colors[idx % n_colors]

            # tracé principal
            ax_plot.plot(lam, Rup, color=color)
            
            # tracés conditionnels (verbose)
            if verbose:
                ax_plot.hlines(ylev, xmin=lam_left, xmax=lam_right,
                            linewidth=2, colors=color)
                ax_plot.scatter(lam[dips], Rup[dips], marker='x', s=40, color=color)
                ax_plot.scatter(lam_max_ls, R_max_ls, marker='x', s=30, color=color)
                ax_plot.scatter(lam_max_rs, R_max_rs, marker='x', s=30, color=color)
                ax_plot.scatter(lam_syms, R_syms, marker='x', s=30, color=color)
                ax_plot.scatter([lam_dip], [Rdip], marker='o', s=70,
                                facecolor='none', edgecolor=color, linewidths=2)
                # ligne debug text
                dips_nm  = ", ".join(f"{l:.1f}" for l in lam[dips])
                scores_str = ", ".join(f"{s:.3e}" for s in scores_list)
                depths_str = ", ".join(f"{d:.3f}"  for d in depths)
                slopes_str = ", ".join(f"{s:.3e}" for s in slopes)
                widths_str = ", ".join(f"{w:.3f}" for w in widths)
            
                # Ligne unique résumé pour ce spectre
                debug_lines.append(
                    f"{label}:  "
                    f"dips=[{dips_nm}]  "
                    f"dip{lam_dip:.1f}nm  "
                    f"depths=[{depths_str}]  "
                    f"depth={depth:.3f}  "
                    f"slopes=[{slopes_str}]  "
                    f"slope={slope:.3e}  "
                    f"FWHMs=[{widths_str}]  "
                    f"FWHM={width_fwhm:.1f}  "
                    f"scores=[{scores_str}]  "
                    f"score={raw_score:.3e}  "
                )  
            
            # stockage pour le tableau
            config_labels.append(label)
            geom_summaries.append(summaries[label][0])
            mat_summaries.append(summaries[label][1])
            fwhm_summaries.append(f"{width_fwhm:.1f} nm")
            lam_summaries.append(f"{lam_dip:.1f} nm")
            S_lam_summaries.append(f"{S_lam_min_abs:.3f} & {S_lam_sym_abs:.3f}") 
            Q_factor_list.append(f"{lam_dip/width_fwhm:.1f}")
            raw_score_list.append(f"{raw_score:.2e}")
        
        
        
        if S_lam_min_vals:
            # norme euclidienne sur chaque couple
            norms = [np.hypot(a, b) for a, b in zip(S_lam_min_vals, S_lam_sym_vals)]
            best_idx   = int(np.argmin(norms))
            best_label = config_labels[best_idx]
            best_min   = S_lam_min_vals[best_idx]
            best_sym   = S_lam_sym_vals[best_idx]
            debug_lines.append(
                f"BEST_CONFIG → {best_label}  "
                f"(S_lam_min={best_min:.3f}, S_lam_sym={best_sym:.3f})"
            )
            
        
        # f) wrapping du debug text
        debug_txt = "\n".join(debug_lines)
        wrapper = textwrap.TextWrapper(width=100, break_long_words=True, replace_whitespace=False)
        wrapped = []
        for line in debug_txt.splitlines():
            wrapped.extend(wrapper.wrap(line) or [""])
        debug_txt = "\n".join(wrapped)
        
        # Affiche le debug dans le widget plot_debug
        plot_debug.clear_output()
        if verbose:
            with plot_debug:
                display(widgets.Textarea(
                    value=debug_txt,
                    layout=widgets.Layout(
                        width='100%',
                        height='200px',
                        overflow_y='auto'
                    )
                ))

        
        # h) finalisation du tracé
        ax_plot.set_xlabel("Wavelength (nm)")
        ax_plot.set_ylabel("Reflectance")
        ax_plot.set_title("Spectres combinés")
        ax_plot.grid(True)
        
        # i) construction du tableau
        config_labels = [lbl.replace("Mat_","\nMat_") for lbl in config_labels]
        if config_labels:
            n = len(config_labels)
            fontsize = 8 if n <= 5 else max(8 - (n - 5), 3)
            table = ax_table.table(
                cellText=[
                    geom_summaries, mat_summaries,
                    fwhm_summaries, lam_summaries,
                    S_lam_summaries, Q_factor_list,
                    raw_score_list
                ],
                colLabels=config_labels,
                rowLabels=[
                    "Geometry", "Material", "FWHM", r"$\lambda_0$",
                    "S_lam L & R", "Q-factor", "Score interne"
                ],
                loc="center", cellLoc="left"
            )
            table.auto_set_font_size(False)
            table.set_fontsize(fontsize)
            table.auto_set_column_width(col=list(range(n)))
            for (r, c), cell in table.get_celld().items():
                if r == -1 or c == -1:
                    cell.set_facecolor("#40466e")
                    cell.set_text_props(weight="bold", color="white", fontsize=fontsize)
                else:
                    cell.set_facecolor("whitesmoke")
                    cell.set_edgecolor("lightgray")
                    cell.set_linewidth(0.5)
                    cell.get_text().set_color(colors[c % len(colors)])
            heights = {}
            for (r, c), cell in table.get_celld().items():
                if r >= 0:
                    lines = cell.get_text().get_text().count("\n") + 1
                    heights[r] = max(heights.get(r, 0), lines)
            for (r, c), cell in table.get_celld().items():
                if r in heights:
                    cell.set_height(0.04 * heights[r])

        # j) affichage final et lien
        with plot_output:
            plot_output.clear_output(wait=True)
            display(fig)
            link = create_download_link(
                fig,
                filename=f"plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            display(link)
            plt.close(fig)

    # 5) Bind du callback et assemblage du tab
    plot_button.on_click(on_plot_button_clicked)
    
    plot_tab = widgets.VBox([
        plot_controls,
        plot_debug,     # ← inséré juste ici
        plot_output
    ])
    
    
    
    
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
        # Ici aussi on doit unpacker les trois, même si on n'utilise que spectra_all
        spectra_all, _, _ = get_all_spectra_and_summaries(
            summary_dir, exp_data_dir, ordered_params
        )
        options = list(spectra_all.keys())
        diff_ref_dropdown.options    = options
        diff_target_dropdown.options = options


    # Initialisations
    update_spectra()
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