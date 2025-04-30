#!/usr/bin/env python3
"""
Module: convergence_analysis.py

Ce module permet de calculer et tracer la convergence de la réflectance (Rup) en fonction
du nombre de modes (n_mode) pour une longueur d'onde fixe. L'interface proposée permet à 
l'utilisateur de choisir :
  - La valeur de la longueur d'onde fixe.
  - Le nombre maximum de modes à tester.
  - La tolérance servant à définir la convergence.
  
Après avoir cliqué sur "Tracer convergence", le module calcule Rup pour chaque n_mode et trace 
la courbe. Un label affiche également le n_mode optimal, défini comme le premier n_mode pour lequel 
la variation absolue de Rup devient inférieure à la tolérance (ce qui signifie la stabilisation des résultats).
"""

import math

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime
import zipfile
from pathlib import Path
import ipywidgets as widgets
from IPython.display import FileLink, display
from simulate_reflectance import simulate_reflectance_single


def compute_convergence(lambda_fixed, n_mode_max, geometry, wave, df_config, json_combined_path, ri_overrides, tolerance, n_step, stable_required, progress_bar=None):
    """
    Calcule la convergence de Rup pour une longueur d'onde fixe en faisant varier n_mode.
    
    Args:
        lambda_fixed (float): La longueur d'onde fixe (en nm).
        n_mode_max (int): Nombre maximum de modes à tester.
        geometry (dict): Configuration géométrique.
        wave (dict): Paramètres d'onde (e.g. {"angle": 0, "polarization": 1}).
        df_config (pd.DataFrame): Configuration des matériaux.
        json_combined_path (str): Chemin vers le JSON combiné.
        ri_overrides (dict): Remplacements pour l'indice de réfraction.
        tolerance (float): Seuil de variation pour définir la convergence.
        n_step (int): Pas de variation pour n_mode.
        stable_required (int): Nombre d'itérations consécutives (en dessous de la tolérance)
                               requis pour considérer que la convergence est stable.
        progress_bar (ipywidgets widget): Optionnel, barre de progression à mettre à jour.
    
    Returns:
        n_modes (np.array): Tableau des valeurs de n_mode testées.
        Rup_vals (list): Liste des valeurs de Rup calculées pour chaque n_mode.
        optimal_n_modes (int): Valeur optimale de n_mode, c'est-à-dire le premier n_mode
                              pour lequel la variation se maintient < tolerance sur stable_required itérations.
    """
    n_modes = np.arange(1, n_mode_max + 1, n_step)
    Rup_vals = []
    
    # Mise à jour de wave avec la longueur d'onde fixe
    wave_updated = dict(wave)
    wave_updated["wavelength"] = lambda_fixed
    
    for idx, n in enumerate(n_modes):
        # simulate_reflectance_single attend une liste de longueurs d'onde ; ici, un seul élément
        Rup, _ = simulate_reflectance_single([lambda_fixed], geometry, wave_updated, df_config, json_combined_path, n_mod=n, ri_overrides=ri_overrides)
        Rup_vals.append(Rup[0])
        if progress_bar is not None:
            progress_bar.value += 1
            time.sleep(0.01)
    
    diffs = np.abs(np.diff(Rup_vals))
    optimal_n_mode = n_mode_max  # Valeur par défaut si aucune stabilité n'est détectée
    stable_count = 0
    
    for i, d in enumerate(diffs):
        if d < tolerance:
            stable_count += 1
            if stable_count >= stable_required:
                optimal_n_mode = int(n_modes[i+1])
                break
        else:
            stable_count = 0
            
    return n_modes, Rup_vals, optimal_n_mode




# --- Widget de convergence ---

stable_required_widget = widgets.IntText(
    value=3, 
    description="Stable required:", 
    layout=widgets.Layout(width='150px')
)


def create_multi_convergence_widget(json_combined_path, all_configs):
    """
    Crée un widget pour tracer la convergence de Rup en fonction de n_mode pour une ou plusieurs configurations.
    La partie haute comporte :
      - Un sélecteur de configuration.
      - Un bouton "Refresh Configs" placé à droite du sélecteur permettant de rafraîchir la liste des configurations.
      - Deux lignes de champs numériques (deux par ligne).
      - Une ligne avec la barre de progression située à gauche et le bouton "Tracer convergence" aligné à droite.
    En dessous, le widget affiche le tracé de convergence.
    """
    # Création d'un widget de sélection multiple pour choisir les configurations à utiliser pour la convergence.
    conv_config_selector = widgets.SelectMultiple(
        options=[(cfg["config_name"], cfg) for cfg in all_configs],
        description="Config convergence:",
        layout=widgets.Layout(width='580px'),
        style={'description_width': 'initial'}
    )
    
    # Bouton pour rafraîchir les configurations de convergence
    refresh_conv_configs_button = widgets.Button(
        description="Refresh Configs", 
        button_style="info",
        tooltip="Rafraîchir les configurations de convergence"
    )
    
    def refresh_conv_configs(b):
        combos_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CONFIGURATIONS", "geom_mat_combinations.json")
        new_configs = []
        if os.path.exists(combos_file):
            with open(combos_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            new_configs = data.get("ALL_COMBINED_CONFIGS", [])
        if new_configs:
            conv_config_selector.options = [(cfg["config_name"], cfg) for cfg in new_configs]
    refresh_conv_configs_button.on_click(refresh_conv_configs)
    
    # On place le sélecteur et le bouton refresh côte à côte.
    conv_config_selector_box = widgets.HBox([conv_config_selector],
                                              layout=widgets.Layout(align_items='center'))

    # Widgets numériques pour saisir les paramètres de convergence.
    lambda_fixed_widget = widgets.FloatText(
        value=700.0, 
        description="λ fixe (nm):", 
        layout=widgets.Layout(width='150px')
    )
    n_mode_max_widget = widgets.IntText(
        value=100, 
        description="n_mode max:", 
        layout=widgets.Layout(width='150px')
    )
    n_mode_step_widget = widgets.IntText(
        value=1, 
        description="Step n_mode:", 
        layout=widgets.Layout(width='150px')
    )
    tolerance_widget = widgets.FloatText(
        value=1e-3, 
        description="Tolérance:", 
        layout=widgets.Layout(width='150px')
    )
    
    # Validation pour empêcher les valeurs négatives
    def validate_positive(change):
        if change['new'] < 0:
            change['owner'].value = 0  
    lambda_fixed_widget.observe(validate_positive, names='value')
    n_mode_max_widget.observe(validate_positive, names='value')
    n_mode_step_widget.observe(validate_positive, names='value')
    tolerance_widget.observe(validate_positive, names='value')

    # Répartition des champs numériques sur deux lignes.
    row1 = widgets.HBox([lambda_fixed_widget, n_mode_max_widget, refresh_conv_configs_button])
    row2 = widgets.HBox([n_mode_step_widget, tolerance_widget, stable_required_widget])
    numeric_controls = widgets.VBox([row1, row2])

    # Création du bouton "Tracer convergence"
    plot_button = widgets.Button(
        description="Run convergence", 
        button_style="primary", 
        layout=widgets.Layout(width="150px")
    )
    
    # Barre de progression
    overall_progress_bar = widgets.IntProgress(
        value=0, 
        min=0, 
        max=1,  # sera défini lors du déclenchement du calcul
        description='Progress:',
        layout=widgets.Layout(width='400px')
    )
    
    spacer = widgets.HBox([], layout=widgets.Layout(flex='1 1 auto'))
    
    button_container = widgets.HBox(
        [overall_progress_bar, spacer, plot_button],
        layout=widgets.Layout(width='100%')
    )
    
    # Widget de sortie pour afficher le tracé de convergence.
    conv_output = widgets.Output(
        layout=widgets.Layout(
            border="1px solid lightgray",
            width='630px',
            height='450px' 
        )
    )
    
    # Widget pour afficher les résultats dans un tableau
    results_table_output = widgets.Output(
        layout=widgets.Layout(border="1px solid lightgray", width='630px', max_height='200px', overflow='auto')
    )
    
    # Lien HTML qui deviendra le bouton de téléchargement
    download_link = widgets.HTML(
        value="",  # on mettra le bon <a> après le calcul
        placeholder="",
        description=""
    )




    def Conv_computation(b):
        conv_output.clear_output()
        
        lambda_fixed = lambda_fixed_widget.value
        n_mode_max   = n_mode_max_widget.value
        n_mode_step  = n_mode_step_widget.value
        tol          = tolerance_widget.value
        stable_required = stable_required_widget.value
        
        selected_configs = conv_config_selector.value
        if not selected_configs:
            with conv_output:
                print("Please select one or more configuration before computation.")
            return
        
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        fig, ax = plt.subplots(figsize=(7.5, 4))
        
        iterations_per_config = len(np.arange(1, n_mode_max + 1, n_mode_step))
        overall_progress_bar.max = len(selected_configs) * iterations_per_config
        overall_progress_bar.value = 0


        # listes pour stocker
        cfg_names   = []
        n_modes_all = []
        Rup_all     = []
        optimal_list= []
        
        
        for idx, cfg in enumerate(selected_configs):
            geometry_cfg = cfg["geometry"]["geometry"]
            material_config_list = cfg["material"]["MATERIALS_CONFIG"]
            df_config = pd.DataFrame(material_config_list)
            ri_overrides = cfg["material"].get("RI_OVERRIDES", {})
            
            n_modes, Rup_vals, optimal_n_mode = compute_convergence(
                lambda_fixed, n_mode_max, geometry_cfg, {"angle": 0, "polarization": 1},
                df_config, json_combined_path, ri_overrides, tolerance=tol, n_step=n_mode_step,
                stable_required=stable_required, progress_bar=overall_progress_bar
            )
            
            cfg_name = cfg["config_name"]
            cfg_names.append(cfg_name)
            n_modes_all.append(n_modes)
            Rup_all.append(Rup_vals)
            optimal_list.append(optimal_n_mode)
            
            
            color = colors[idx % len(colors)]
            ax.plot(n_modes, Rup_vals, label=cfg["config_name"], color=color)
        
        ax.set_xlabel("n_mode")
        ax.set_ylabel("Rup")
        ax.set_title(f"Convergence pour λ = {lambda_fixed} nm", fontsize=10)
        ax.grid(True)

        # -- légende dynamique à multiples colonnes sous le graphique --
        n_cfg    = len(selected_configs)
        max_rows = 10                             # nombre max de lignes par colonne
        ncol     = math.ceil(n_cfg / max_rows)    # nombre de colonnes
        # place la légende sous le plot, centrée
        ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.15),
            ncol=ncol,
            fontsize=8,
            frameon=False
        )
        # ajuste le bas du figure pour faire de la place à la légende
        fig.subplots_adjust(bottom=0.25)
        
        with conv_output:
            display(fig)
            plt.close(fig)


        # 1) Construction du DataFrame des résultats
        df_results = pd.DataFrame({
            "config_name":      cfg_names,
            "optimal_n_modes":   optimal_list
        })

        # 2) Affichage du tableau
        with results_table_output:
            results_table_output.clear_output()
            display(
                df_results.style
                    .set_table_styles([
                        {"selector": "th", "props": [("background-color", "#4B8BBE"),
                                                     ("color", "white"),
                                                     ("font-size","14px")]},
                        {"selector": "td", "props": [("font-size","12px")]}
                    ])
                    .set_caption("▶ Optimal n_mode per configuration")
                    .hide(axis="index")
            )


        # 3) Sauvegarde des json 
        module_dir = Path(__file__).resolve().parent       # …/Gap_Plasmon_2D/Workspace/modules
        project_root = module_dir.parent                    # …/Gap_Plasmon_2D/Workspace
        convergence_dir = project_root / "Convergence"
        convergence_dir.mkdir(exist_ok=True)

        # 3a) Master JSON metadata
        master_path = convergence_dir / "convergence_results.json"
        if master_path.exists():
            master = json.loads(master_path.read_text(encoding="utf-8"))
        else:
            master = {"configs": {}}

        # --- Pour chaque configuration du run courant, on ajoute sa méta-donnée ---
        for cfg_name, optimal_n in zip(cfg_names, optimal_list):
            entry = {
                "timestamp":       datetime.now().isoformat(),
                "lambda_fixed":    lambda_fixed,
                "n_mode_max":      n_mode_max,
                "n_mode_step":     n_mode_step,
                "tolerance":       tol,
                "stable_required": stable_required,
                "optimal_n_mode":  optimal_n
            }
            # Si c'est la première fois qu'on voit cfg_name, on crée la liste
            if cfg_name not in master["configs"]:
                master["configs"][cfg_name] = []
            # On ajoute la méta-donnée
            master["configs"][cfg_name].append(entry)

        # --- Écriture du master JSON ---
        master_path.write_text(
            json.dumps(master, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        # --- JSON par config pour le spectre (inchangé) ---
        for cfg_name, modes, vals in zip(cfg_names, n_modes_all, Rup_all):
            safe = cfg_name.replace(" ", "_")
            spec_path = convergence_dir / f"{safe}_spectrum.json"
            spectrum = [
                {"n_mode": int(n), "Rup": float(r)}
                for n, r in zip(modes, vals)
            ]
            spec_path.write_text(
                json.dumps(spectrum, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )

        # --- Mise à jour du lien de téléchargement pour le master JSON ---
        rel = os.path.relpath(str(master_path), start=os.getcwd())
        # Préfixe Jupyter
        href = "/files/" + rel.replace(os.sep, "/")
        download_link.value = (
            f'<a href="{href}" download '
            f'style="text-decoration:none; padding:6px 12px; '
            f'background-color:#007bff; color:white; border-radius:4px; '
            f'font-weight:bold;">'
            f'Download convergence_results.json</a>'
        )


    
    plot_button.on_click(Conv_computation)
    
    conv_controls = widgets.VBox([numeric_controls, button_container])
    widget = widgets.VBox([
        conv_config_selector_box,
        conv_controls,
        conv_output,
        results_table_output,
        download_link
    ], layout=widgets.Layout(width='53%', justify_content='flex-end'))

    
    return widget
