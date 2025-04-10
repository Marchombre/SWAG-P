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

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

import ipywidgets as widgets
from IPython.display import display

from simulate_reflectance import simulate_reflectance_single

def compute_convergence(lambda_fixed, n_mode_max, geometry, wave, df_config, json_combined_path, ri_overrides, tolerance=1e-3, n_step=1, progress_bar=None):
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
        tolerance (float): Seuil de variation pour définir la convergence (par défaut 1e-3).
        n_step (int): Pas de variation pour n_mode (par défaut 1).
        progress_bar (ipywidgets widget): Optionnel, barre de progression à mettre à jour.
    
    Returns:
        n_modes (np.array): Tableau des valeurs de n_mode testées.
        Rup_vals (list): Liste des valeurs de Rup calculées pour chaque n_mode.
        optimal_n_mode (int): Valeur optimale de n_mode, c'est-à-dire le premier n_mode
                              pour lequel la différence avec la valeur précédente est < tolerance.
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
        # Mise à jour de la barre de progression globale si fournie (incrémentation cumulative)
        if progress_bar is not None:
            progress_bar.value += 1
            # Court délai pour permettre à l'interface de se rafraîchir
            time.sleep(0.01)
    
    # Calcul des différences successives
    diffs = np.abs(np.diff(Rup_vals))
    optimal_n_mode = n_mode_max
    for i, d in enumerate(diffs):
        if d < tolerance:
            optimal_n_mode = int(n_modes[i+1])
            break
    
    return n_modes, Rup_vals, optimal_n_mode




# --- Widget de convergence ---
def create_multi_convergence_widget(json_combined_path, all_configs):
    """
    Crée un widget pour tracer la convergence de Rup en fonction de n_mode pour une ou plusieurs configurations.
    La partie haute comporte :
      - Un sélecteur de configuration.
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

    # Widgets numériques pour saisir les paramètres de convergence.
    lambda_fixed_widget = widgets.FloatText(
        value=700.0, 
        description="λ fixe (nm):", 
        layout=widgets.Layout(width='150px')
    )
    n_mode_max_widget = widgets.IntText(
        value=200, 
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
    row1 = widgets.HBox([lambda_fixed_widget, n_mode_max_widget])
    row2 = widgets.HBox([n_mode_step_widget, tolerance_widget])
    numeric_controls = widgets.VBox([row1, row2])

    # Création du bouton "Tracer convergence"
    plot_button = widgets.Button(
        description="Run convergence", 
        button_style="primary", 
        layout=widgets.Layout(width="150px")
    )
    
    # Création de la barre de progression globale.
    overall_progress_bar = widgets.IntProgress(
        value=0, 
        min=0, 
        max=1,  # La valeur max sera définie lors du déclenchement du calcul
        description='Progress:',
        layout=widgets.Layout(width='400px')
    )
    
    # Création d'un "spacer" pour séparer la barre et le bouton dans la même ligne.
    spacer = widgets.HBox([], layout=widgets.Layout(flex='1 1 auto'))
    
    # Création du container qui positionne la barre de progression à gauche et le bouton à droite.
    button_container = widgets.HBox(
        [overall_progress_bar, spacer, plot_button],
        layout=widgets.Layout(width='100%')
    )
    
    # Widget de sortie pour le tracé de convergence.
    conv_output = widgets.Output(
        layout=widgets.Layout(border="1px solid lightgray", width='630px', height='400px')
    )

    # Widget HTML pour afficher le n_mode optimal.
    optimal_label = widgets.HTML(
        value="",
        layout=widgets.Layout(width='450px')
    )

    def Conv_computation(b):
        # Efface les sorties précédentes
        conv_output.clear_output()
        optimal_label.value = ""
        
        # Récupération des paramètres saisis
        lambda_fixed = lambda_fixed_widget.value
        n_mode_max   = n_mode_max_widget.value
        n_mode_step  = n_mode_step_widget.value
        tol          = tolerance_widget.value
        
        selected_configs = conv_config_selector.value
        if not selected_configs:
            with conv_output:
                print("Please select one or more configuration before computation.")
            return
        
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        fig, ax = plt.subplots(figsize=(7.5, 4))
        optimal_texts = []
        
        # Calcul du nombre total d'itérations pour la barre de progression globale
        iterations_per_config = len(np.arange(1, n_mode_max + 1, n_mode_step))
        overall_progress_bar.max = len(selected_configs) * iterations_per_config
        overall_progress_bar.value = 0

        # Pour chaque configuration, calculer et tracer la convergence en utilisant la barre de progression globale.
        for idx, cfg in enumerate(selected_configs):
            geometry_cfg = cfg["geometry"]["geometry"]
            material_config_list = cfg["material"]["MATERIALS_CONFIG"]
            df_config = pd.DataFrame(material_config_list)
            ri_overrides = cfg["material"].get("RI_OVERRIDES", {})
            
            # Passage de la barre de progression globale pour mise à jour cumulative
            n_modes, Rup_vals, optimal_n_mode = compute_convergence(
                lambda_fixed, n_mode_max, geometry_cfg, {"angle": 0, "polarization": 1},
                df_config, json_combined_path, ri_overrides, tolerance=tol, n_step=n_mode_step,
                progress_bar=overall_progress_bar
            )
            
            color = colors[idx % len(colors)]
            ax.plot(n_modes, Rup_vals, marker='o', label=cfg["config_name"], color=color)
            optimal_texts.append(f'<span style="color:{color};">{cfg["config_name"]}: optimal n_mode = {optimal_n_mode}</span>')
        
        ax.set_xlabel("n_mode")
        ax.set_ylabel("Rup")
        ax.set_title(f"Convergence pour λ = {lambda_fixed} nm", fontsize=10)
        ax.grid(True)
        ax.legend(fontsize=8)
        plt.tight_layout()
        optimal_label.value = " | ".join(optimal_texts)
        
        with conv_output:
            display(fig)
            plt.close(fig)
    
    plot_button.on_click(Conv_computation)
    
    conv_controls = widgets.VBox([numeric_controls, button_container])
    widget = widgets.VBox([conv_config_selector, optimal_label, conv_controls, conv_output], 
                          layout=widgets.Layout(width='53%', justify_content='flex-end'))
    
    return widget
