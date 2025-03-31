#!/usr/bin/env python3
# plot_combined_Rup_vs_exp.py

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from data_readers import read_all_combos, read_experimental_data

def plot_combined_Rup_vs_exp(sim_summary_file, exp_file, exp_file2):
    """
    Utilise read_all_combos pour extraire les spectres simulés depuis le fichier simulation_summary_XXX.txt,
    et lit les données expérimentales depuis exp_file et exp_file2 en utilisant read_experimental_data.
    Trace ensuite les courbes simulées et expérimentales sur le même graphique.
    
    Paramètres :
      - sim_summary_file : chemin vers le fichier simulation_summary (issu du sous-dossier de notebooks, par exemple Summary_Simulation)
      - exp_file : chemin vers le premier fichier de données expérimentales
      - exp_file2 : chemin vers le second fichier de données expérimentales
    """
    # Extraction des spectres simulés
    combos = read_all_combos(sim_summary_file)
    
    # Création de la figure
    plt.figure(figsize=(10, 6))
    
    # Tracé des courbes simulées
    for combo_name, (wavelengths, Rup_values) in combos.items():
        plt.plot(wavelengths, Rup_values, '-', linewidth=2, label=combo_name)
    
    # Lecture et tracé des courbes expérimentales
    exp_wl, exp_R = read_experimental_data(exp_file)
    exp_wl2, exp_R2 = read_experimental_data(exp_file2)
    plt.plot(exp_wl, exp_R, linewidth=2, label="Experimental Rup S1")
    plt.plot(exp_wl2, exp_R2, linewidth=2, label="Experimental Rup S2")
    
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance (Rup)")
    plt.title("Reflectance: Simulation vs Experimental")
    plt.legend()
    plt.grid(True)
    
    # Sauvegarde de la figure dans le dossier Figures
    module_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(module_dir)  # Workspace est le parent de modules
    figures_dir = os.path.join(workspace_dir, "Figures")
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = os.path.join(figures_dir, f"combined_Rup_{timestamp}.png")
    plt.savefig(fig_path, bbox_inches='tight')
    print(f"Figure saved in: {fig_path}")

if __name__ == "__main__":
    # Exemple d'utilisation : adaptez les chemins ci-dessous selon votre structure réelle.
    # Les données se trouvent dans le sous-dossier "Summary_Simulation" du dossier notebooks.
    sim_summary_file = "/chemin/vers/notebooks/Summary_Simulation/simulation_summary_exemple.txt"
    exp_file = "/chemin/vers/notebooks/Summary_Simulation/Data_structure1.txt"
    exp_file2 = "/chemin/vers/notebooks/Summary_Simulation/Data_structure2.txt"
    plot_combined_Rup_vs_exp(sim_summary_file, exp_file, exp_file2)
