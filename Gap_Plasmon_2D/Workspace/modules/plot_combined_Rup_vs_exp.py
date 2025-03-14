#!/usr/bin/env python3
# plot_combined_Rup_vs_exp.py

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# On importe la fonction de lecture des données simulées depuis le module plot_all_combos.py
from plot_all_combos import read_all_combos

def read_experimental_data(file_path):
    """
    Lit le fichier de données expérimentales (Data_structure1.txt) qui contient
    un en-tête suivi des données sous le format :
    
      Wavelengths (nm)     R
      450.0, 0.2654618528289272
      452.7638190954774, 0.2701075857791835
      ...
      
    Les lignes d'en-tête (ne contenant pas de virgule) sont ignorées.
    Retourne (wavelengths, R_values) sous forme de tableaux numpy.
    """
    wavelengths = []
    R_values = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ',' not in line:
                continue
            parts = line.split(',')
            try:
                wl = float(parts[0].strip())
                R_val = float(parts[1].strip())
                wavelengths.append(wl)
                R_values.append(R_val)
            except Exception:
                continue
    if not wavelengths or not R_values:
        raise ValueError("Aucune donnée expérimentale n'a été trouvée dans le fichier.")
    return np.array(wavelengths), np.array(R_values)

def plot_combined_Rup_vs_exp(sim_summary_file, exp_file):
    """
    Utilise la fonction read_all_combos du module plot_all_combos pour extraire
    les spectres simulés depuis le fichier simulation_summary_XXX.txt, et lit
    les données expérimentales depuis exp_file. Les deux spectres sont tracés
    sur le même graphique.
    """
    # Extraction des données simulées via le module plot_all_combos
    combos = read_all_combos(sim_summary_file)
    
    plt.figure(figsize=(10, 6))
    
    # Tracé des courbes simulées pour chaque combo
    for combo_name, (wavelengths, Rup_values) in combos.items():
        plt.plot(wavelengths, Rup_values, '-', linewidth=2, label=combo_name)
    
    # Extraction des données expérimentales
    exp_wl, exp_R = read_experimental_data(exp_file)
    
    # Tracé du spectre expérimental
    plt.plot(exp_wl, exp_R, linewidth=2, label="Experimental Rup")
    
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance (Rup)")
    plt.title("Reflectance Comparison: Simulation vs Experimental")
    plt.legend()
    plt.grid(True)
    
    # Sauvegarde de la figure dans le dossier Figures
    module_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(module_dir)
    figures_dir = os.path.join(workspace_dir, "Figures")
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = os.path.join(figures_dir, f"combined_Rup_{timestamp}.png")
    plt.savefig(fig_path, bbox_inches='tight')
    plt.show()
    print(f"Figure saved in: {fig_path}")

if __name__ == "__main__":
    # Exemple d'utilisation : adaptez les chemins selon votre arborescence
    module_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(module_dir)
    notebooks_dir = os.path.join(workspace_dir, "notebooks")
    summary_dir = os.path.join(notebooks_dir, "Summary_Simulation")
    sim_summary_file = os.path.join(summary_dir, "simulation_summary_20250314_135436.txt")
    exp_file = os.path.join(summary_dir, "Data_structure1.txt")
    
    plot_combined_Rup_vs_exp(sim_summary_file, exp_file)
