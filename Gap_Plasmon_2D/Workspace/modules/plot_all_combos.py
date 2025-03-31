#!/usr/bin/env python3
# plot_all_combos.py

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from data_readers import read_all_combos

def plot_all_combos(sim_summary_file):
    """
    Lit le fichier simulation_summary_XXX.txt, extrait toutes les combinaisons à l'aide de read_all_combos,
    et trace sur le même graphique les courbes simulées (Rup) de chacun des combos.
    """
    combos = read_all_combos(sim_summary_file)
    
    plt.figure(figsize=(10, 6))
    for combo_name, (wavelengths, Rup_values) in combos.items():
        plt.plot(wavelengths, Rup_values, '-', linewidth=2, label=combo_name)
    
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance (Rup)")
    plt.title("Combined Reflectance Spectra for All Combos")
    plt.legend()
    plt.grid(True)
    
    # Sauvegarde dans le dossier Figures
    module_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(module_dir)  # Les modules et notebooks sont dans le même workspace
    figures_dir = os.path.join(workspace_dir, "Figures")
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = os.path.join(figures_dir, f"all_combos_Rup_{timestamp}.png")
    plt.savefig(fig_path, bbox_inches='tight')
    print(f"Figure saved in: {fig_path}")
