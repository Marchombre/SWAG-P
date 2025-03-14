#!/usr/bin/env python3
# plot_all_combos.py

import os
import re
import ast
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def read_all_combos(file_path):
    """
    Lit le fichier simulation_summary_XXX.txt et extrait pour chaque combo (délimité par "Combo name:")
    les points de réflectance Rup. La plage de longueurs d'onde est déduite des lignes de points
    de réflectance du premier combo rencontré.
    
    Retourne un dictionnaire dont les clés sont les noms de combo et les valeurs sont un tuple
    (wavelengths, Rup_values) sous forme de tableaux numpy.
    """
    combos = {}
    lambda_range = None  # sera défini à partir des longueurs d'onde du premier combo
    current_combo = None
    current_wavelengths = []
    current_Rup = []
    reading_points = False

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        # Détection d'un nouveau combo
        if line.startswith("Combo name:"):
            # Si un combo est en cours de lecture, le sauvegarder
            if current_combo is not None and current_Rup:
                if lambda_range is None:
                    lambda_range = np.array(current_wavelengths)
                combos[current_combo] = (lambda_range, np.array(current_Rup))
            current_combo = line.split("Combo name:")[1].strip()
            current_wavelengths = []
            current_Rup = []
            reading_points = False
            continue

        # Déclenchement de la lecture des points lorsque la section "Reflectance points" est trouvée
        if "Reflectance points" in line:
            reading_points = True
            continue

        # Extraction des points de réflectance
        if reading_points and line.startswith("λ="):
            # Exemple attendu : "λ=450.0 nm -> Rup=0.2779394076767168, Rdown=0.2424285804027109"
            m = re.search(r"λ=([\d\.]+)\s*nm\s*->\s*Rup=([\d\.eE\+\-]+)", line)
            if m:
                try:
                    wl_val = float(m.group(1))
                    rup_val = float(m.group(2))
                    current_wavelengths.append(wl_val)
                    current_Rup.append(rup_val)
                except Exception:
                    continue
        # Arrêter la lecture si une ligne de séparation est rencontrée
        if reading_points and re.match(r"^-{10,}", line):
            reading_points = False

    # Sauvegarder le dernier combo lu
    if current_combo is not None and current_Rup:
        if lambda_range is None:
            lambda_range = np.array(current_wavelengths)
        combos[current_combo] = (lambda_range, np.array(current_Rup))
    
    if not combos:
        raise ValueError("Aucun combo n'a pu être extrait du fichier.")
    return combos

def plot_all_combos(sim_summary_file):
    """
    Lit le fichier simulation_summary_XXX.txt, extrait toutes les combinaisons,
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
    
    # Sauvegarde de la figure dans le dossier Figures
    module_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(module_dir)
    figures_dir = os.path.join(workspace_dir, "Figures")
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = os.path.join(figures_dir, f"all_combos_Rup_{timestamp}.png")
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
    
    plot_all_combos(sim_summary_file)
