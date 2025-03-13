#!/usr/bin/env python3
# simulate_and_plot.py

import os
import matplotlib.pyplot as plt
from datetime import datetime

from simulate_reflectance import simulate_reflectance_all_combos

def run_simulation_all_combos(lambda_range, wave, n_mod, json_combined_path):
    """
    Exécute la simulation de réflectance pour toutes les combinaisons
    géométrie-matériaux (geom_mat_combinations.json) et affiche le résultat
    sur un seul graphique, avec une légende pour chaque combo.
    
    Paramètres
    ----------
    lambda_range : array_like
        Plage de longueurs d'onde (en nm).
    wave : dict
        Paramètres de l'onde (angle, polarisation, etc.).
    n_mod : int
        Nombre de modes RCWA.
    json_combined_path : str
        Chemin vers le fichier JSON combiné contenant les données ExpData/BrendelBormann.

    Retourne
    -------
    results : dict
        Dictionnaire {combo_name: (Rup_values, Rdown_values)} renvoyé par simulate_reflectance_all_combos.
    """
    # 1) Appel de la simulation multi-combos
    results = simulate_reflectance_all_combos(lambda_range, wave, n_mod, json_combined_path)
    
    # 2) Création d'une figure
    plt.figure(figsize=(10, 6))
    
    # 3) Pour chaque combo, on trace Rup et Rdown
    for combo_name, (Rup, Rdown) in results.items():
        plt.plot(lambda_range, Rup, label=f'Rup - {combo_name}')
        plt.plot(lambda_range, Rdown, label=f'Rdown - {combo_name}')

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.title("Reflectance Simulation - All Geometry/Material Combos")
    plt.legend()
    plt.grid(True)

    # 4) Sauvegarde
    module_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(module_dir)
    figures_dir = os.path.join(workspace_dir, "Figures")
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = os.path.join(figures_dir, f"reflectance_combos_{timestamp}.png")
    plt.savefig(fig_path, bbox_inches='tight')
    plt.show()
    
    print(f"Figure sauvegardée dans : {fig_path}")
    return results
