#!/usr/bin/env python3
# simulate_and_plot.py

import os
import matplotlib.pyplot as plt
import re

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
    
    # 3) Pour chaque combo, on trace Rup (et éventuellement Rdown)
    for combo_name, (Rup, Rdown) in results.items():
        plt.plot(lambda_range, Rup, label=f'Rup - {combo_name}')
        # plt.plot(lambda_range, Rdown, label=f'Rdown - {combo_name}')

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.title("Reflectance Simulation") # Faire un titre automatique
    plt.legend()
    plt.grid(True)

    # 4) Sauvegarde de la figure dans le dossier Figures
    module_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(module_dir)
    figures_dir = os.path.join(workspace_dir, "Figures")
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # Construit le suffixe du nom de fichier à partir de la liste des matériaux utilisés
    # On prend les clés de results, on les trie et on les joint avec un underscore
    materials_used = "_".join(sorted(results.keys()))
    # Remplace les espaces par des underscores et supprime les caractères indésirables
    materials_used_clean = re.sub(r'\s+', '_', materials_used)
    materials_used_clean = re.sub(r'[^A-Za-z0-9_]', '', materials_used_clean)

    fig_path = os.path.join(figures_dir, f"reflectance_{materials_used_clean}.png")
    plt.savefig(fig_path, bbox_inches='tight')
    plt.show()
    
    print(f"Figure sauvegardée dans : {fig_path}")
    return results
