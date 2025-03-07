# simulate_and_plot.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simulate_reflectance import simulate_reflectance

def run_simulation(lambda_range, n_mod, geometry, wave, materials_config, json_path):
    """
    Exécute la simulation de réflectance sur une plage de longueurs d'onde et affiche le résultat
    avec un tableau récapitulatif des paramètres géométriques et de la configuration matériaux.
    
    Paramètres
    ----------
    lambda_range : array_like
        Plage de longueurs d'onde (en nm).
    n_mod : int
        Nombre de modes RCWA.
    geometry : dict
        Dictionnaire définissant la géométrie du système.
    wave : dict
        Dictionnaire des paramètres d'onde (angle, polarization, etc.).
    materials_config : DataFrame
        Configuration matériaux (issu des dropdowns, MATERIALS_CONFIG).
    json_path : str
        Chemin vers le fichier JSON pour les données ExpData.
    
    Retourne
    --------
    Rup_values, Rdown_values : listes
        Valeurs de réflectance calculées pour chaque longueur d'onde.
    """
    # Exécuter la simulation via la fonction existante
    Rup_values, Rdown_values = simulate_reflectance(lambda_range, geometry, wave, materials_config, json_path, n_mod)
    
    # Création du plot de réflectance
    plt.figure(figsize=(10,6))
    plt.plot(lambda_range, Rup_values, 'o-', label='Rup')
    plt.plot(lambda_range, Rdown_values, 's-', label='Rdown')
    plt.xlabel("Longueur d'onde (nm)")
    plt.ylabel("Réflectance")
    plt.legend()
    plt.title("Simulation de réflectance")
    plt.grid(True)
    
    # Préparer le tableau récapitulatif pour la géométrie
    geom_df = pd.DataFrame(list(geometry.items()), columns=['Paramètre Géométrique', 'Valeur'])
    # Préparer le tableau pour la configuration matériaux
    mat_df = materials_config.copy()  # colonnes: key et material
    
    cellText_geom = geom_df.values.tolist()
    cellText_mat = mat_df.values.tolist()
    
    # Afficher le tableau géométrique en bas à gauche
    table_geom = plt.table(cellText=cellText_geom, colLabels=geom_df.columns,
                            loc='bottom', bbox=[0, -0.45, 0.5, 0.3])
    table_geom.auto_set_font_size(False)
    table_geom.set_fontsize(8)
    
    # Afficher le tableau des matériaux en bas à droite
    table_mat = plt.table(cellText=cellText_mat, colLabels=mat_df.columns,
                           loc='bottom', bbox=[0.5, -0.45, 0.5, 0.3])
    table_mat.auto_set_font_size(False)
    table_mat.set_fontsize(8)
    
    # Ajouter des titres pour les tableaux
    plt.text(0.25, -0.5, 'Paramètres Géométriques', ha='center', fontsize=13, transform=plt.gca().transAxes)
    plt.text(0.75, -0.5, 'Configuration Matériaux', ha='center', fontsize=13, transform=plt.gca().transAxes)
    
    plt.subplots_adjust(bottom=0.3)
    plt.show()
    
    return Rup_values, Rdown_values
