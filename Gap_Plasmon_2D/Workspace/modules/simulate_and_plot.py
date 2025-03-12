import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from simulate_reflectance import simulate_reflectance  # Assurez-vous que ce module est accessible

# Chemin Workspace (supposé être le parent du dossier courant)
workspace_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))

# Répertoire pour sauvegarder les figures
figures_dir = os.path.join(workspace_dir, "Figures")
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

def run_simulation(lambda_range, n_mod, geometry, wave, materials_config, json_path):
    """
    Exécute la simulation de réflectance sur une plage de longueurs d'onde et affiche le résultat,
    ainsi qu'un tableau récapitulatif des paramètres géométriques et de la configuration des matériaux.
    
    Parameters
    ----------
    lambda_range : array_like
        Plage de longueurs d'onde (en nm).
    n_mod : int
        Nombre de modes RCWA.
    geometry : dict
        Dictionnaire définissant la géométrie du système.
    wave : dict
        Dictionnaire des paramètres de l'onde (angle, polarisation, etc.).
    materials_config : DataFrame
        Configuration des matériaux (issue du widget MATERIALS_CONFIG).  
        Chaque ligne contient deux colonnes : "key" et "material".  
        Pour un matériau de type "RefractiveIndex", le dictionnaire peut contenir les clés "shelf", "book", "page" et "data" (chemin vers le fichier YAML).
    json_path : str
        Chemin vers le fichier JSON combiné contenant les données ExpData.
    
    Returns
    -------
    Rup_values, Rdown_values : lists
        Valeurs de réflectance calculées pour chaque longueur d'onde.
    """
    # Exécute la simulation en passant la configuration des matériaux
    Rup_values, Rdown_values = simulate_reflectance(lambda_range, geometry, wave, materials_config, json_path, n_mod)
    
    # Générer un nom de fichier avec timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = os.path.join(figures_dir, f"reflectance_simulation_{timestamp}.png")
    
    # Création de la figure
    plt.figure(figsize=(10, 6))
    plt.plot(lambda_range, Rup_values, label='Rup')
    plt.plot(lambda_range, Rdown_values, label='Rdown')
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.title("Reflectance Simulation")
    plt.legend()
    plt.grid(True)
    
    # Préparation des tableaux récapitulatifs
    geom_df = pd.DataFrame(list(geometry.items()), columns=['Geometric Parameter', 'Value'])
    # On convertit la colonne "material" en chaîne pour un affichage plus lisible
    mat_df = materials_config.copy()
    mat_df['material'] = mat_df['material'].apply(lambda d: str(d))
    
    cellText_geom = geom_df.values.tolist()
    cellText_mat = mat_df.values.tolist()
    
    # Ajout des tableaux dans la figure
    table_geom = plt.table(cellText=cellText_geom, colLabels=geom_df.columns,
                           loc='bottom', bbox=[0, -0.45, 0.5, 0.3])
    table_geom.auto_set_font_size(False)
    table_geom.set_fontsize(8)
    
    table_mat = plt.table(cellText=cellText_mat, colLabels=mat_df.columns,
                          loc='bottom', bbox=[0.5, -0.45, 0.5, 0.3])
    table_mat.auto_set_font_size(False)
    table_mat.set_fontsize(8)
    
    # Ajout des titres pour les tableaux
    plt.text(0.25, -0.5, 'Geometric Parameters', ha='center', fontsize=13, transform=plt.gca().transAxes)
    plt.text(0.75, -0.5, 'Materials Configuration', ha='center', fontsize=13, transform=plt.gca().transAxes)
    
    plt.subplots_adjust(bottom=0.3)
    
    # Sauvegarde et affichage de la figure
    plt.savefig(fig_path, bbox_inches='tight')
    plt.show()
    
    return Rup_values, Rdown_values
