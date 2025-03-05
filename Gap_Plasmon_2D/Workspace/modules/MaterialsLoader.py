import json
import os
import numpy as np

# Définition du chemin vers le fichier JSON
JSON_PATH = "/home/chardon-grossard/Bureau/SWAG-P/Gap_Plasmon_2D/Workspace/data/BB_materials.json"

def load_materials():
    """
    Charge les paramètres des matériaux depuis le fichier JSON.
    """
    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError(f"Fichier JSON introuvable : {JSON_PATH}")
    
    with open(JSON_PATH, "r") as file:
        materials_data = json.load(file)
    
    return materials_data

def get_material_params(material_name, materials_data):
    """
    Récupère les paramètres d'un matériau donné depuis le JSON.
    """
    if material_name not in materials_data:
        raise ValueError(f"Matériau '{material_name}' non trouvé dans le fichier JSON.")

    material = materials_data[material_name]

    f0 = material["f0"]
    Gamma0 = material["Gamma0"]
    omega_p = material["omega_p"]
    
    # Convertir les listes en numpy arrays
    f = np.array(material["f"])
    omega = np.array(material["omega"])
    gamma = np.array(material["Gamma"])
    sigma = np.array(material["sigma"])
    
    return f0, omega_p, Gamma0, f, omega, gamma, sigma
