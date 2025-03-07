# MaterialsConfiguration.py
import json
import pandas as pd
from Functions_ExpData import get_n_k, compute_permittivity

def load_combined_materials(json_path):
    """
    Charge le fichier JSON combiné contenant les données ExpData et BrendelBormann.
    
    Retourne un dictionnaire des matériaux.
    """
    with open(json_path, 'r') as f:
        materials_data = json.load(f)
    return materials_data

def get_material_params(material_name, materials_data):
    """
    Extrait les paramètres pour un matériau donné depuis le dictionnaire materials_data.
    
    Retourne un tuple : (f0, omega_p, Gamma0, f, omega, gamma, sigma, model).
    """
    if material_name in materials_data:
        material = materials_data[material_name]
        try:
            f0 = material["f0"]
            omega_p = material["omega_p"]
            Gamma0 = material["Gamma0"]
            f = material["f"]
            omega = material["omega"]
            gamma = material["Gamma"]
            sigma = material["sigma"]
            model = material.get("model", "").lower()
            return f0, omega_p, Gamma0, f, omega, gamma, sigma, model
        except KeyError as e:
            raise ValueError(f"Les paramètres pour '{material_name}' ne sont pas complets dans le fichier JSON combiné: {e}")
    else:
        raise ValueError(f"Matériau '{material_name}' non trouvé dans le fichier JSON combiné.")

def build_material_configuration_dynamic(df_config, lambda_val, json_path):
    """
    Construit un dictionnaire de permittivités à partir d'un DataFrame de configuration pour une longueur d'onde donnée,
    en chargeant un fichier JSON combiné contenant les données ExpData et BrendelBormann.
    
    Pour chaque matériau dans le DataFrame (colonnes "key" et "material") :
      - Si la valeur est "None" (insensible à la casse), on affecte 1.0 (équivalent à l'air).
      - Sinon, si le matériau figure dans le fichier JSON combiné (comparaison insensible à la casse) et que son
        champ "model" vaut "expdata", on utilise get_n_k pour interpoler n et k et on calcule ε = (n+ik)².
      - Sinon, si le matériau figure dans le fichier JSON combiné et que son "model" est autre (ex. "brendelbormann"),
        on utilise get_material_params puis compute_permittivity.
      - Sinon, on tente de convertir la valeur en float (pour une constante custom).
      - Sinon, une erreur est levée.
    """
    materials_data = load_combined_materials(json_path)
    # Construire un dictionnaire pour la recherche insensible à la casse
    available_materials = {k.lower(): k for k in materials_data.keys()}
    
    materials_perm = {}
    for idx, row in df_config.iterrows():
        key = row['key']
        mat = row['material'].strip()
        mat_lower = mat.lower()
        if mat_lower == "none":
            materials_perm[key] = 1.0
        elif mat_lower in available_materials:
            actual_mat = available_materials[mat_lower]
            material = materials_data[actual_mat]
            model = material.get("model", "").lower()
            if model == "expdata":
                # Traitement via ExpData
                n_val, k_val = get_n_k(actual_mat, lambda_val, json_path)
                perm = (n_val + 1j * k_val) ** 2
                materials_perm[key] = perm
            else:
                # Traitement via le modèle BB
                try:
                    f0, omega_p, Gamma0, f, omega, gamma, sigma, model = get_material_params(actual_mat, materials_data)
                    perm = compute_permittivity(lambda_val, f0, omega_p, Gamma0, f, omega, gamma, sigma, N=50)
                    materials_perm[key] = perm
                except KeyError as e:
                    raise ValueError(f"Les paramètres pour '{actual_mat}' ne sont pas complets: {e}")
        else:
            try:
                const_val = float(mat)
                materials_perm[key] = const_val
            except ValueError:
                raise ValueError(f"Matériau '{mat}' non trouvé dans le fichier JSON combiné, "
                                 "et ne peut être interprété comme une constante.")
    return materials_perm
