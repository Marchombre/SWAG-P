# Material_Configuration.py
import json
import pandas as pd
from Functions_ExpData import get_n_k, compute_permittivity
from MaterialsLoader import load_materials, get_material_params

def build_material_configuration_dynamic(df_config, lambda_val, json_path_expdata):
    """
    Construit un dictionnaire de permittivités à partir d'un DataFrame de configuration pour une longueur d'onde donnée.
    
    Pour chaque matériau :
      - Si la valeur est "None" (insensible à la casse), on affecte None.
      - Sinon, si le nom figure dans le fichier JSON ExpData (les clés du fichier),
        on utilise get_n_k pour interpoler n et k et on calcule ε = (n+ik)².
      - Sinon, si le nom figure dans les données BB (chargées via load_materials),
        on utilise get_material_params et compute_permittivity (qui utilise BBFaddeeva).
      - Sinon, on tente de convertir la valeur en float. Si cela réussit,
        on considère que l'utilisateur définit directement une valeur constante de permittivité.
      - Sinon, une erreur est levée.
    """
    # Charger les données du JSON ExpData pour obtenir les matériaux disponibles
    with open(json_path_expdata, 'r') as f:
        expdata_json = json.load(f)
    available_expdata = list(expdata_json.keys())
    
    # Charger les données des matériaux BB via MaterialsLoader
    materials_data = load_materials()
    
    materials_perm = {}
    for idx, row in df_config.iterrows():
        key = row['key']
        mat = row['material'].strip()  # enlever les espaces éventuels
        if mat.lower() == "none":
            materials_perm[key] = None
        elif mat in available_expdata:
            # Traitement via ExpData (interpolation des données)
            n_val, k_val = get_n_k(mat, lambda_val, json_path_expdata)
            perm = (n_val + 1j * k_val) ** 2
            materials_perm[key] = perm
        elif mat in materials_data:
            # Traitement via le modèle BB (Brendel-Bormann)
            f0, omega_p, Gamma0, f, omega, gamma, sigma, model = get_material_params(mat, materials_data)
            perm = compute_permittivity(lambda_val, f0, omega_p, Gamma0, f, omega, gamma, sigma, N=50)
            materials_perm[key] = perm
        else:
            # On essaie d'interpréter le matériau comme une valeur numérique custom
            try:
                const_val = float(mat)
                materials_perm[key] = const_val
            except ValueError:
                raise ValueError(f"Matériau '{mat}' non trouvé dans les données ExpData ou BB, "
                                 "et ne peut être interprété comme une constante.")
    return materials_perm
