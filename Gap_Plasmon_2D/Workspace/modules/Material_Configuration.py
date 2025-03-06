# Material_Configuration.py
import pandas as pd
from Functions_ExpData import get_n_k, compute_permittivity
from MaterialsLoader import load_materials, get_material_params

def build_material_configuration_dynamic(df_config, lambda_val, json_path_expdata):
    """
    Construit un dictionnaire de permittivités à partir d'un DataFrame de configuration pour une longueur d'onde donnée.
    
    Pour chaque matériau, si son nom figure dans la liste des matériaux gérés par ExpData 
    (ex. "BK7", "Water", "SiA", "Air"), on utilise get_n_k pour interpoler n et k.
    Sinon, on suppose que le matériau est modélisé par Brendel‑Bormann et on utilise get_material_params 
    et compute_permittivity.
    
    Si l'utilisateur a choisi "None", la clé correspondante recevra la valeur None.
    """
    materials_data = load_materials()
    expdata_materials = ["BK7", "Water", "SiA", "Si", "Air"]  # À adapter si besoin
    materials_perm = {}
    for idx, row in df_config.iterrows():
        key = row['key']
        mat = row['material']
        # Si l'utilisateur a choisi "None", on affecte None
        if mat == "None":
            materials_perm[key] = None
        elif mat in expdata_materials:
            n_val, k_val = get_n_k(mat, lambda_val, json_path_expdata)
            perm = (n_val + 1j * k_val)**2
            materials_perm[key] = perm
        else:
            f0, omega_p, Gamma0, f, omega, gamma, sigma, model = get_material_params(mat, materials_data)
            perm = compute_permittivity(lambda_val, f0, omega_p, Gamma0, f, omega, gamma, sigma, N=50)
            materials_perm[key] = perm
    return materials_perm


