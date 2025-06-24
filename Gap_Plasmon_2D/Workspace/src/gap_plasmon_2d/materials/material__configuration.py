from gap_plasmon_2d import paths
# Material_Configuration.py
import os
import re
import json
import numpy as np
from gap_plasmon_2d.models.functions__models__data import get_material_permittivity

# Expression régulière pour identifier une expression numérique
numeric_expr_pattern = re.compile(r'^[\d\.\+\-\*\/\s\(\)]+$')

workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(str(paths.DATA_DIR))

def build_material_configuration_dynamic(df_config, lambda_val_nm, json_combined_path, ri_overrides=None):
    """
    Construit un dictionnaire de permittivités {role: ε} à partir d'une configuration (DataFrame).
    Pour chaque rôle, le dictionnaire associé à "material" doit contenir une clé "type" :
      - "None" : retourne 1.0.
      - "RefractiveIndex" : extrait n et k via le module refractiveindexINFO.
      - "Standard" : utilise get_material_permittivity pour obtenir la permittivité.
      - "Custom" : évalue une expression numérique.
    """
    if ri_overrides is None:
        ri_overrides = {}

    with open(json_combined_path, "r", encoding="utf-8") as f:
        materials_data = json.load(f)
    available_materials = {k.lower(): k for k in materials_data.keys()}
    
    materials_perm = {}
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir_local = os.path.join(str(paths.DATA_DIR))

    for _, row in df_config.iterrows():
        role = row["key"]
        mat_dict = row["material"] if isinstance(row["material"], dict) else {"type": str(row["material"]).strip()}
        mtype = mat_dict.get("type", "").lower()

        if mtype == "none":
            materials_perm[role] = 1.0

        elif mtype == "refractiveindex":

            shelf = mat_dict.get("shelf", "").strip()
            book  = mat_dict.get("book", "").strip()
            page  = mat_dict.get("page", "").strip()
            data_field = mat_dict.get("data", "").strip()

            
            if data_field:
                filename = os.path.join(data_dir_local, data_field)
                if not os.path.exists(filename):
                    raise ValueError(f"Le fichier spécifié dans 'data' n'existe pas : {filename}")
            else:
                from gap_plasmon_2d.materials.refractiveindex_info import RefractiveIndex
                RI_instance = RefractiveIndex()
                filename = RI_instance.getMaterialFilename(shelf, book, page)
                if not filename:
                    raise ValueError(f"Impossible de trouver le fichier pour shelf '{shelf}', book '{book}', page '{page}'.")
            from gap_plasmon_2d.materials.refractiveindex_info import Material, NoExtinctionCoefficient
            mat_instance = Material(filename)
            try:
                n = mat_instance.getRefractiveIndex(lambda_val_nm)
                k = mat_instance.getExtinctionCoefficient(lambda_val_nm)

            except NoExtinctionCoefficient:
                    n = mat_instance.getRefractiveIndex(lambda_val_nm)
                    k = 0.0

            except Exception as e:
                    raise ValueError(f"Erreur lors de la récupération des indices depuis {filename} : {e}")    

            eps = (n + 1j * k) ** 2
            materials_perm[role] = eps

        elif mtype == "standard":
            mat_str = mat_dict.get("material", "").strip()
            # Utilisation centralisée dans Functions_Models_Data
            perm = get_material_permittivity(mat_str, lambda_val_nm, json_combined_path, data_dir_local)
            materials_perm[role] = perm

        elif mtype == "custom":
            expr = mat_dict.get("expression", "").strip()
            try:
                const_val = eval(expr, {"__builtins__": {}}, {})
                materials_perm[role] = const_val
            except Exception as e:
                raise ValueError(f"Erreur lors de l'évaluation de l'expression personnalisée '{expr}' : {e}")
        else:
            raise ValueError(f"Type de matériau invalide '{mtype}' pour le rôle '{role}'.")

    return materials_perm
