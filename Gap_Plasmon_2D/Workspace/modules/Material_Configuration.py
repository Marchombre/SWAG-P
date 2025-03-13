import os
import re
import json
import numpy as np

from Functions_Models_Data import get_n_k, compute_permittivity

# Expression régulière pour identifier une expression numérique
numeric_expr_pattern = re.compile(r'^[\d\.\+\-\*\/\s\(\)]+$')

# Définition des chemins
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(workspace_dir, "data")

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
            gamma = material["Gamma"]   # Remarquez que la clé est "Gamma" dans le JSON
            sigma = material["sigma"]
            model = material.get("model", "").lower()
            return f0, omega_p, Gamma0, f, omega, gamma, sigma, model
        except KeyError as e:
            raise ValueError(f"Les paramètres pour '{material_name}' sont incomplets dans le fichier JSON combiné : {e}")
    else:
        raise ValueError(f"Le matériau '{material_name}' n'est pas présent dans le fichier JSON combiné.")

def build_material_configuration_dynamic(df_config, lambda_val_nm, json_combined_path, ri_overrides=None):
    """
    Construit un dictionnaire de permittivités {role: ε} à partir d'une configuration (DataFrame).
    
    Pour chaque rôle, le DataFrame doit contenir une ligne dont la colonne "material" est un dictionnaire
    avec au moins la clé "type" (possibles valeurs : "None", "RefractiveIndex", "Standard" ou "Custom").
    
    - "None": retourne 1.0.
    - "RefractiveIndex": le dictionnaire doit contenir "shelf", "book", "page" et éventuellement "data"
         (chemin relatif vers le fichier YAML). Le module `refractiveindexINFO` est utilisé pour extraire
         l'indice n et le coefficient k, puis on calcule ε = (n + 1j*k)².
    - "Standard": le dictionnaire doit contenir "material" (la chaîne indiquant le matériau standard).
         Si le matériau se trouve dans le fichier JSON combiné, alors :
           - pour le modèle "expdata", on utilise get_n_k pour obtenir n et k et calcule ε = (n+1j*k)²,
           - sinon on utilise compute_permittivity.
         Si le matériau n'est pas présent dans le JSON, on recherche un fichier texte dans data_dir.
    - "Custom": le dictionnaire doit contenir "expression" (une expression numérique à évaluer).
    
    Retourne un dictionnaire {role: ε}.
    """
    if ri_overrides is None:
        ri_overrides = {}

    # Chargement des données de matériaux depuis le fichier JSON combiné
    with open(json_combined_path, "r", encoding="utf-8") as f:
        materials_data = json.load(f)
    
    # Construction d'un dictionnaire de recherche insensible à la casse
    available_materials = {k.lower(): k for k in materials_data.keys()}
    materials_perm = {}

    # Chemin local pour les fichiers YAML et les fichiers texte
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir_local = os.path.join(base_path, "data")

    for _, row in df_config.iterrows():
        role = row["key"]
        # On s'assure que "material" est un dictionnaire (issu de notre MaterialSelector)
        mat_dict = row["material"] if isinstance(row["material"], dict) else {"type": str(row["material"]).strip()}
        mtype = mat_dict.get("type", "").lower()

        if mtype == "none":
            materials_perm[role] = 1.0

        elif mtype == "refractiveindex":
            shelf = mat_dict.get("shelf", "").strip()
            book  = mat_dict.get("book", "").strip()
            page  = mat_dict.get("page", "").strip()
            data_field = mat_dict.get("data", "").strip()  # chemin relatif vers le fichier YAML

            if not shelf or not book or not page:
                materials_perm[role] = 1.0
            else:
                if data_field:
                    filename = os.path.join(data_dir_local, data_field)
                    if not os.path.exists(filename):
                        raise ValueError(f"Le fichier spécifié dans 'data' n'existe pas : {filename}")
                else:
                    from refractiveindexINFO import RefractiveIndex
                    RI_instance = RefractiveIndex()
                    filename = RI_instance.getMaterialFilename(shelf, book, page)
                    if not filename:
                        raise ValueError(f"Impossible de trouver le fichier pour shelf '{shelf}', book '{book}', page '{page}'.")
                
                from refractiveindexINFO import Material, NoExtinctionCoefficient
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
            # Dans ce mode, on attend que mat_dict contienne la clé "material"
            mat_str = mat_dict.get("material", "").strip()
            mat_lower = mat_str.lower()
            if mat_lower in available_materials:
                actual_mat = available_materials[mat_lower]
                material = materials_data[actual_mat]
                model = material.get("model", "").lower()
                if model == "expdata":
                    n_val, k_val = get_n_k(actual_mat, lambda_val_nm, json_combined_path)
                    materials_perm[role] = (n_val + 1j * k_val) ** 2
                else:
                    try:
                        f0, omega_p, Gamma0, f, omega, gamma, sigma, _ = get_material_params(actual_mat, materials_data)
                        perm = compute_permittivity(lambda_val_nm, f0, omega_p, Gamma0, f, omega, gamma, sigma, N=50)
                        materials_perm[role] = perm
                    except KeyError as e:
                        raise ValueError(f"Les paramètres pour '{actual_mat}' sont incomplets : {e}")
            else:
                # Recherche d'un fichier texte correspondant à mat_str dans data_dir_local
                import glob
                pattern = os.path.join(data_dir_local, f"{mat_str}.txt")
                txt_files = glob.glob(pattern)
                if not txt_files:
                    pattern = os.path.join(data_dir_local, f"*{mat_str}*.txt")
                    txt_files = glob.glob(pattern)
                    if not txt_files:
                        raise ValueError(f"Le matériau standard '{mat_str}' n'a pas été trouvé dans le JSON ni dans data_dir_local.")
                txt_file = txt_files[0]
                try:
                    # Lecture du fichier texte (attendu avec les colonnes 'wl', 'n' et 'k')
                    data = np.genfromtxt(txt_file, delimiter=None, names=True)
                    required_cols = ['wl', 'n', 'k']
                    if not all(col in data.dtype.names for col in required_cols):
                        raise ValueError(f"Le fichier {txt_file} doit contenir les colonnes {required_cols}.")
                    # Conversion de lambda_val_nm (nm) en micromètres
                    lambda_val_um = lambda_val_nm * 0.001
                    n_val = np.interp(lambda_val_um, data['wl'], data['n'])
                    k_val = np.interp(lambda_val_um, data['wl'], data['k'])
                    materials_perm[role] = (n_val + 1j * k_val) ** 2
                except Exception as e:
                    raise ValueError(f"Erreur lors de la lecture du fichier {txt_file} pour le matériau '{mat_str}' : {e}")

        elif mtype == "custom":
            expr = mat_dict.get("expression", "").strip()
            try:
                # Évaluation sécurisée de l'expression numérique
                const_val = eval(expr, {"__builtins__": {}}, {})
                materials_perm[role] = const_val
            except Exception as e:
                raise ValueError(f"Erreur lors de l'évaluation de l'expression personnalisée '{expr}' : {e}")
        else:
            raise ValueError(f"Type de matériau invalide '{mtype}' pour le rôle '{role}'.")

    return materials_perm
