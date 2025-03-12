import os
import re
import json
import numpy as np
import yaml
import scipy.interpolate

from Functions_ExpData import get_n_k, compute_permittivity
from refractiveindexINFO import RefractiveIndexMaterial, RefractiveIndexData

# Expression régulière pour identifier une expression numérique
numeric_expr_pattern = re.compile(r'^[\d\.\+\-\*\/\s\(\)]+$')

# Définition des chemins
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
catalog_path = os.path.join(workspace_dir, "catalog_nk.yml")
data_dir = os.path.join(workspace_dir, "data")

###############################################################################
# Fallback local : Recherche dans catalog_nk.yml et lecture du fichier de données
###############################################################################
def normalize_name_local(name):
    """
    Convertit la chaîne en minuscules, supprime les espaces en début/fin,
    et remplace espaces, tirets, underscores et slashes par un underscore unique.
    Exemple : "C2H4O2 - acetic acid" -> "c2h4o2_acetic_acid"
    """
    s = name.lower().strip()
    s = re.sub(r"[ \/_-]+", "_", s)
    return s

# Chemin local vers le catalog (pour le fallback local)
LOCAL_CATALOG_PATH = os.path.join(os.path.dirname(__file__), "catalog_nk.yml")

def _find_page_dict_in_catalog(shelf, book, page):
    """
    Parcourt catalog_nk.yml pour retrouver l'entrée correspondant à (shelf, book, page).
    Retourne le dictionnaire associé ou None.
    """
    with open(LOCAL_CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = yaml.load(f, Loader=yaml.BaseLoader)
    for sh in catalog:
        if "SHELF" in sh and sh["SHELF"] == shelf:
            for bk in sh.get("content", []):
                if "BOOK" in bk and bk["BOOK"] == book:
                    for pg in bk.get("content", []):
                        if "PAGE" in pg and pg["PAGE"] == page:
                            return pg
    return None

def _interpolate_tabulated(nk_lines, wavelength_nm, mode="nk"):
    """
    Parse une chaîne de données tabulées (exemple : "500 1.45 0\n600 1.44 0\n...")
    et interpole les valeurs de n et k à la longueur d'onde wavelength_nm.
    Le paramètre mode peut être "n", "k" ou "nk".
    Retourne un tuple (n, k).
    """
    rows = nk_lines.strip().split("\n")
    table = []
    for r in rows:
        rr = r.strip()
        if rr:
            parts = rr.split()
            table.append(list(map(float, parts)))
    table = np.array(table)
    wl = table[:, 0]
    if mode == "n" and table.shape[1] >= 2:
        nvals = table[:, 1]
        n_val = float(np.interp(wavelength_nm, wl, nvals))
        return (n_val, 0.0)
    elif mode == "k" and table.shape[1] >= 2:
        kvals = table[:, 1]
        k_val = float(np.interp(wavelength_nm, wl, kvals))
        return (0.0, k_val)
    elif mode == "nk" and table.shape[1] >= 3:
        nvals = table[:, 1]
        kvals = table[:, 2]
        n_val = float(np.interp(wavelength_nm, wl, nvals))
        k_val = float(np.interp(wavelength_nm, wl, kvals))
        return (n_val, k_val)
    return (0.0, 0.0)

def _interpolate_formula(data_block, wavelength_nm):
    """
    Interpole les valeurs n et k pour un bloc de type "formula X" (X = 1..9)
    tel que défini dans les données de refractiveindex.info.
    Cette implémentation simplifiée doit être étendue pour une compatibilité complète.
    """
    ftype = data_block.get("type", "").split()  # ex: ["formula", "1"]
    formula_type = ftype[1] if len(ftype) > 1 else "1"
    coeff_str = data_block.get("coefficients", "")
    wls_str = data_block.get("wavelength_range", "")
    if not coeff_str or not wls_str:
        return (1.0, 0.0)
    wl_range = list(map(float, wls_str.split()))
    lam_array = np.linspace(wl_range[0], wl_range[1], 100)
    # Pour l'exemple, on renvoie n=1 pour toute longueur d'onde
    n_array = np.ones_like(lam_array)
    n_val = float(np.interp(wavelength_nm, lam_array, n_array))
    return (n_val, 0.0)

def get_refractive_index_epsilon_from_file(file_path, lambda_val):
    """
    Lit le fichier situé à file_path et renvoie ε = (n + 1j*k)**2 pour lambda_val (en nm).
    
    - Si le fichier est un .txt, on suppose un format avec une ligne d'en-tête : wl  n  k.
    - Pour un fichier YAML, on gère les données "tabulated" et "formula".
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        data = np.loadtxt(file_path, skiprows=1)
        if data.size == 0:
            raise ValueError("No data found in text file.")
        wl = data[:, 0]
        n_array = data[:, 1]
        k_array = data[:, 2] if data.ndim > 1 and data.shape[1] > 2 else np.zeros_like(n_array)
        n_val = float(np.interp(lambda_val, wl, n_array))
        k_val = float(np.interp(lambda_val, wl, k_array))
        return (n_val + 1j*k_val)**2
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            datafile = yaml.load(file, Loader=yaml.BaseLoader)
        for data in datafile.get("DATA", []):
            datatype = data.get("type").split()
            if datatype[0] == "tabulated":
                rows = data.get("data").strip().split("\n")
                table = [list(map(float, row.split())) for row in rows if row.strip()]
                table = np.array(table)
                if table.size == 0:
                    continue
                if datatype[1] == "n":
                    wl = table[:, 0]
                    n_array = table[:, 1]
                    n_val = float(np.interp(lambda_val, wl, n_array))
                    return (n_val + 0j)**2
                elif datatype[1] == "k":
                    wl = table[:, 0]
                    k_array = table[:, 1]
                    k_val = float(np.interp(lambda_val, wl, k_array))
                    return (0 + 1j*k_val)**2
                elif datatype[1] == "nk":
                    wl = table[:, 0]
                    n_array = table[:, 1]
                    k_array = table[:, 2]
                    n_val = float(np.interp(lambda_val, wl, n_array))
                    k_val = float(np.interp(lambda_val, wl, k_array))
                    return (n_val + 1j*k_val)**2
            elif datatype[0] == "formula":
                coefficients = list(map(float, data.get("coefficients").split()))
                wavelength_range = np.array(data.get("wavelength_range").split(), dtype=float)
                if wavelength_range[1] / wavelength_range[0] > 20:
                    wl = np.logspace(np.log10(wavelength_range[0]), np.log10(wavelength_range[1]), 101)
                else:
                    wl = np.linspace(wavelength_range[0], wavelength_range[1], 101)
                formula_type = int(datatype[1])
                refr_index_obj = RefractiveIndexData.setupRefractiveIndex(
                    formula=formula_type,
                    rangeMin=float(wavelength_range[0]),
                    rangeMax=float(wavelength_range[1]),
                    coefficients=coefficients
                )
                n_val_array = refr_index_obj.getRefractiveIndex(np.array([lambda_val]))
                return (float(n_val_array[0]) + 0j)**2
        raise ValueError("No valid data found in file.")

###############################################################################
# Fonctions finales : configuration dynamique des matériaux
###############################################################################
def get_epsilon_refractiveindex(shelf, book, page, lambda_val_nm):
    """
    Utilise la classe pip RefractiveIndexMaterial pour obtenir n et k à la longueur d'onde lambda_val_nm.
    En cas d'échec (par exemple, matériau non trouvé dans la base interne), le fallback local sera utilisé.
    Retourne ε = (n + i k)².
    """
    rim = RefractiveIndexMaterial(shelf, book, page)
    lam_array = np.array([lambda_val_nm], dtype=float)
    n_array = rim.get_refractive_index(lam_array)
    k_array = rim.get_extinction_coefficient(lam_array)
    if len(n_array) == 0 or len(k_array) == 0:
        raise ValueError("The refractiveindex library did not return any n,k data.")
    n_val, k_val = float(n_array[0]), float(k_array[0])
    return (n_val + 1j*k_val)**2







def get_material_params(material_name, materials_data):
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
            raise ValueError(f"Incomplete parameters for '{material_name}': {e}")
    else:
        raise ValueError(f"Material '{material_name}' not found in JSON.")




def build_material_configuration_dynamic(df_config, lambda_val_nm, json_path, ri_overrides=None):
    """
    Construit un dictionnaire de permittivités {role: ε} à partir d'une configuration (DataFrame).

    Chaque ligne du DataFrame doit contenir une configuration sous forme de dictionnaire,
    avec la clé "type" pouvant prendre l'une des valeurs : "None", "RefractiveIndex", "Standard" ou "Custom".
      - Pour "None" : retourne 1.0.
      - Pour "RefractiveIndex" : le dictionnaire doit contenir "shelf", "book", "page" et idéalement "data"
            (le chemin relatif vers le fichier YAML dans le catalogue). Si "data" est fourni, il sera utilisé
            en priorité pour obtenir ε via get_refractive_index_epsilon_from_file. Sinon, on tente d'obtenir ε
            via RefractiveIndexMaterial puis par fallback.
      - Pour "Standard" : le dictionnaire doit contenir "material" (la chaîne indiquant le matériau standard),
            et on récupère les données du JSON combiné en utilisant get_n_k (pour ExpData) ou compute_permittivity (pour Brendel–Bormann).
      - Pour "Custom" : le dictionnaire doit contenir "expression" (l'expression numérique à évaluer).
    
    Retourne un dictionnaire {role: ε}.
    """
    if ri_overrides is None:
        ri_overrides = {}

    with open(json_path, "r", encoding="utf-8") as f:
        materials_data = json.load(f)

    available_materials = {k.lower(): k for k in materials_data.keys()}
    materials_perm = {}
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir_local = os.path.join(base_path, "data")

    for _, row in df_config.iterrows():
        role = row["key"]
        # On s'attend à ce que row["material"] soit un dictionnaire
        mat_dict = row["material"] if isinstance(row["material"], dict) else {"type": str(row["material"]).strip()}
        mtype = mat_dict.get("type", "").lower()

        if mtype == "none":
            materials_perm[role] = 1.0

        elif mtype == "refractiveindex":
            # Récupération des identifiants et du chemin 'data' provenant du selector
            shelf = mat_dict.get("shelf", "")
            book  = mat_dict.get("book", "")
            page  = mat_dict.get("page", "")
            data_field = mat_dict.get("data", "").strip()  # Ce champ doit être rempli dans le selector
            if not shelf or not book or not page:
                materials_perm[role] = 1.0
            else:
                if data_field:
                    # Utiliser directement le chemin fourni
                    filename = os.path.join(data_dir, data_field)
                    eps = get_refractive_index_epsilon_from_file(filename, lambda_val_nm)
                else:
                    try:
                        # Première tentative via la classe locale RefractiveIndexMaterial
                        eps = get_epsilon_refractiveindex(shelf, book, page, lambda_val_nm)
                    except (AssertionError, ValueError):
                        # En cas d'échec, utiliser le fallback pour obtenir le chemin
                        from refractiveindexINFO import RefractiveIndex
                        RI_instance = RefractiveIndex()
                        filename = RI_instance.getMaterialFilename(shelf, book, page)
                        if not filename:
                            raise ValueError(f"Unable to locate data file for shelf '{shelf}', book '{book}', page '{page}'.")
                        eps = get_refractive_index_epsilon_from_file(filename, lambda_val_nm)
                materials_perm[role] = eps

        elif mtype == "standard":
            # Le dict doit contenir la clé "material" qui est la chaîne du matériau standard.
            mat_str = mat_dict.get("material", "").strip()
            if mat_str.lower() in available_materials:
                actual_mat = available_materials[mat_str.lower()]
                mat_info = materials_data[actual_mat]
                model = mat_info.get("model", "").lower()
                if model == "expdata":
                    n_val, k_val = get_n_k(actual_mat, lambda_val_nm, json_path)
                    materials_perm[role] = (n_val + 1j*k_val)**2
                elif "brendel" in model or "bormann" in model:
                    try:
                        params = get_material_params(actual_mat, materials_data)
                    except KeyError as e:
                        raise ValueError(f"Incomplete parameters for '{actual_mat}': {e}")
                    eps_val = compute_permittivity(lambda_val_nm, *params[:-1], N=50)
                    materials_perm[role] = eps_val
                else:
                    raise ValueError(f"Unrecognized model '{model}' for material '{mat_str}'.")
            else:
                raise ValueError(f"Standard material '{mat_str}' not found in JSON.")

        elif mtype == "custom":
            expr = mat_dict.get("expression", "").strip()
            try:
                const_val = eval(expr, {"__builtins__": {}}, {})
                materials_perm[role] = const_val
            except Exception as e:
                raise ValueError(f"Error evaluating custom expression '{expr}': {e}")
        else:
            raise ValueError(f"Invalid material type '{mtype}' for role '{role}'.")

    return materials_perm
