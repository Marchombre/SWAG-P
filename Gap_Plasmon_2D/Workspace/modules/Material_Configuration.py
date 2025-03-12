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
    """
    Extracts the parameters for a given material from the materials_data dictionary.
    
    Returns a tuple: (f0, omega_p, Gamma0, f, omega, gamma, sigma, model).
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
            raise ValueError(f"The parameters for '{material_name}' are incomplete in the combined JSON file: {e}")
    else:
        raise ValueError(f"Material '{material_name}' not found in the combined JSON file.")



def build_material_configuration_dynamic(df_config, lambda_val_nm, json_combined_path, ri_overrides=None):
    """
    Construit un dictionnaire de permittivités {role: ε} à partir d'une configuration (DataFrame).

    Chaque ligne du DataFrame doit contenir une configuration sous forme de dictionnaire,
    avec la clé "type" pouvant prendre l'une des valeurs : "None", "RefractiveIndex", "Standard" ou "Custom".
      - Pour "None" : retourne 1.0.
      - Pour "RefractiveIndex" : le dictionnaire doit contenir "shelf", "book", "page" et éventuellement "data"
            (le chemin relatif vers le fichier YAML dans le catalogue). Si "data" est fourni, il sera utilisé en priorité
            pour localiser le fichier. Sinon, le fichier sera recherché via shelf, book et page via la méthode getMaterialFilename.
            Une fois le fichier trouvé, le module refractiveindexINFO est utilisé pour extraire l'indice n et le coefficient k,
            puis calculer ε = (n + 1j*k)² (avec un k=0 par défaut si non spécifié).
      - Pour "Standard" : le dictionnaire doit contenir "material" (la chaîne indiquant le matériau standard),
            et on récupère les données du JSON combiné en utilisant get_n_k (pour ExpData) ou compute_permittivity (pour Brendel–Bormann).
      - Pour "Custom" : le dictionnaire doit contenir "expression" (l'expression numérique à évaluer).
    
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

    # Répertoire local pour les fichiers YAML (utilisé pour le champ "data")
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir_local = os.path.join(base_path, "data")

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
                    # Par défaut, on considère k = 0 si le coefficient d'extinction n'est pas spécifié
                    n = mat_instance.getRefractiveIndex(lambda_val_nm)
                    k = 0.0
                except Exception as e:
                    raise ValueError(f"Erreur lors de la récupération des indices pour le fichier {filename} : {e}")
                eps = (n + 1j * k) ** 2
                materials_perm[role] = eps

        elif mtype == "standard":
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
                raise ValueError(f"Le matériau standard '{mat_str}' n'a pas été trouvé dans le fichier JSON.")

        elif mtype == "custom":
            expr = mat_dict.get("expression", "").strip()
            try:
                const_val = eval(expr, {"__builtins__": {}}, {})
                materials_perm[role] = const_val
            except Exception as e:
                raise ValueError(f"Erreur lors de l'évaluation de l'expression personnalisée '{expr}': {e}")
        else:
            raise ValueError(f"Type de matériau invalide '{mtype}' pour le rôle '{role}'.")

    return materials_perm
