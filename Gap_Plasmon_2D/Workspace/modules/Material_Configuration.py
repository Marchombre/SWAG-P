import json
import pandas as pd
import numpy as np
import yaml
import os
import re
import scipy.interpolate

from Functions_ExpData import get_n_k, compute_permittivity

# Import de la classe RefractiveIndexMaterial depuis refractiveindex.py
from refractiveindex import RefractiveIndexMaterial

def normalize_name_local(name):
    """
    Convertit la chaîne en minuscules, supprime les espaces en début/fin,
    et remplace espaces, tirets, underscores et slashes par un underscore unique.
    Exemple : "C2H4O2 - acetic acid" -> "c2h4o2_acetic_acid"
    """
    s = name.lower().strip()
    s = re.sub(r"[ \/_-]+", "_", s)
    return s

# Chemin vers le fichier catalog_nk.yml local
LOCAL_CATALOG_PATH = os.path.join(os.path.dirname(__file__), "catalog_nk.yml")

def lookup_ri_catalog(material_name):
    """
    Recherche dans catalog_nk.yml un matériau dont les champs SHELF, BOOK et PAGE 
    correspondent de manière flexible à material_name.
    
    Si material_name contient ":", il doit être au format "shelf:book:page".
    Sinon, la chaîne normalisée est comparée aux trois champs.
    
    Renvoie (shelf, book, page) avec les valeurs originales.
    """
    if ":" in material_name:
        parts = material_name.split(":")
        if len(parts) != 3:
            raise ValueError("Material name must be in the format 'shelf:book:page' when using ':' as separator.")
        shelf_target, book_target, page_target = (normalize_name_local(part) for part in parts)
    else:
        shelf_target = book_target = page_target = normalize_name_local(material_name)
    
    with open(LOCAL_CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = yaml.load(f, Loader=yaml.BaseLoader)
    for shelf in catalog:
        shelf_val = shelf.get("SHELF", "")
        shelf_norm = normalize_name_local(shelf_val)
        if shelf_target not in shelf_norm and shelf_norm not in shelf_target:
            continue
        for book in shelf.get("content", []):
            if "DIVIDER" in book:
                continue
            book_val = book.get("BOOK", "")
            book_norm = normalize_name_local(book_val)
            if book_target not in book_norm and book_norm not in book_target:
                continue
            for page in book.get("content", []):
                if "DIVIDER" in page:
                    continue
                page_val = page.get("PAGE", "")
                page_norm = normalize_name_local(page_val)
                if page_target in page_norm or page_norm in page_target:
                    return shelf_val, book_val, page_val
    raise ValueError(f"Material '{material_name}' not found in catalog.")

def find_data_file(shelf, book, base_path):
    """
    Recherche récursivement dans base_path/data le premier fichier YAML situé dans le répertoire correspondant à shelf et book.
    Si data/<shelf>/<book> n'existe pas, recherche dans tous les sous-dossiers de data un dossier dont le nom normalisé correspond au book.
    Renvoie le chemin complet du fichier trouvé, ou None.
    """
    data_root = os.path.join(base_path, "data")
    target_dir = os.path.join(data_root, shelf, book)
    if os.path.isdir(target_dir):
        for root, dirs, files in os.walk(target_dir):
            for file in files:
                if file.lower().endswith((".yml", ".yaml")):
                    return os.path.join(root, file)
    else:
        normalized_book = normalize_name_local(book)
        for root, dirs, files in os.walk(data_root):
            for d in dirs:
                if normalize_name_local(d) == normalized_book:
                    candidate = os.path.join(root, d)
                    for r, ds, fs in os.walk(candidate):
                        for f in fs:
                            if f.lower().endswith((".yml", ".yaml")):
                                return os.path.join(r, f)
    return None

def find_txt_file(material_name, data_dir):
    """
    Recherche récursivement dans data_dir un fichier texte (.txt) dont le nom (sans extension)
    correspond, après normalisation, à material_name.
    Renvoie le chemin complet du fichier trouvé, ou None.
    """
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(".txt"):
                base = os.path.splitext(f)[0]
                if normalize_name_local(base) == normalize_name_local(material_name):
                    return os.path.join(root, f)
    return None

def get_refractive_index_epsilon_from_file(file_path, lambda_val):
    """
    Lit le fichier situé à file_path et renvoie ε = (n + 1j*k)**2 pour la longueur d'onde lambda_val (en nm).
    
    - Pour un fichier .txt, on suppose un format à colonnes : wl  n  k (avec une ligne d'en-tête).
    - Pour un fichier YAML, on gère les données "tabulated" et "formula".
      Pour "formula", le calcul est délégué à RefractiveIndexData.
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
                    k_val = 0
                    return (n_val + 1j*k_val)**2
                elif datatype[1] == "k":
                    wl = table[:, 0]
                    k_array = table[:, 1]
                    k_val = float(np.interp(lambda_val, wl, k_array))
                    n_val = 0
                    return (n_val + 1j*k_val)**2
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
                from refractiveindex import RefractiveIndexData
                refr_index_obj = RefractiveIndexData.setupRefractiveIndex(
                    formula=formula_type,
                    rangeMin=float(wavelength_range[0]),
                    rangeMax=float(wavelength_range[1]),
                    coefficients=coefficients
                )
                n_val_array = refr_index_obj.getRefractiveIndex(np.array([lambda_val]))
                return (float(n_val_array[0]) + 1j*0)**2
        raise ValueError("No valid data found in file.")

def get_material_params(material_name, materials_data):
    """
    Extrait les paramètres d'un matériau depuis materials_data (JSON combiné).
    Renvoie un tuple : (f0, omega_p, Gamma0, f, omega, gamma, sigma, model).
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
            raise ValueError(f"Incomplete parameters for '{material_name}': {e}")
    else:
        raise ValueError(f"Material '{material_name}' not found in JSON.")

# Regex pour vérifier une expression numérique (chiffres, opérateurs, espaces et parenthèses)
numeric_expr_pattern = re.compile(r'^[\d\.\+\-\*\/\s\(\)]+$')

def build_material_configuration_dynamic(df_config, lambda_val, json_path, ri_overrides=None):
    """
    Construit un dictionnaire de permittivités {role: ε} à partir d'une configuration (DataFrame).

    Pour chaque rôle :
      - "None" renvoie 1.0.
      - Si le matériau est défini dans le JSON combiné, on utilise get_n_k (pour ExpData)
        ou compute_permittivity (pour BrendelBormann) selon le champ "model".
      - Si la valeur commence par "RefractiveIndex", on tente d'obtenir les indices via RefractiveIndexMaterial
        (en utilisant getRefractiveIndex et getExtinctionCoefficient).
        La recherche des identifiants se fait de manière flexible via lookup_ri_catalog.
        En cas d'échec, on se rabat sur une recherche locale dans le dossier "data"
        en cherchant d'abord un fichier YAML, puis un fichier TXT.
      - Sinon, on tente d'évaluer la valeur comme une constante via eval si c'est une expression numérique.
        Si ce n'est pas le cas, on tente de trouver un fichier TXT correspondant.
    """
    if ri_overrides is None:
        ri_overrides = {}
    with open(json_path, 'r') as f:
        materials_data = json.load(f)
    available_materials = {k.lower(): k for k in materials_data.keys()}
    materials_perm = {}
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_path, "data")

    for idx, row in df_config.iterrows():
        key = row['key']
        mat = row['material'].strip()
        mat_lower = mat.lower()

        # Cas "None"
        if mat_lower == "none":
            materials_perm[key] = 1.0
            continue

        # Cas présent dans le JSON combiné (ExpData ou BB)
        if mat_lower in available_materials:
            actual_mat = available_materials[mat_lower]
            material = materials_data[actual_mat]
            model = material.get("model", "").lower()
            if model == "expdata":
                n_val, k_val = get_n_k(actual_mat, lambda_val, json_path)
                materials_perm[key] = (n_val + 1j*k_val)**2
            else:
                try:
                    params = get_material_params(actual_mat, materials_data)
                    perm = compute_permittivity(lambda_val, *params[:-1], N=50)
                    materials_perm[key] = perm
                except KeyError as e:
                    raise ValueError(f"Incomplete parameters for '{actual_mat}': {e}")
            continue

        # Cas RefractiveIndex
        if mat.startswith("RefractiveIndex"):
            try:
                if key in ri_overrides:
                    override = ri_overrides[key]
                    if "name" in override and override["name"]:
                        shelf, book, page = lookup_ri_catalog(override["name"])
                    else:
                        shelf = override.get("shelf", "main")
                        book = override.get("book")
                        page = override.get("page")
                        if not book or not page:
                            raise ValueError(f"Override for role '{key}' incomplete: 'book' and 'page' required.")
                else:
                    shelf, book, page = lookup_ri_catalog(key)
                try:
                    rim = RefractiveIndexMaterial(shelf=shelf, book=book, page=page)
                    n = rim.material.getRefractiveIndex(np.array([lambda_val]))
                    try:
                        k = rim.material.getExtinctionCoefficient(np.array([lambda_val]))
                    except Exception:
                        k = 0.0
                    epsilon = (n + 1j*k)**2
                    materials_perm[key] = epsilon
                    continue
                except Exception:
                    # Fallback : recherche locale dans le dossier data
                    data_file = find_data_file(shelf, book, base_path)
                    if data_file is not None:
                        epsilon = get_refractive_index_epsilon_from_file(data_file, lambda_val)
                        materials_perm[key] = epsilon
                        continue
                    txt_file = find_txt_file(mat, data_dir)
                    if txt_file is not None:
                        epsilon = get_refractive_index_epsilon_from_file(txt_file, lambda_val)
                        materials_perm[key] = epsilon
                        continue
                    raise ValueError(f"No data file found for shelf '{shelf}' and book '{book}'")
            except Exception as e:
                raise ValueError(f"RefractiveIndex material for role '{key}' not found: {e}")
        else:
            # Tentative d'évaluer comme constante si c'est une expression numérique
            if numeric_expr_pattern.match(mat):
                try:
                    const_val = eval(mat, {"__builtins__": {}}, {})
                    materials_perm[key] = const_val
                except Exception as e2:
                    raise ValueError(f"Error evaluating numeric expression '{mat}': {e2}")
            else:
                # Sinon, on tente de trouver un fichier TXT correspondant
                txt_file = find_txt_file(mat, data_dir)
                if txt_file is not None:
                    epsilon = get_refractive_index_epsilon_from_file(txt_file, lambda_val)
                    materials_perm[key] = epsilon
                else:
                    raise ValueError(f"Material '{mat}' not found in JSON, nor via RefractiveIndex, and is not a valid numeric expression.")

    return materials_perm
