# Material_Configuration.py
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
    Par exemple : "C2H4O2 - acetic acid" -> "c2h4o2_acetic_acid"
    """
    s = name.lower().strip()
    s = re.sub(r"[ \/_-]+", "_", s)
    return s

# Chemin vers le fichier catalog_nk.yml local
LOCAL_CATALOG_PATH = os.path.join(os.path.dirname(__file__), "catalog_nk.yml")

def lookup_ri_catalog(material_name):
    """
    Recherche dans catalog_nk.yml un matériau dont le BOOK correspond (comparaison souple).
    Renvoie (shelf, book, page) avec les noms originaux.
    """
    target = normalize_name_local(material_name)
    with open(LOCAL_CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = yaml.load(f, Loader=yaml.BaseLoader)
    for shelf in catalog:
        if "SHELF" not in shelf:
            continue
        for book in shelf.get("content", []):
            if "DIVIDER" in book:
                continue
            book_orig = book.get("BOOK", "")
            book_norm = normalize_name_local(book_orig)
            # Si target est contenu dans book_norm ou inversement
            if target in book_norm or book_norm in target:
                for page in book.get("content", []):
                    if "DIVIDER" in page:
                        continue
                    return shelf["SHELF"], book_orig, page["PAGE"]
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

def get_refractive_index_epsilon_from_file(file_path, lambda_val):
    """
    Lit le fichier YAML situé à file_path, extrait les données et interpole (ou évalue)
    les valeurs de n et k pour la longueur d'onde lambda_val (en nm).
    Prend en charge les types "tabulated" et "formula" (ici, les types "1" et "2" sont gérés).
    Renvoie ε = (n + 1j*k)**2.
    """
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
            if wavelength_range[1]/wavelength_range[0] > 20:
                wl = np.logspace(np.log10(wavelength_range[0]), np.log10(wavelength_range[1]), 101)
            else:
                wl = np.linspace(wavelength_range[0], wavelength_range[1], 101)
            formula_type = datatype[1]
            if formula_type == "1":
                n_val_array = (1 + coefficients[0] +
                               coefficients[1] / (1 - (coefficients[2] / wl) ** 2) +
                               coefficients[3] / (1 - (coefficients[4] / wl) ** 2) +
                               coefficients[5] / (1 - (coefficients[6] / wl) ** 2) +
                               coefficients[7] / (1 - (coefficients[8] / wl) ** 2) +
                               coefficients[9] / (1 - (coefficients[10] / wl) ** 2) +
                               coefficients[11] / (1 - (coefficients[12] / wl) ** 2) +
                               coefficients[13] / (1 - (coefficients[14] / wl) ** 2) +
                               coefficients[15] / (1 - (coefficients[16] / wl) ** 2)) ** 0.5
            elif formula_type == "2":
                n_val_array = (1 + coefficients[0] +
                               coefficients[1] / (1 - coefficients[2] / wl ** 2) +
                               coefficients[3] / (1 - coefficients[4] / wl ** 2) +
                               coefficients[5] / (1 - coefficients[6] / wl ** 2) +
                               coefficients[7] / (1 - coefficients[8] / wl ** 2) +
                               coefficients[9] / (1 - coefficients[10] / wl ** 2) +
                               coefficients[11] / (1 - coefficients[12] / wl ** 2) +
                               coefficients[13] / (1 - coefficients[14] / wl ** 2) +
                               coefficients[15] / (1 - coefficients[16] / wl ** 2)) ** 0.5
            else:
                raise NotImplementedError(f"Formula type '{formula_type}' not implemented")
            n_val = float(np.interp(lambda_val, wl, n_val_array))
            k_val = 0
            return (n_val + 1j*k_val)**2
    raise ValueError("No valid data found in file.")

def get_material_params(material_name, materials_data):
    """
    Extrait les paramètres d'un matériau depuis materials_data.
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

def build_material_configuration_dynamic(df_config, lambda_val, json_path, ri_overrides=None):
    """
    Construit un dictionnaire de permittivités {role: ε} à partir d'une configuration (DataFrame).

    Pour chaque rôle :
      - "None" renvoie 1.0.
      - Si le matériau est défini dans le JSON combiné, on utilise get_n_k ou compute_permittivity selon le modèle.
      - Si la valeur est "RefractiveIndex", on tente d'abord d'obtenir les indices via RefractiveIndexMaterial
        (en utilisant les méthodes getRefractiveIndex et getExtinctionCoefficient de la classe Material dans refractiveindex.py).
        Pour plus de flexibilité, la recherche dans le catalogue est faite avec une normalisation sur les noms.
        En cas d'échec, on se rabat sur une recherche locale dans le dossier "data" via find_data_file.
      - Sinon, on évalue la valeur comme une constante.
    """
    if ri_overrides is None:
        ri_overrides = {}
    materials_data = json.load(open(json_path, 'r'))
    available_materials = {k.lower(): k for k in materials_data.keys()}
    materials_perm = {}
    # base_path correspond au répertoire parent contenant catalog_nk.yml et le dossier data.
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
                n_val, k_val = get_n_k(actual_mat, lambda_val, json_path)
                materials_perm[key] = (n_val + 1j*k_val)**2
            else:
                try:
                    params = get_material_params(actual_mat, materials_data)
                    perm = compute_permittivity(lambda_val, *params[:-1], N=50)
                    materials_perm[key] = perm
                except KeyError as e:
                    raise ValueError(f"Incomplete parameters for '{actual_mat}': {e}")
        else:
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
                        # Première tentative via RefractiveIndexMaterial
                        rim = RefractiveIndexMaterial(shelf=shelf, book=book, page=page)
                        # Si l'instanciation échoue à cause d'une nomenclature non standard, on réessaie avec des valeurs canoniques
                        n = rim.material.getRefractiveIndex(np.array([lambda_val]))
                        k = rim.material.getExtinctionCoefficient(np.array([lambda_val]))
                        epsilon = (n + 1j*k)**2
                        materials_perm[key] = epsilon
                        continue
                    except Exception as e:
                        # En cas d'échec, on tente de recalculer les identifiants canoniques à partir du BOOK fourni
                        try:
                            canonical = lookup_ri_catalog(book)
                            shelf, book, page = canonical
                            rim = RefractiveIndexMaterial(shelf=shelf, book=book, page=page)
                            n = rim.material.getRefractiveIndex(np.array([lambda_val]))
                            k = rim.material.getExtinctionCoefficient(np.array([lambda_val]))
                            epsilon = (n + 1j*k)**2
                            materials_perm[key] = epsilon
                            continue
                        except Exception as e2:
                            # Fallback : recherche locale dans le dossier data
                            data_file = find_data_file(shelf, book, base_path)
                            if data_file is None:
                                raise ValueError(f"No data file found for shelf '{shelf}' and book '{book}'")
                            epsilon = get_refractive_index_epsilon_from_file(data_file, lambda_val)
                            materials_perm[key] = epsilon
                            continue
                except Exception as e:
                    raise ValueError(f"RefractiveIndex material for role '{key}' not found: {e}")
            else:
                try:
                    const_val = eval(mat, {"__builtins__": {}}, {})
                    materials_perm[key] = const_val
                except Exception as e2:
                    raise ValueError(f"Material '{mat}' not found in JSON, nor via RefractiveIndex, and cannot be evaluated as constant: {e2}")
    return materials_perm
