#!/usr/bin/env python3
"""
Module: data_readers.py

Ce module réunit toutes les fonctions de lecture et de pré‑traitement des données utilisées dans votre projet.

Fonctions disponibles :
  - read_all_combos(file_path) :
      Extrait, depuis un fichier simulation_summary_XXX.txt, les points de réflectance Rup par combo.
  - read_experimental_data(file_path) :
      Lit un fichier de données expérimentales et renvoie (wavelengths, R_values).
  - parse_simulation_summary(file_path) :
      Parse un fichier de simulation pour extraire les informations de chaque combo (label, geometry, material).
  - parse_experimental_data_summary(file_path) :
      Parse un fichier expérimental pour extraire les résumés de la configuration.
  - list_sim_summary_files(summary_dir) :
      Retourne la liste triée des fichiers de simulation (pattern "simulation_summary*.txt") présents dans summary_dir.
  - list_exp_data_files(exp_data_dir) :
      Retourne la liste triée des fichiers expérimentaux (pattern "Data_structure*.txt") présents dans exp_data_dir.
  - get_simulation_label(base_label, file_path, label_to_tag) :
      Construit un label unique pour une simulation à partir du nom de base et du nom du fichier.
  - get_all_spectra_and_summaries(summary_dir, exp_data_dir, ordered_params) :
      Agrège tous les spectres (simulés et expérimentaux) et construit leurs résumés (geometry et material)
      à l’aide d’une liste de tuples (clé, nom à afficher) passée dans ordered_params.
"""

import os
import glob
import re
import ast
import numpy as np

# --- Fonctions de lecture existantes ---

def read_all_combos(file_path):
    """
    Lit le fichier simulation_summary_XXX.txt et extrait pour chaque combo (délimité par "Combo name:")
    les points de réflectance Rup. La plage de longueurs d'onde est déduite des lignes de points
    de réflectance du premier combo rencontré.
    
    Retourne un dictionnaire dont les clés sont les noms de combo et les valeurs sont un tuple
    (wavelengths, Rup_values) sous forme de tableaux numpy.
    """
    combos = {}
    lambda_range = None  # sera défini à partir des longueurs d'onde du premier combo
    current_combo = None
    current_wavelengths = []
    current_Rup = []
    reading_points = False

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith("Combo name:"):
            if current_combo is not None and current_Rup:
                if lambda_range is None:
                    lambda_range = np.array(current_wavelengths)
                combos[current_combo] = (lambda_range, np.array(current_Rup))
            current_combo = line.split("Combo name:")[1].strip()
            current_wavelengths = []
            current_Rup = []
            reading_points = False
            continue

        if "Reflectance points" in line:
            reading_points = True
            continue

        if reading_points and line.startswith("λ="):
            m = re.search(r"λ=([\d\.]+)\s*nm\s*->\s*Rup=([\d\.eE\+\-]+)", line)
            if m:
                try:
                    wl_val = float(m.group(1))
                    rup_val = float(m.group(2))
                    current_wavelengths.append(wl_val)
                    current_Rup.append(rup_val)
                except Exception:
                    continue

        if reading_points and re.match(r"^-{10,}", line):
            reading_points = False

    if current_combo is not None and current_Rup:
        if lambda_range is None:
            lambda_range = np.array(current_wavelengths)
        combos[current_combo] = (lambda_range, np.array(current_Rup))
    
    if not combos:
        raise ValueError("Aucun combo n'a pu être extrait du fichier.")
    return combos

def read_experimental_data(file_path):
    """
    Lit le fichier de données expérimentales (Data_structure*.txt) qui contient
    un en-tête suivi des données sous le format :
    
      Wavelengths (nm)     R
      450.0, 0.2654618528289272
      452.7638190954774, 0.2701075857791835
      ...
    
    Les lignes d'en-tête (ne contenant pas de virgule) sont ignorées.
    Retourne (wavelengths, R_values) sous forme de tableaux numpy.
    """
    wavelengths = []
    R_values = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ',' not in line:
                continue
            parts = line.split(',')
            try:
                wl = float(parts[0].strip())
                R_val = float(parts[1].strip())
                wavelengths.append(wl)
                R_values.append(R_val)
            except Exception:
                continue
    if not wavelengths or not R_values:
        raise ValueError("Aucune donnée expérimentale n'a été trouvée dans le fichier.")
    return np.array(wavelengths), np.array(R_values)



def parse_simulation_summary(file_path):
    """
    Lit et parse le fichier simulation_summary_XXX.txt afin d'extraire,
    pour chaque combo, le label, la géométrie et la configuration matériau.
    
    Retourne une liste de dictionnaires.
    """
    combos = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Erreur lors de la lecture de {file_path}: {e}")
        return combos
    pattern = re.compile(
        r"Combo name:\s*(?P<label>.*?)\s*\n"
        r"(?:.*?\n)*?"
        r"Geometry:\s*(?P<geometry>\{.*?\})\s*\n"
        r"(?:.*?\n)*?"
        r"Material config \(df_config\):\s*(?P<material>\[.*?\])",
        re.DOTALL
    )
    for match in pattern.finditer(content):
        label = match.group("label").strip().replace(" - ", "\n")
        geom_str = match.group("geometry").strip()
        mat_str = match.group("material").strip()
        try:
            geometry = ast.literal_eval(geom_str)
        except Exception:
            geometry = {}
        try:
            material = ast.literal_eval(mat_str)
        except Exception:
            material = []
        combos.append({"label": label, "geometry": geometry, "material": material})
    return combos

def parse_experimental_data_summary(file_path):
    """
    Lit et parse un fichier de données expérimentales afin d'extraire,
    pour certaines clés attendues, un résumé de la configuration sous forme
    d'un dictionnaire avec les clés "geometry" et "material".
    """
    expected_keys = [
        "Environnement",
        "Cube",
        "Gap diélectrique / n =",
        "Fonctionnalisation diélectrique / n =",
        "Couche métallique",
        "Substrat"
    ]
    geom_lines = []
    mat_lines = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                for key in expected_keys:
                    if line.startswith(key):
                        parts = line.split(":", 1)
                        if len(parts) < 2:
                            continue
                        value = parts[1].strip()
                        if "/" in value:
                            tokens = [tok.strip() for tok in value.split("/")]
                            if key == "Cube":
                                mat_val = tokens[0]
                                geom_val = tokens[-1] if "nm" in tokens[-1] else ""
                            else:
                                mat_val = tokens[0]
                                geom_val = tokens[-1] if "nm" in tokens[-1] else ""
                        else:
                            mat_val = ""
                            geom_val = value
                        geom_lines.append(f"{key}: {geom_val}".strip())
                        mat_lines.append(f"{key}: {mat_val}".strip())
                        break
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier expérimental {file_path}: {e}")
    return {"geometry": "\n".join(geom_lines), "material": "\n".join(mat_lines)}

# --- Fonctions extraites de interactive_simulation.py ---

def list_sim_summary_files(summary_dir):
    """
    Retourne la liste triée des fichiers de simulation summary présents dans summary_dir.
    Le pattern utilisé est "simulation_summary*.txt".
    """
    pattern = os.path.join(summary_dir, "simulation_summary*.txt")
    files = glob.glob(pattern)
    files.sort()
    return files

def list_exp_data_files(exp_data_dir):
    """
    Retourne la liste triée des fichiers expérimentaux présents dans exp_data_dir.
    Le pattern utilisé est "Data_structure*.txt".
    """
    pattern = os.path.join(exp_data_dir, "Data_structure*.txt")
    files = glob.glob(pattern)
    files.sort()
    return files

def get_simulation_label(base_label, file_path, label_to_tag):
    """
    Construit un label unique pour une simulation à partir du nom de base et du nom du fichier.
    Ce label est enrichi avec une version extraite du nom de fichier et un compteur
    pour éviter les doublons.
    """
    fname = os.path.basename(file_path)
    tag = os.path.splitext(fname)[0]
    prefix = "simulation_summary_RCWA_"
    version = ""
    if tag.startswith(prefix):
        remainder = tag[len(prefix):]
        parts = remainder.split("_", 1)
        if parts:
            version = parts[0]
    if base_label not in label_to_tag:
        label_to_tag[base_label] = {}
    if version not in label_to_tag[base_label]:
        label_to_tag[base_label][version] = 1
        return f"{base_label} ({version})" if version else base_label
    else:
        label_to_tag[base_label][version] += 1
        count = label_to_tag[base_label][version]
        return f"{base_label} ({version} {count})"

def get_all_spectra_and_summaries(summary_dir, exp_data_dir, ordered_params):
    """
    Parcourt les fichiers de simulation et expérimentaux, récupère les spectres et
    prépare pour chacun un résumé (geometry et material) à l'aide de ordered_params.
    
    Parameters:
      - summary_dir: répertoire contenant les fichiers simulation_summary*.txt.
      - exp_data_dir: répertoire contenant les fichiers Data_structure*.txt.
      - ordered_params: liste de tuples (clé, nom_affiché) utilisée pour formater les résumés.
      
    Retourne:
      - spectra: dictionnaire {label: (wavelengths, reflectance_values)}.
      - summaries: dictionnaire {label: (geometry_summary, material_summary)}.
    """
    spectra = {}
    summaries = {}
    label_to_tag = {}
    sim_files = list_sim_summary_files(summary_dir)
    for fpath in sim_files:
        combos = read_all_combos(fpath)
        sim_configs = parse_simulation_summary(fpath)
        for combo_label, (wl, R) in combos.items():
            base_label = combo_label.replace(" - ", "\n")
            if base_label in spectra:
                new_label = get_simulation_label(base_label, fpath, label_to_tag)
            else:
                fname = os.path.basename(fpath)
                tag = os.path.splitext(fname)[0]
                prefix = "simulation_summary_RCWA_"
                if tag.startswith(prefix):
                    remainder = tag[len(prefix):]
                    parts = remainder.split("_", 1)
                    tag = parts[0] if parts else ""
                label_to_tag[base_label] = {tag: 1}
                new_label = f"{base_label} ({tag})" if tag else base_label
            spectra[new_label] = (wl, R)
            found = False
            for cfg in sim_configs:
                cfg_label = cfg.get("label", "Unknown").replace(" - ", "\n")
                if cfg_label == base_label:
                    geom = cfg.get("geometry", {})
                    geom_lines = []
                    for key, disp_name in ordered_params:
                        if key in geom:
                            geom_lines.append(f"{disp_name}: {geom[key]}")
                    geom_summary = "\n".join(geom_lines)
                    mat = cfg.get("material", [])
                    mat_lines = []
                    if isinstance(mat, list):
                        for entry in mat:
                            key = entry.get("key", "")
                            disp_name = key
                            for k, dname in ordered_params:
                                if k == key:
                                    disp_name = dname
                                    break
                            mat_info = entry.get("material", {})
                            mtype = mat_info.get("type", "").strip().lower()
                            if mtype == "standard":
                                val = mat_info.get("material", "").strip()
                            elif mtype == "custom":
                                val = mat_info.get("expression", "").strip()
                            else:
                                val = ""
                            if val:
                                mat_lines.append(f"{disp_name}: {val}")
                    mat_summary = "\n".join(mat_lines)
                    summaries[new_label] = (geom_summary, mat_summary)
                    found = True
                    break
            if not found:
                summaries[new_label] = ("", "")
    exp_files = list_exp_data_files(exp_data_dir)
    for fpath in exp_files:
        data = read_experimental_data(fpath)
        if data:
            base_lbl = os.path.basename(fpath)
            spectra[base_lbl] = data
            exp_data = parse_experimental_data_summary(fpath)
            geom_summary = exp_data.get("geometry", "")
            mat_summary = exp_data.get("material", "")
            summaries[base_lbl] = (geom_summary, mat_summary)
    return spectra, summaries
