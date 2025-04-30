#!/usr/bin/env python3
# Indique au système d’utiliser l’interpréteur Python 3 pour exécuter ce script.

"""
Module: data_readers.py

Ce module réunit toutes les fonctions de lecture et de pré‑traitement des données utilisées.

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
# Le docstring ci‑dessus décrit le rôle du module et la liste de ses fonctions disponibles.

import os
import glob
import re
import ast
import numpy as np


# --- Fonctions de lecture ---


def get_material_str_clean(simulation_details):
    """
    Extrait et retourne une chaîne construite à partir des configurations
    matérielles contenues dans simulation_details.
    
    Pour chaque configuration matérielle, la fonction récupère la valeur 
    associée à la clé (par exemple "perm_gap") et ajoute le nom de la couche 
    (la partie après "perm_") suivi d'un underscore devant la valeur nettoyée.
    
    Par exemple, pour :
        "key": "perm_gap",
        "material": {
            "type": "Custom",
            "expression": "1.45**2"
        }
    le fragment généré sera "gap_1.45**2".
    """
    roles_order = [
        "perm_env", "perm_reso", "perm_gap", "perm_mol",
        "perm_func", "perm_diel", "perm_metalliclayer",
        "perm_accroche", "perm_sub"
    ]
    suffix_parts = []
    if simulation_details:
        # On prend la première configuration pour constituer le nom de fichier
        first_combo = next(iter(simulation_details.values()))
        for role in roles_order:
            val = ""
            for entry in first_combo["material_config"]:
                if entry.get("key", "").strip() == role:
                    mat_info = entry.get("material", {})
                    mtype = mat_info.get("type", "").strip().lower()
                    if mtype == "standard":
                        val = mat_info.get("material", "").strip()
                    elif mtype == "refractiveindex":
                        book = mat_info.get("book", "")
                        page = mat_info.get("page", "")
                        val = f"Book: {book}, Page: {page}"                        
                    elif mtype == "custom":
                        val = mat_info.get("expression", "").strip()
                    break
            if val.lower() != "none" and val != "":
                # Nettoyage de la valeur pour conserver uniquement les caractères alphanumériques, points, astérisques et plus
                val_clean = re.sub(r'[^A-Za-z0-9\.\*\+]', '', val)
                # Extraction de l'indice de la couche (la partie après "perm_")
                layer_index = role.split("_", 1)[-1]
                suffix_parts.append(f"{layer_index}_{val_clean}")
    filtered_parts = [part for part in suffix_parts if part]
    return "_".join(filtered_parts)



def read_all_combos(file_path):
    """
    Lit le fichier simulation_summary_XXX.txt et extrait pour chaque combo (délimité par "Combo name:")
    les points de réflectance Rup. La plage de longueurs d'onde est déduite des lignes de points
    de réflectance du premier combo rencontré.
    
    Retourne un dictionnaire dont les clés sont les noms de combo et les valeurs sont un tuple
    (wavelengths, Rup_values) sous forme de tableaux numpy.
    """
    combos = {}                       # Dictionnaire final : clé = nom de combo, valeur = (λ, Rup)
    lambda_range = None              # Sera défini lorsque le premier combo fournit sa liste de λ
    current_combo = None             # Nom du combo en cours de lecture
    current_wavelengths = []         # Liste temporaire des λ pour ce combo
    current_Rup = []                 # Liste temporaire des Rup pour ce combo
    reading_points = False           # Indicateur : sommes‑nous dans la section “Reflectance points” ?

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()        # Lecture de toutes les lignes du fichier

    for line in lines:
        line = line.strip()          # Suppression des espaces en début/fin de ligne
        if line.startswith("Combo name:"):
            # Si on rencontre une nouvelle section de combo :
            if current_combo is not None and current_Rup:
                # Si un combo précédent était en cours et a des données Rup
                if lambda_range is None:
                    # Pour le premier combo, fixer la plage de λ
                    lambda_range = np.array(current_wavelengths)
                # Enregistrer le combo précédent dans le dict
                combos[current_combo] = (lambda_range, np.array(current_Rup))
            # Extraire et mémoriser le nouveau nom de combo
            current_combo = line.split("Combo name:")[1].strip()
            # Réinitialiser les listes de points
            current_wavelengths = []
            current_Rup = []
            reading_points = False  # Nous ne lirons les points qu’après la ligne « Reflectance points »
            continue                  # Passer à la ligne suivante

        if "Reflectance points" in line:
            # Dès qu’on voit cette mention, on passe en mode lecture des points
            reading_points = True
            continue

        if reading_points and line.startswith("λ="):
            # Extraction des valeurs λ et Rup dans la ligne
            m = re.search(
                r"λ=([\d\.]+)\s*nm\s*->\s*Rup=([\d\.eE\+\-]+)",
                line
            )
            if m:
                try:
                    wl_val  = float(m.group(1))  # Conversion du premier groupe en float
                    rup_val = float(m.group(2))  # Conversion du second groupe en float
                    current_wavelengths.append(wl_val)
                    current_Rup.append(rup_val)
                except Exception:
                    # En cas d’erreur de conversion, on ignore cette ligne
                    continue

        if reading_points and re.match(r"^-{10,}", line):
            # Une ligne de tirets longs marque la fin des points de réflectance
            reading_points = False

    # Après la boucle, vérifier s’il reste un combo en cours à enregistrer
    if current_combo is not None and current_Rup:
        if lambda_range is None:
            lambda_range = np.array(current_wavelengths)
        combos[current_combo] = (lambda_range, np.array(current_Rup))
    
    if not combos:
        # Si aucun combo n’a été extrait, c’est une erreur
        raise ValueError("Aucun combo n'a pu être extrait du fichier.")
    return combos  # Retourne le dict complet

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
    wavelengths = []  # Liste pour stocker les valeurs de λ
    R_values    = []  # Liste pour les valeurs de réflectance expérimentale
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()      # Supprimer espaces en début/fin
            if not line or ',' not in line:
                # Sauter les lignes vides ou sans virgule (en-tête)
                continue
            parts = line.split(',')  # Séparer λ et R autour de la virgule
            try:
                wl    = float(parts[0].strip())  # Conversion en float
                R_val = float(parts[1].strip())  # Conversion en float
                wavelengths.append(wl)
                R_values.append(R_val)
            except Exception:
                # En cas d’erreur (ligne mal formée), on l’ignore
                continue
    if not wavelengths or not R_values:
        # Si aucune donnée valide n’a été trouvée, on lève une erreur
        raise ValueError("Aucune donnée expérimentale n'a été trouvée dans le fichier.")
    return np.array(wavelengths), np.array(R_values)  # Retourne deux arrays NumPy

def parse_simulation_summary(file_path):
    """
    Lit et parse le fichier simulation_summary_XXX.txt afin d'extraire,
    pour chaque combo, le label, la géométrie et la configuration matériau.
    
    Retourne une liste de dictionnaires contenant ces informations.
    """
    combos = []  # Liste qui contiendra un dict par combo
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()  # Lecture intégrale du contenu du fichier
    except Exception as e:
        print(f"Erreur lors de la lecture de {file_path}: {e}")
        return combos      # On retourne une liste vide en cas d’erreur

    # Expression régulière multi‑ligne pour capturer label, geometry et material
    pattern = re.compile(
        r"Combo name:\s*(?P<label>.*?)\s*\n"              # label
        r"(?:.*?\n)*?"                                     # lignes intermédiaires
        r"Geometry:\s*\n"                                  # Geometry: suivi d'un saut de ligne
        r"(?P<geometry>\{.*?\})\s*\n"                      # capture du dict complet
        r"(?:.*?\n)*?"
        r"Material config \(df_config\):\s*\n"             # Material: suivi d'un saut de ligne
        r"(?P<material>\[.*?\])\s*\n"                      # capture de la liste principale
        r"RI Overrides:",                                  # on marque la fin juste avant ce champ
        re.DOTALL
    )

    
    # Parcours de toutes les correspondances dans le contenu
    for match in pattern.finditer(content):
        label   = match.group("label").strip().replace(" - ", "\n")
        geom_str= match.group("geometry").strip()
        mat_str = match.group("material").strip()
        
        # extrait le bloc Metrics: jusqu'à la ligne de tirets
        metrics_block = re.search(
            r"Metrics:\s*\n(?P<bm>(?:\s{2,}.+\n)+)",
            content[match.start():], re.MULTILINE)
        metrics = {}
        if metrics_block:
            for line in metrics_block.group("bm").splitlines():
                # ligne du type "  FWHM          : 12.5 nm"
                parts = line.strip().split(":", 1)
                if len(parts)==2:
                    key = parts[0].strip()
                    val = parts[1].strip()
                    metrics[key] = val

        try:
            geometry = ast.literal_eval(geom_str)
        except Exception:
            geometry = {}
        try:
            material = ast.literal_eval(mat_str)
        except Exception:
            material = []
        combos.append({
            "label":    label,
            "geometry": geometry,
            "material": material,
            "metrics":  metrics   # <--- on stocke ici
        })
            
        
    return combos  # Retourne la liste de dicts

    



def parse_experimental_data_summary(file_path):
    """
    Parse le header d'un fichier expérimental en découpant chaque ligne
    contenant ':' ou '/' pour en extraire deux résumés :
      - geometry : épaisseurs (nm ou '∞' pour l'environnement)
      - material : nom ou notation 'n = …' pour chaque couche.
    Accepte un nombre quelconque de couches (plus ou moins de 6).
    """
    geom_lines = []
    mat_lines  = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                # ignorer vide
                continue

            # déterminer label et reste, même si c'est "Gap ... / ..." sans ":"
            if ':' in line:
                label, rest = line.split(':', 1)
            elif '/' in line:
                label, rest = line.split('/', 1)
            else:
                # ni ":" ni "/", ce n'est pas une ligne de couche
                continue

            label = label.strip()
            rest  = rest.strip().lstrip('/').strip()

            # on découpe ensuite par '/'
            tokens = [tok.strip() for tok in rest.split('/') if tok.strip()]

            # --- Géométrie ---
            if label.lower().startswith("environnement"):
                thickness = "∞"
            else:
                # dernier token avec "nm"
                thickness = next((tok for tok in reversed(tokens) if "nm" in tok), "")

            # --- Matériau ---
            # toujours premier token (Air, Argent, n = 1.45, etc.)
            material = tokens[0] if tokens else ""

            geom_lines.append(f"{label}: {thickness}")
            mat_lines.append(f"{label}: {material}")

    return {
        "geometry": "\n".join(geom_lines),
        "material": "\n".join(mat_lines)
    }



def list_sim_summary_files(summary_dir):
    """
    Retourne la liste triée des fichiers de simulation summary présents dans summary_dir.
    Le pattern utilisé est "simulation_summary*.txt".
    """
    pattern = os.path.join(summary_dir, "simulation_summary*.txt")
    files   = glob.glob(pattern)  # Recherche tous les fichiers correspondants
    files.sort()                  # Tri alphabétique
    return files                  # Renvoie la liste triée

def list_exp_data_files(exp_data_dir):
    """
    Retourne la liste triée des fichiers expérimentaux présents dans exp_data_dir.
    Le pattern utilisé est "Data_structure*.txt".
    """
    pattern = os.path.join(exp_data_dir, "Data_structure*.txt")
    files   = glob.glob(pattern)
    files.sort()
    return files

def get_simulation_label(base_label, file_path, label_to_tag):
    """
    Construit un label unique pour une simulation à partir du nom de base et du nom du fichier.
    Ce label est enrichi avec une version extraite du nom de fichier et un compteur
    pour éviter les doublons.
    """
    fname = os.path.basename(file_path)      # Extrait le nom de fichier
    tag   = os.path.splitext(fname)[0]       # Enlève l’extension
    prefix = "simulation_summary_RCWA_"
    version = ""
    if tag.startswith(prefix):
        # Si le tag commence par le préfixe, extraire la partie version
        remainder = tag[len(prefix):]
        parts     = remainder.split("_", 1)
        if parts:
            version = parts[0]
    # Initialisation du sous‑dico pour ce base_label
    if base_label not in label_to_tag:
        label_to_tag[base_label] = {}
    if version not in label_to_tag[base_label]:
        # Premier exemplaire de cette version
        label_to_tag[base_label][version] = 1
        return f"{base_label} ({version})" if version else base_label
    else:
        # Version déjà vue : incrémenter le compteur
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
    spectra     = {}  # Dictionnaire final des spectres simulés et exp.
    summaries   = {}  # Dictionnaire final des résumés (geometry, material)
    label_to_tag= {}  # Pour gérer les doublons de labels
    metrics_dict = {}
    
    sim_files   = list_sim_summary_files(summary_dir)  # Liste des fichiers simulation
    
    for fpath in sim_files:
        combos     = read_all_combos(fpath)             # Lire les points Rpup par combo
        sim_configs= parse_simulation_summary(fpath)    # Lire les metadata par combo
        for combo_label, (wl, R) in combos.items():
            base_label = combo_label.replace(" - ", "\n")
            if base_label in spectra:
                # Si ce base_label existe déjà, on génère un nouveau label unique
                new_label = get_simulation_label(base_label, fpath, label_to_tag)
            else:
                # Sinon, on initialise le mapping et on crée le premier label
                fname = os.path.basename(fpath)
                tag   = os.path.splitext(fname)[0]
                prefix= "simulation_summary_RCWA_"
                if tag.startswith(prefix):
                    remainder = tag[len(prefix):]
                    parts     = remainder.split("_", 1)
                    tag       = parts[0] if parts else ""
                label_to_tag[base_label] = {tag: 1}
                new_label = f"{base_label} ({tag})" if tag else base_label
            spectra[new_label] = (wl, R)  # Stockage du spectre simulé
            # Recherche du summary correspondant dans sim_configs
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
                            key       = entry.get("key", "")
                            disp_name = key
                            for k, dname in ordered_params:
                                if k == key:
                                    disp_name = dname
                                    break
                            mat_info = entry.get("material", {})
                            mtype    = mat_info.get("type", "").strip().lower()
                            if mtype == "standard":
                                val = mat_info.get("material", "").strip()
                            elif mtype == "custom":
                                val = mat_info.get("expression", "").strip()
                            elif mtype == "refractiveindex":
                                book = mat_info.get("book", "")
                                page = mat_info.get("page", "")
                                val = f"Book: {book}, Page: {page}"                                
                            else:
                                val = ""
                            if val:
                                mat_lines.append(f"{disp_name}: {val}")
                    mat_summary = "\n".join(mat_lines)
                    summaries[new_label] = (geom_summary, mat_summary)
                    
                    # metrics
                    metrics = cfg.get("metrics", {})
                    metrics_dict[new_label] = metrics
                    
                    found = True
                    break
            if not found:
                # Si aucun summary n’a été trouvé, on met des champs vides
                summaries[new_label] = ("", "")
                metrics_dict[new_label] = {}
                
    # Lecture des fichiers expérimentaux
    exp_files = list_exp_data_files(exp_data_dir)
    for fpath in exp_files:
        data     = read_experimental_data(fpath)          # (λ, R) expérimentaux
        if data:
            base_lbl = os.path.basename(fpath)
            spectra[base_lbl]   = data                    # Ajout au dict spectra
            exp_data    = parse_experimental_data_summary(fpath)
            geom_summary= exp_data.get("geometry", "")
            mat_summary = exp_data.get("material", "")
            summaries[base_lbl] = (geom_summary, mat_summary)

    return spectra, summaries, metrics_dict  # Retourne les deux dictionnaires finaux
