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

import os, glob, re, ast
from collections import defaultdict
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import Any, Dict, List, Optional

# ------------------------------------------------------------------ #
#                       extraction utilitaires                       #
# ------------------------------------------------------------------ #
def _extract_points(line):
    """
    Détecte et renvoie (λ, Rup, Rup_dn) à partir d’une ligne du type :
        λ=650.0 nm -> Rup=0.123, Rup_dn=0.118
    ou (ancienne version) :
        λ=650.0 nm -> Rup=0.123
    """
    m = re.search(
        r"λ\s*=\s*([\d\.]+)\s*nm\s*->\s*Rup\s*=\s*([\d\.eE\+\-]+)"
        r"(?:\s*,\s*Rup_dn\s*=\s*([\d\.eE\+\-]+))?", line)
    if not m:
        return None
    lam  = float(m.group(1))
    Rup  = float(m.group(2))
    Rup_dn   = float(m.group(3)) if m.group(3) is not None else None
    return lam, Rup, Rup_dn



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


# ------------------------------------------------------------------ #
#                  lecture d’un fichier  summary RCWA                #
# ------------------------------------------------------------------ #
# deux regex compactes (plus lisibles que la précédente)
_re_rup    = re.compile(r"λ\s*=\s*([\d\.]+).*?Rup\s*=\s*([\d\.Ee\+\-]+)")
_re_rup_dn = re.compile(r"λ\s*=\s*([\d\.]+).*?Rup_dn\s*=\s*([\d\.Ee\+\-]+)")

def read_all_combos(path):
    """
    Renvoie deux dictionnaires synchronisés :
      • combos_Rup    {combo → (λ_array, Rup_array)}
      • combos_Rup_dn {combo → (λ_array, Rup_dn_array)}  (clé absente si le
        bloc « Rup_dn » n’existe pas pour la combo)
    """
    combos_Rup, combos_Rup_dn = {}, {}

    # ---------- lecture brute du fichier ---------------------------
    with open(path, encoding="utf‑8") as f:
        lines = [ln.rstrip() for ln in f]

    # ---------- parse ligne par ligne ------------------------------
    cur_name, section = None, None                   # section = None | 'rup' | 'rupdn'
    λ_buf, rup_buf, rupdn_buf = [], [], []

    def flush():
        """Enregistre les buffers courants dans les deux dictionnaires"""
        if cur_name and rup_buf:                     # on a au moins Rup
            λ = np.asarray(λ_buf, float)
            combos_Rup[cur_name] = (λ, np.asarray(rup_buf, float))
            if rupdn_buf:                            # bloc « Rup_dn » présent
                combos_Rup_dn[cur_name] = (λ, np.asarray(rupdn_buf, float))
                
    for raw in lines:
        ln = raw.strip()                      #  enlève les blancs à gauche/droite

        # ---------- nouvelle combo ----------
        if ln.startswith("Combo name:"):
            flush()
            cur_name = ln.split("Combo name:")[1].strip()
            λ_buf, rup_buf, rupdn_buf = [], [], []
            section = None
            continue

        # ---------- détection de bloc ----------
        if ln.startswith("Reflectance points"):
            section = "rupdn" if "Rup_dn" in ln else "rup"
            continue

        # ---------- points Rup ----------
        if section == "rup":
            m = _re_rup.match(ln)             # ln est déjà ‘strippé’ -> OK
            if m:
                λ_buf .append(float(m.group(1)))
                rup_buf.append(float(m.group(2)))
            continue

        # ---------- points Rup_dn ----------
        if section == "rupdn":
            m = _re_rup_dn.match(ln)
            if m:
                rupdn_buf.append(float(m.group(2)))
            continue

    flush()                                           # dernière combo
    return combos_Rup, combos_Rup_dn


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





from gap_plasmon_2d.materials.material__configuration import build_material_configuration_dynamic
def get_baseline_n(cfg, layer_key, lam_ref, json_combined_path):
    """
    Extrait l'indice de réfraction de base n0 pour la couche `layer_key`,
    en utilisant build_material_configuration_dynamic pour tous les types.

    Parameters
    ----------
    cfg : dict
        Configuration complète d'une combo, avec cfg['material']['MATERIALS_CONFIG'].
    layer_key : str
        Clé de la couche ciblée (ex. "perm_gap", "perm_reso", …).
    lam_ref : float
        Longueur d'onde de référence (en nm) à laquelle on évalue epsilon
        (typiquement lam_dip ou la moyenne de lam).
    json_combined_path : str
        Chemin vers le JSON combiné (passé à build_material_configuration_dynamic).

    Returns
    -------
    float
        Indice n0 pour la couche layer_key.
    """

    # 1) On construit un DataFrame pandas à partir de la liste MATERIALS_CONFIG
    df_config = pd.DataFrame(cfg['material']['MATERIALS_CONFIG'])

    # 2) On appelle build_material_configuration_dynamic pour obtenir ε(role) → permittivité
    #    On ne fournit pas d'override ici, puisque c'est la valeur de base.
    eps_dict = build_material_configuration_dynamic(
        df_config,
        lam_ref,
        json_combined_path,
        ri_overrides=None
    )

    # 3) On récupère la permittivité complexe (ou réelle) de la couche layer_key
    if layer_key not in eps_dict:
        raise KeyError(f"Couche '{layer_key}' introuvable dans eps_dict renvoyé")
    eps = eps_dict[layer_key]

    # 4) On extrait l'indice n0 : sqrt de la partie réelle de ε
    #    (si ε est complexe, on prend la partie réelle pour n)
    n0 = np.sqrt(np.real(eps))

    return float(n0)








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
    
    

# ------------------------------------------------------------------ #
#            agrégation complète pour l’interface graphique           #
# ------------------------------------------------------------------ #
def get_all_spectra_and_summaries(summary_dir, exp_data_dir, ordered_params):
    """
    Retourne quatre dictionnaires synchronisés :
        Rup_dict, Rup_dn_dict, summaries, metrics
    """

    Rup_dict, Rup_dn_dict = {}, {}
    summaries, metrics_dict, delta_n_dict = {}, {}, {}
    label_to_tag = {}

    # ---------- fichiers de simulation ------------------------------
    for fpath in list_sim_summary_files(summary_dir):
        combos_Rup, combos_Rup_dn = read_all_combos(fpath)
        sim_cfgs = parse_simulation_summary(fpath)

        for combo_name, (λ, Rup) in combos_Rup.items():
            base = combo_name.replace(" - ", "\n")
            label = get_simulation_label(base, fpath, label_to_tag)

            Rup_dict[label] = (λ, Rup)
            if combo_name in combos_Rup_dn:
                Rup_dn_dict[label] = combos_Rup_dn[combo_name]

            # --------- résumé geometry / material -------------------
            match_cfg = next((c for c in sim_cfgs
                              if c["label"].replace(" - ", "\n") == base), None)
            if match_cfg:
                # geometry
                geom_lines = [f"{d}: {match_cfg['geometry'].get(k)}"
                              for k, d in ordered_params
                              if k in match_cfg['geometry']]
                # material
                mat_lines  = []
                for entry in match_cfg['material']:
                    key = entry['key']
                    disp = next((d for k, d in ordered_params if k == key), key)
                    mat = entry['material']; typ = mat['type'].lower()
                    if typ == "standard":  val = mat['material']
                    elif typ == "custom":  val = mat['expression']
                    else:                  val = f"Book: {mat.get('book','')}, Page: {mat.get('page','')}"
                    mat_lines.append(f"{disp}: {val}")
                summaries[label] = ("\n".join(geom_lines), "\n".join(mat_lines))
                metrics_dict[label] = match_cfg.get("metrics", {})
                # ←–– extraire Δn depuis les metrics (clé "Δn")
                dn_str = metrics_dict[label].get("Δn")
                try:
                    delta_n_dict[label] = float(dn_str)
                except Exception:
                    delta_n_dict[label] = None
            else:
                summaries[label] = ("", "")
                metrics_dict[label] = {}
                delta_n_dict[label] = None

    # ---------- fichiers expérimentaux ------------------------------
    for fpath in list_exp_data_files(exp_data_dir):
        try:
            λ, Rexp = read_experimental_data(fpath)
        except Exception:
            continue
        lbl = os.path.basename(fpath)
        Rup_dict[lbl] = (λ, Rexp)
        summaries[lbl] = tuple(parse_experimental_data_summary(fpath).values())
        metrics_dict[lbl] = {}
        delta_n_dict[lbl] = None
        # (pas de spectre Rup_dn pour les données expérimentales)

    return Rup_dict, Rup_dn_dict, summaries, metrics_dict, delta_n_dict





# --------------------------------------------------------------------------- #
#             Lecture d’un fichier HDF5 produit par save_optimization_hdf5   #
# --------------------------------------------------------------------------- #


def read_optimization_hdf5(
    h5path: str,
    run_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Charge un run d’optimisation dans un fichier *.h5* et renvoie toutes les
    données pertinentes dans un dictionnaire Python.

    Parameters
    ----------
    h5path
        Chemin vers le fichier HDF5.
    run_key
        Nom du groupe à extraire (ex. "budget10_pop05_20250627T153012").
        Si None, on prend le premier groupe disponible.

    Returns
    -------
    out : dict
        {
          'run_key': str,         # clé du groupe lu
          'config_name': str,
          'budget': int,
          'Npop': int,
          'mode': str,
          'best_cost': float,
          'keys': List[str],
          'lowers': np.ndarray,
          'uppers': np.ndarray,
          'conv_best': np.ndarray,
          'conv_evals': np.ndarray,
          'cf_final': np.ndarray,
          'best': np.ndarray,
          'best_final': np.ndarray,
          'best_after_eval': Optional[np.ndarray],
          'spectra': {
               'wavelength': np.ndarray,
               'Rup': np.ndarray,
               'Rdown': Optional[np.ndarray]
          }   # seulement si présent
        }
    """
    with h5py.File(h5path, "r") as f:
        # 1) sélection du groupe
        available = list(f.keys())
        if not available:
            raise RuntimeError(f"{h5path!r} ne contient aucun groupe de run")
        if run_key is None:
            run_key = available[0]
        elif run_key not in available:
            raise KeyError(f"run_key {run_key!r} non trouvé dans {h5path!r}")
        grp = f[run_key]

        # 2) helper pour décoder bytes → str
        def _decode(v):
            if isinstance(v, (bytes, np.bytes_)):
                return v.decode()
            return v

        # 3) lecture des attributs
        out: Dict[str, Any] = {
            "run_key": run_key,
            "config_name": _decode(grp.attrs.get("config_name", "")),
            "budget": int(grp.attrs["budget"]),
            "Npop": int(grp.attrs["Npop"]),
            "mode": _decode(grp.attrs["mode"]),
            "best_cost": float(grp.attrs["best_cost"]),
        }

        # -- fixe : renvoie None si absent ----------------------

        out["fixed_lambda"] = float(grp.attrs["fixed_lambda"]) \
            if "fixed_lambda" in grp.attrs else None

        out["n_modes"] = int(grp.attrs["n_modes"]) if "n_modes" in grp.attrs else None


        # 4) paramètres de l’espace de recherche
        params_grp = grp["parameters"]
        out.update({
            "keys": [_decode(k) for k in params_grp["keys"][:]],
            "lowers": params_grp["lowers"][:],
            "uppers": params_grp["uppers"][:],
        })

        # 5) convergence & population finale
        out["conv_best"] = grp["conv_best"][:]
        out["conv_evals"] = grp["conv_evals"][:]
        out["cf_final"] = grp["cf_final"][:]

        # 6) meilleurs individus
        out["best"] = grp["best"][:]
        out["best_final"] = grp["best_final"][:]
        if "best_after_eval" in grp:
            out["best_after_eval"] = grp["best_after_eval"][:]
        else:
            out["best_after_eval"] = None

        # 7) spectra 
        if "spectra" in grp:
            spec = grp["spectra"]
            spectra: Dict[str, Any] = {
                "wavelength": spec["wavelength"][:],
                "Rup": spec["Rup"][:]
            }
            if "Rdown" in spec:
                spectra["Rdown"] = spec["Rdown"][:]
            else:
                spectra["Rdown"] = None
            out["spectra"] = spectra

        # 8) fixed parameters 
        if "fixed" in grp:
            fx = grp["fixed"]
            keys = [k.decode() for k in fx["keys"][:]]
            vals = fx["values"][:]
            out["fixed"] = dict(zip(keys, vals))
        else:
            out["fixed"] = {}


        # 9) geometry  
        geom = {}
        if "geometry" in grp:
            g = grp["geometry"]
            if {"keys", "values"} <= set(g):
                # NOUVEAU format ordonné
                keys = [k.decode() for k in g["keys"][...]]
                vals = g["values"][...]
                geom = dict(zip(keys, vals))
            else:
                # Ancien format : attributs (ordre perdu)
                for k, v in g.attrs.items():
                    geom[k] = float(v)
        out["geometry"] = geom



    return out


# --------------------------------------------------------------------------- #
#                       Lister les fichiers d’optimisation                    #
# --------------------------------------------------------------------------- #
def list_optimization_files(summary_opt_dir: str) -> List[str]:
    p = Path(summary_opt_dir)
    files = sorted(
        p.rglob("*.h5"),
        key=lambda f: f.as_posix()
    )
    return [str(f) for f in files]




def list_runs_in_h5(h5path: str) -> List[str]:
    """
    Renvoie la liste des groupes (run_keys) présents dans le fichier HDF5.
    """
    try:
        with h5py.File(h5path, "r") as f:
            return list(f.keys())
    except Exception:
        return []
