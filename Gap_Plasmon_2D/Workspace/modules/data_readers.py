# data_readers.py

#!/usr/bin/env python3
"""
Module: data_readers.py

Ce module réunit toutes les fonctions de lecture utilisées dans votre projet.

Fonctions disponibles :
  - read_all_combos(file_path) : Extrait, depuis un fichier simulation_summary_XXX.txt,
    pour chaque combo (délimité par "Combo name:"), les points de réflectance Rup.
  - read_experimental_data(file_path) : Lit un fichier de données expérimentales en utilisant exactement votre procédé d'origine.
"""

import re
import numpy as np

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
    
    Cette fonction utilise exactement le même procédé de lecture que dans votre implémentation d'origine.
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
