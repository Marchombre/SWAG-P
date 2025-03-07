# MaterialsConfiguration.py
import json
import pandas as pd
from Functions_ExpData import get_n_k, compute_permittivity

def load_combined_materials(json_path):
    """
    Loads the combined JSON file containing ExpData and BrendelBormann data.
    
    Returns a dictionary of materials.
    """
    with open(json_path, 'r') as f:
        materials_data = json.load(f)
    return materials_data

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

def build_material_configuration_dynamic(df_config, lambda_val, json_path):
    """
    Constructs a dictionary of permittivities from a configuration DataFrame for a given wavelength,
    by loading a combined JSON file containing ExpData and BrendelBormann data.
    
    For each material in the DataFrame (columns "key" and "material"):
      - If the value is "None" (case-insensitive), assign 1.0 (equivalent to air).
      - Otherwise, if the material is found in the combined JSON file (case-insensitive comparison)
        and its "model" field is "expdata", use get_n_k to interpolate n and k and compute ε = (n+ik)².
      - Otherwise, if the material is found in the combined JSON file and its "model" is something else
        (e.g., "brendelbormann"), use get_material_params then compute_permittivity.
      - Otherwise, attempt to convert the value to float (for a custom constant).
      - Otherwise, an error is raised.
    """
    materials_data = load_combined_materials(json_path)
    # Build a case-insensitive lookup dictionary
    available_materials = {k.lower(): k for k in materials_data.keys()}
    
    materials_perm = {}
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
                # Processing via ExpData
                n_val, k_val = get_n_k(actual_mat, lambda_val, json_path)
                perm = (n_val + 1j * k_val) ** 2
                materials_perm[key] = perm
            else:
                # Processing via the BB model
                try:
                    f0, omega_p, Gamma0, f, omega, gamma, sigma, model = get_material_params(actual_mat, materials_data)
                    perm = compute_permittivity(lambda_val, f0, omega_p, Gamma0, f, omega, gamma, sigma, N=50)
                    materials_perm[key] = perm
                except KeyError as e:
                    raise ValueError(f"The parameters for '{actual_mat}' are incomplete: {e}")
        else:
            try:
                const_val = float(mat)
                materials_perm[key] = const_val
            except ValueError:
                raise ValueError(f"Material '{mat}' not found in the combined JSON file, and cannot be interpreted as a constant.")
    return materials_perm
