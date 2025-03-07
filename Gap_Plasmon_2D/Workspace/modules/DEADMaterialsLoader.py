import json
import os
import numpy as np

# By default, it is assumed that the file is located in a 'data' folder at the root of the project.
JSON_PATH = os.path.abspath(os.path.join('/home/chardon-grossard/Bureau/SWAG-P/Gap_Plasmon_2D/Workspace/', 'data', 'BB_materials.json'))

def load_materials():
    """
    Loads the material parameters from the JSON file.
    """
    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError(f"JSON file not found: {JSON_PATH}")
    
    with open(JSON_PATH, "r") as file:
        materials_data = json.load(file)
    
    return materials_data

def get_material_params(material_name, materials_data):
    """
    Retrieves the parameters for a given material from the JSON file.
    Also returns the model (for example, "BrendelBormann") to allow choosing the calculation method.
    """
    if material_name not in materials_data:
        raise ValueError(f"Material '{material_name}' not found in the JSON file.")

    material = materials_data[material_name]

    f0 = material["f0"]
    Gamma0 = material["Gamma0"]
    omega_p = material["omega_p"]
    f = np.array(material["f"])
    omega = np.array(material["omega"])
    gamma = np.array(material["Gamma"])
    sigma = np.array(material["sigma"])
    model = material.get("model", "BrendelBormann")  # By default, we consider "BrendelBormann"
    
    return f0, omega_p, Gamma0, f, omega, gamma, sigma, model
