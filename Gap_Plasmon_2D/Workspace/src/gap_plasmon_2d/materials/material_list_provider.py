import json
import os

def load_combined_materials(json_path):
    """
    Loads the combined JSON file containing data for ExpData and BrendelBormann.
    
    Parameters
    ----------
    json_path : str
        Path to the combined JSON file.
        
    Returns
    -------
    combined : dict
        Dictionary containing the materials data.
    """
    with open(json_path, 'r') as f:
        combined = json.load(f)
    return combined

def get_available_materials(json_path):
    """
    Returns the sorted list of available materials by loading the combined JSON file.
    
    Parameters
    ----------
    json_path : str
        Path to the combined JSON file.
        
    Returns
    -------
    all_materials : list
        Sorted list of available material names.
    """
    combined = load_combined_materials(json_path)
    all_materials = sorted(list(combined.keys()))
    return all_materials

def get_available_materials_extended(json_path, data_dir):
    """
    Returns a sorted list of available materials by combining the keys from the JSON file
    with the names (without extension) of any .txt files found recursively in data_dir.
    
    Parameters
    ----------
    json_path : str
        Path to the combined JSON file.
    data_dir : str
        Path to the directory containing material data files (e.g., TXT files).
    
    Returns
    -------
    all_materials : list
        Sorted list of available material names.
    """
    materials = set(get_available_materials(json_path))
    
    # Parcours récursif du dossier data pour trouver les fichiers .txt
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(".txt"):
                name = os.path.splitext(f)[0]  # Ex : "ITO" pour "ITO.txt"
                materials.add(name)
    
    return sorted(materials)


