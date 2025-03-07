# material_list_provider.py
import json

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
