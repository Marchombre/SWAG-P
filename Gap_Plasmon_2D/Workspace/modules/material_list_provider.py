# material_list_provider.py
import json

def load_combined_materials(json_path):
    """
    Charge le fichier JSON combiné contenant les données pour ExpData et BrendelBormann.
    
    Paramètre
    ---------
    json_path : str
        Chemin vers le fichier JSON combiné.
        
    Retourne
    --------
    combined : dict
        Dictionnaire contenant les données des matériaux.
    """
    with open(json_path, 'r') as f:
        combined = json.load(f)
    return combined

def get_available_materials(json_path):
    """
    Retourne la liste triée des matériaux disponibles en chargeant le fichier JSON combiné.
    
    Paramètre
    ---------
    json_path : str
        Chemin vers le fichier JSON combiné.
        
    Retourne
    --------
    all_materials : list
        Liste triée des noms de matériaux disponibles.
    """
    combined = load_combined_materials(json_path)
    all_materials = sorted(list(combined.keys()))
    return all_materials
