
from . import nkexplorer


"""
nkexplorer_interface.py

Ce module sert d'interface pour solliciter nkexplorer depuis le notebook.
Il importe le module nkexplorer (qui ne doit pas être modifié) et appelle
sa fonction principale pour récupérer les données de matériaux de type RefractiveIndex.
La fonction launch_nkexplorer() retourne un dictionnaire dont les clés sont
les rôles (par exemple "perm_env", "perm_dielec", etc.) et les valeurs des dictionnaires
contenant 'shelf', 'book' et 'page'.
"""

def launch_nkexplorer():
    try:
        import nkexplorer
    except ImportError:
        raise ImportError("Le module nkexplorer n'a pas été trouvé dans le dossier tools.")
    
    # On suppose que nkexplorer fournit une fonction (par exemple run_explorer ou main)
    # qui lance son interface et retourne un dictionnaire des données.
    try:
        nk_data = nkexplorer.run_explorer()
    except AttributeError:
        # Si run_explorer n'est pas disponible, on tente avec main()
        nk_data = nkexplorer.main()
    
    if not isinstance(nk_data, dict):
        raise ValueError("nkexplorer n'a pas retourné un dictionnaire valide.")
    
    return nk_data
