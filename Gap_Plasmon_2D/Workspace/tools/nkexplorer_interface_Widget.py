# nkexplorer_interface.py
import ipywidgets as widgets
from IPython.display import display
from nkexplorer_widget import NKExplorerWidget, page_ids, page_names, page_paths

def launch_nkexplorer():
    """
    Lance l'interface NK Explorer en mode ipywidgets dans le Notebook.
    Affiche l'interface et ajoute un bouton « Valider ».
    Lorsque l'utilisateur clique sur le bouton, les sélections sont affichées dans la console.
    La fonction retourne un dictionnaire qui se met à jour via le callback.
    """
    explorer = NKExplorerWidget()
    
    submit_button = widgets.Button(description="Valider")
    output = widgets.Output()
    
    # On combine l'interface et le bouton dans un container vertical.
    container = widgets.VBox([explorer.ui, submit_button, output])
    display(container)
    
    # Dictionnaire qui contiendra les sélections
    result = {"shelf": None, "book": None, "page": None}
    
    def on_submit(b):
        result["shelf"] = explorer.shelf_dropdown.value
        result["book"] = explorer.book_dropdown.value
        # Pour la page, on parcourt les boutons radio et on prend celui qui est actif
        for i, rb in enumerate(explorer.page_radios):
            if rb.value:
                result["page"] = {"id": page_ids[i], "name": page_names[i], "data": page_paths[i]}
                break
        with output:
            output.clear_output()
            print("Sélection validée :", result)
    
    submit_button.on_click(on_submit)
    
    return result
