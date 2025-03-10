import os
import ipywidgets as widgets
import pandas as pd
from IPython.display import display
import yaml

# Import de la fonction étendue depuis material_list_provider
from material_list_provider import get_available_materials_extended

# Liste par défaut des rôles utilisés dans la simulation
DEFAULT_ROLES = [
    "perm_env", "perm_dielec", "perm_sub", "perm_reso",
    "perm_metalliclayer", "perm_accroche", "perm_func", "perm_mol"
]

# Chemin local vers le fichier catalog_nk.yml
CATALOG_PATH = os.path.join(os.path.dirname(__file__), "catalog_nk.yml")

def get_catalog_options_local(catalog_path):
    """
    Charge le fichier catalog_nk.yml et extrait les listes de shelves, books et pages,
    ainsi que les dictionnaires pour l'autocomplétion interdépendante.
    """
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = yaml.load(f, Loader=yaml.BaseLoader)
    shelves = set()
    shelf_to_books = {}
    book_to_pages = {}
    for sh in catalog:
        if "SHELF" in sh:
            shelf_name = sh["SHELF"]
            shelves.add(shelf_name)
            shelf_to_books[shelf_name] = []
            for bk in sh.get("content", []):
                if "BOOK" in bk:
                    book_name = bk["BOOK"]
                    shelf_to_books[shelf_name].append(book_name)
                    book_to_pages[(shelf_name, book_name)] = []
                    for pg in bk.get("content", []):
                        if "PAGE" in pg:
                            page_name = pg["PAGE"]
                            book_to_pages[(shelf_name, book_name)].append(page_name)
    return (sorted(shelves),
            sorted({b for books in shelf_to_books.values() for b in books}),
            sorted({p for p_list in book_to_pages.values() for p in p_list}),
            shelf_to_books,
            book_to_pages)

# Charge les options du catalogue local.
CATALOG_SHELVES, CATALOG_BOOKS, CATALOG_PAGES, shelf_to_books, book_to_pages = get_catalog_options_local(CATALOG_PATH)

# Définissez ici le chemin vers le dossier data
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def create_material_selector(json_path, roles=DEFAULT_ROLES):
    """
    Crée une interface interactive (dropdowns et champs supplémentaires)
    pour la sélection des matériaux.
    
    La liste des matériaux disponibles est générée dynamiquement à partir du fichier JSON combiné
    ET des fichiers .txt présents dans le dossier data.
    
    En plus de "None" et "Custom", le dropdown inclut l'option "RefractiveIndex" qui permet de spécifier
    shelf, book et page pour un matériau depuis la base refractiveindex.info.
    
    La configuration finale est stockée dans la variable globale MATERIALS_CONFIG,
    et les éventuels overrides pour RefractiveIndex dans RI_OVERRIDES.
    """
    # Utilisation de la fonction étendue pour inclure les fichiers TXT
    all_materials = get_available_materials_extended(json_path, DATA_DIR)
    all_materials_with_options = ["None", "Custom", "RefractiveIndex"] + all_materials

    dropdowns = {}
    custom_inputs = {}  # Pour l'option "Custom"
    ri_inputs = {}      # Pour l'option "RefractiveIndex": tuple (shelf, book, page)
    widget_boxes = []

    def make_on_change(custom_widget, ri_box_widget):
        def _on_change(change):
            if change['new'] == "Custom":
                custom_widget.layout.display = 'block'
                ri_box_widget.layout.display = 'none'
                for child in ri_box_widget.children:
                    child.layout.display = 'none'
            elif change['new'] == "RefractiveIndex":
                custom_widget.layout.display = 'none'
                custom_widget.value = ""
                for child in ri_box_widget.children:
                    child.layout.display = 'block'
                ri_box_widget.layout.display = 'flex'
            elif change['new'] == "None":
                custom_widget.layout.display = 'none'
                custom_widget.value = ""
                ri_box_widget.layout.display = 'none'
                for child in ri_box_widget.children:
                    child.layout.display = 'none'
        return _on_change

    def on_shelf_change(change, book_widget, page_widget):
        selected_shelf = change['new']
        if selected_shelf in shelf_to_books:
            book_widget.options = sorted(shelf_to_books[selected_shelf])
        else:
            book_widget.options = []
        page_widget.options = []
        page_widget.value = ""

    def on_book_change(change, shelf_widget, page_widget):
        selected_book = change['new']
        if shelf_widget.value:
            key = (shelf_widget.value, selected_book)
            page_widget.options = sorted(book_to_pages.get(key, []))
        else:
            page_widget.options = []
        page_widget.value = ""

    for role in roles:
        dropdown = widgets.Dropdown(
            options=all_materials_with_options,
            description=role,
            style={'description_width': 'initial'}
        )
        custom_text = widgets.Text(
            value="",
            description="Custom:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(display='none')
        )
        ri_shelf = widgets.Combobox(
            placeholder="Select shelf",
            options=CATALOG_SHELVES,
            description="Shelf:",
            ensure_option=True,
            layout=widgets.Layout(width='120px', display='none')
        )
        ri_book = widgets.Combobox(
            placeholder="Select book",
            options=[],
            description="Book:",
            ensure_option=True,
            layout=widgets.Layout(width='120px', display='none')
        )
        ri_page = widgets.Combobox(
            placeholder="Select page",
            options=[],
            description="Page:",
            ensure_option=True,
            layout=widgets.Layout(width='120px', display='none')
        )
        ri_box = widgets.HBox([ri_shelf, ri_book, ri_page])
        ri_box.layout.display = 'none'
        
        ri_shelf.observe(lambda change, bw=ri_book, pw=ri_page: on_shelf_change(change, bw, pw), names='value')
        ri_book.observe(lambda change, sw=ri_shelf, pw=ri_page: on_book_change(change, sw, pw), names='value')
        
        dropdown.observe(make_on_change(custom_text, ri_box), names='value')
        
        dropdowns[role] = dropdown
        custom_inputs[role] = custom_text
        ri_inputs[role] = (ri_shelf, ri_book, ri_page)
        
        role_box = widgets.HBox([dropdown, custom_text, ri_box])
        widget_boxes.append(role_box)
    
    button_create_df = widgets.Button(description="Validate Materials Configuration")
    output_df = widgets.Output()
    
    def on_create_df(b):
        config = {"key": [], "material": []}
        ri_overrides = {}
        for role in roles:
            value = dropdowns[role].value
            if value == "Custom":
                mat_value = custom_inputs[role].value.strip()
                if mat_value == "":
                    mat_value = "None"
            elif value == "RefractiveIndex":
                shelf = ri_inputs[role][0].value.strip()
                book = ri_inputs[role][1].value.strip()
                page = ri_inputs[role][2].value.strip()
                if shelf == "" or book == "" or page == "":
                    mat_value = "None"
                else:
                    mat_value = f"RefractiveIndex: {book}"
                    ri_overrides[role] = {"shelf": shelf, "book": book, "page": page}
            else:
                mat_value = value
            config["key"].append(role)
            config["material"].append(mat_value)
        df_config = pd.DataFrame(config)
        with output_df:
            output_df.clear_output()
            print("Selected materials configuration:")
            display(df_config)
        import __main__
        __main__.MATERIALS_CONFIG = df_config
        __main__.RI_OVERRIDES = ri_overrides
        print("MATERIALS_CONFIG and RI_OVERRIDES have been defined in the global __main__ space")
    
    button_create_df.on_click(on_create_df)
    
    widget_container = widgets.VBox(widget_boxes + [button_create_df, output_df])
    return widget_container
