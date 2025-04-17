import os
import yaml
import json
import glob
import ipywidgets as widgets
from IPython.display import display, clear_output
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================================
# Définition des chemins globaux
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
WORKSPACE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(WORKSPACE_DIR, "data")
CONFIGURATIONS_DIR = os.path.join(WORKSPACE_DIR, "CONFIGURATIONS")
CATALOG_PATH = os.path.join(WORKSPACE_DIR, "catalog_nk.yml")
JSON_COMBINED_PATH = os.path.join(DATA_DIR, "combined_materials.json")

# ============================================================================
# Fonctions utilitaires
# ============================================================================

def html_sub_to_unicode(text):
    """
    Convertit les balises HTML <sub>...</sub> en indices Unicode.
    """
    subscripts = {"0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
                  "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉"}
    while "<sub>" in text and "</sub>" in text:
        start = text.find("<sub>")
        end = text.find("</sub>", start)
        if start == -1 or end == -1:
            break
        sub_text = text[start+5:end]
        converted = "".join(subscripts.get(ch, ch) for ch in sub_text)
        text = text[:start] + converted + text[end+6:]
    return text

def load_catalog_full(catalog_file):
    """
    Charge le fichier YAML du catalogue complet.
    """
    with open(catalog_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_combined_materials(json_combined_path):
    """
    Charge le fichier JSON contenant la configuration combinée des matériaux.
    """
    with open(json_combined_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_standard_materials(json_combined_path, data_directory):
    """
    Recherche et retourne la liste triée des matériaux standards,
    en extrayant les clés du JSON et les fichiers .txt dans le dossier de données.
    """
    found = set()
    if os.path.isfile(json_combined_path):
        try:
            materials_data = load_combined_materials(json_combined_path)
            found.update(materials_data.keys())
        except Exception as e:
            print(f"[WARNING] Problem reading JSON '{json_combined_path}': {e}")
    else:
        print(f"[WARNING] JSON file '{json_combined_path}' not found.")
    if os.path.isdir(data_directory):
        for root, dirs, files in os.walk(data_directory):
            for fn in files:
                if fn.lower().endswith(".txt"):
                    name = os.path.splitext(fn)[0]
                    found.add(name)
    else:
        print(f"[WARNING] Data directory '{data_directory}' is not valid.")
    return sorted(found)

def get_lambda_range_from_txt(material_name, data_dir):
    """
    Recherche dans data_dir un fichier texte correspondant à material_name et
    retourne (lambda_min, lambda_max) en nm.
    Les longueurs d'onde dans le fichier sont supposées être en µm.
    """
    pattern = os.path.join(data_dir, f"{material_name}.txt")
    txt_files = glob.glob(pattern)
    if not txt_files:
        pattern = os.path.join(data_dir, f"*{material_name}*.txt")
        txt_files = glob.glob(pattern)
        if not txt_files:
            raise ValueError(f"Le fichier texte pour '{material_name}' n'a pas été trouvé dans {data_dir}.")
    txt_file = txt_files[0]
    with open(txt_file, "r") as f:
        lines = f.readlines()
    nb_lines = len(lines)
    wl_data = []
    for idx in range(2, nb_lines - 2):
        line = lines[idx].strip()
        if line:
            try:
                vals = [float(v) for v in line.split()]
                if len(vals) >= 3:
                    wl_data.append(vals[0])
            except ValueError:
                continue
    if not wl_data:
        raise ValueError(f"Aucune donnée de longueur d'onde trouvée dans le fichier pour '{material_name}'.")
    wl_data = np.array(wl_data)  # en µm
    return float(np.min(wl_data) * 1000), float(np.max(wl_data) * 1000)

def get_lambda_bounds(material_name, json_combined_path, data_dir):
    """
    Retourne les bornes (lambda_min, lambda_max) en nm pour le matériau identifié par material_name.
    """
    with open(json_combined_path, "r", encoding="utf-8") as f:
        materials_data = json.load(f)
    material_lower = material_name.lower()
    found = None
    for key in materials_data:
        if key.lower() == material_lower:
            found = key
            break
    if found is not None:
        material = materials_data[found]
        model = material.get("model", "").strip().lower()
        if model == "expdata":
            if "wavelength_range" in material and isinstance(material["wavelength_range"], list) and len(material["wavelength_range"]) >= 2:
                return float(material["wavelength_range"][0]), float(material["wavelength_range"][1])
            elif "wavelength_list" in material and isinstance(material["wavelength_list"], list) and material["wavelength_list"]:
                wl = material["wavelength_list"]
                return float(min(wl)), float(max(wl))
            else:
                return get_lambda_range_from_txt(found, data_dir)
        elif model == "brendelbormann":
            if "wavelength_range" in material and isinstance(material["wavelength_range"], list) and len(material["wavelength_range"]) >= 2:
                return float(material["wavelength_range"][0]), float(material["wavelength_range"][1])
            else:
                return None
        else:
            raise ValueError(f"Modèle '{material.get('model')}' non supporté pour le matériau '{found}'.")
    else:
        try:
            return get_lambda_range_from_txt(material_name, data_dir)
        except Exception as e:
            raise ValueError(f"Matériau '{material_name}' non trouvé dans le JSON ni via un fichier texte : {e}")

def get_lambda_bounds_refractiveindex(config_identifier, data_dir):
    """
    Pour un matériau de type RefractiveIndex, récupère la plage en λ (en nm)
    en se basant sur la configuration (dict) contenant les clés 'shelf', 'book', 'page' et éventuellement 'data'.
    """
    from refractiveindexINFO import RefractiveIndex, Material
    if not isinstance(config_identifier, dict):
        raise ValueError("Pour un matériau RefractiveIndex, une configuration dict est attendue.")
    shelf = config_identifier.get("shelf", "").strip()
    book  = config_identifier.get("book", "").strip()
    page  = config_identifier.get("page", "").strip()
    data_field = config_identifier.get("data", "").strip()
    if data_field:
        filename = os.path.join(data_dir, data_field)
        if not os.path.exists(filename):
            raise ValueError(f"Le fichier spécifié dans 'data' n'existe pas : {filename}")
    else:
        RI_instance = RefractiveIndex()
        filename = RI_instance.getMaterialFilename(shelf, book, page)
        if not filename:
            raise ValueError(f"Impossible de trouver le fichier pour shelf '{shelf}', book '{book}', page '{page}'.")
    mat_instance = Material(filename)
    if hasattr(mat_instance, "originalData") and isinstance(mat_instance.originalData, dict):
        if "wavelength (um)" in mat_instance.originalData:
            wavelengths_um = mat_instance.originalData["wavelength (um)"]
            if len(wavelengths_um) == 0:
                raise ValueError("Aucune donnée de longueur d'onde trouvée dans le matériau.")
            return float(np.min(wavelengths_um) * 1000), float(np.max(wavelengths_um) * 1000)
    if hasattr(mat_instance, "getWavelengthBounds"):
        bounds = mat_instance.getWavelengthBounds()
        if isinstance(bounds, (tuple, list)) and len(bounds) == 2:
            return float(bounds[0]), float(bounds[1])
    raise ValueError("Aucune donnée de plage de longueur d'onde trouvée pour le matériau RefractiveIndex.")

def compute_epsilon(n_func, k_func, lam_range):
    """
    Calcule ε(λ) à partir des fonctions n(λ) et k(λ).
    """
    return (n_func(lam_range) + 1j * k_func(lam_range))**2

# ============================================================================
# Widgets pour la navigation dans le catalogue RefractiveIndex
# ============================================================================

class RefractiveIndexArboWidget:
    """
    Widget permettant de naviguer dans le catalogue YAML de refractiveindex.
    """
    def __init__(self, library):
        self.library = library
        self.shelf_dropdown = widgets.Dropdown(description="Shelf:")
        self.book_dropdown = widgets.Dropdown(description="Book:")
        self.page_dropdown = widgets.Dropdown(description="Page:")
        self.container = widgets.HBox([self.shelf_dropdown, self.book_dropdown, self.page_dropdown])
        self._populate_shelf()
        self.shelf_dropdown.observe(self.on_shelf_changed, names="value")
        self.book_dropdown.observe(self.on_book_changed, names="value")
    
    def _populate_shelf(self):
        options = []
        for i, entry in enumerate(self.library):
            if "SHELF" in entry:
                raw_disp = entry.get("name", entry["SHELF"])
                disp = html_sub_to_unicode(raw_disp)
                options.append((disp, i))
            elif "DIVIDER" in entry:
                options.append((f"—— {entry['DIVIDER']} ——", None))
        self.shelf_dropdown.options = options
    
    def on_shelf_changed(self, change):
        val = change["new"]
        if val is None:
            self.book_dropdown.options = []
            self.page_dropdown.options = []
            return
        shelf_item = self.library[val]
        content = shelf_item.get("content", [])
        book_options = []
        for j, bk in enumerate(content):
            if "BOOK" in bk:
                raw_disp = bk.get("name", bk["BOOK"])
                disp = html_sub_to_unicode(raw_disp)
                book_options.append((disp, j))
            elif "DIVIDER" in bk:
                book_options.append((f"—— {bk['DIVIDER']} ——", None))
        self.book_dropdown.options = book_options
        self.page_dropdown.options = []
    
    def on_book_changed(self, change):
        book_val = change["new"]
        shelf_val = self.shelf_dropdown.value
        if shelf_val is None or book_val is None:
            self.page_dropdown.options = []
            return
        shelf_item = self.library[shelf_val]
        shelf_content = shelf_item.get("content", [])
        if not (0 <= book_val < len(shelf_content)):
            self.page_dropdown.options = []
            return
        book_dict = shelf_content[book_val]
        page_options = []
        for pg in book_dict.get("content", []):
            if "PAGE" in pg:
                raw_disp = html_sub_to_unicode(pg.get("name", pg["PAGE"]))
                page_options.append((raw_disp, pg["PAGE"]))
            elif "DIVIDER" in pg:
                page_options.append((f"—— {pg['DIVIDER']} ——", None))
        self.page_dropdown.options = page_options
        if page_options:
            for opt in page_options:
                if opt[1] is not None:
                    self.page_dropdown.value = opt[1]
                    break
    
    def get_selection(self):
        shelf_val = self.shelf_dropdown.value
        if shelf_val is None:
            return None
        shelf_item = self.library[shelf_val]
        shelf_key = shelf_item["SHELF"]
        book_val = self.book_dropdown.value
        if book_val is None:
            return None
        shelf_content = shelf_item.get("content", [])
        if not (0 <= book_val < len(shelf_content)):
            return None
        book_dict = shelf_content[book_val]
        if "BOOK" not in book_dict:
            return None
        book_key = book_dict["BOOK"]
        selected_page_value = self.page_dropdown.value
        if selected_page_value is None:
            return None
        page_dict = next((pg for pg in book_dict.get("content", []) 
                          if "PAGE" in pg and pg["PAGE"] == selected_page_value), None)
        page_name = page_dict.get("name", page_dict["PAGE"]) if page_dict else selected_page_value
        return {
            "shelf": shelf_key,
            "book": book_key,
            "page": selected_page_value,
            "data": page_dict.get("data", "") if page_dict else ""
        }
    
    def set_selection(self, selection):
        shelf_index = next((i for i, entry in enumerate(self.library)
                            if entry.get("SHELF", "") == selection.get("shelf", "")), None)
        if shelf_index is None:
            return
        self.shelf_dropdown.value = shelf_index
        self.on_shelf_changed({"new": shelf_index})
        book_set = False
        for option in self.book_dropdown.options:
            idx = option[1]
            if (option[0] == selection.get("book", "") or 
                (idx is not None and self.library[self.shelf_dropdown.value].get("content", [])[idx].get("BOOK", "") == selection.get("book", ""))):
                self.book_dropdown.value = idx
                book_set = True
                break
        if book_set:
            self.on_book_changed({"new": self.book_dropdown.value})
        for option in self.page_dropdown.options:
            if option[1] == selection.get("page", ""):
                self.page_dropdown.value = option[1]
                break

# ============================================================================
# Widget pour configurer un matériau comparé avec boutons "Draw" et "Delete materials"
# ============================================================================
class ComparisonMaterialWidget:
    def __init__(self, standard_list, library, remove_callback=None):
        self.standard_list = standard_list
        self.library = library
        self.remove_callback = remove_callback

        # Remplacer "Comparer" par "Draw" et "Supprimer" par "Delete materials"
        self.mode_dropdown = widgets.Dropdown(
            options=["Standard", "Custom", "RefractiveIndex"],
            value="Standard",
            description="Mode:"
        )
        self.custom_text = widgets.Text(placeholder="Enter expression", description="Expr:")
        self.standard_dropdown = widgets.Dropdown(options=self.standard_list, description="Standard:")
        self.ri_widget = RefractiveIndexArboWidget(self.library)
        self.draw_btn = widgets.Button(description="Draw", button_style="info")
        self.remove_btn = widgets.Button(description="Delete materials", button_style="danger", layout=widgets.Layout(width='auto'))
        self.config_box = widgets.HBox([self.mode_dropdown, self.custom_text, self.standard_dropdown, self.ri_widget.container])
        self.button_box = widgets.HBox([self.draw_btn, self.remove_btn])
        self.container = widgets.VBox([self.config_box, self.button_box])
        self._update_visibility()
        self.mode_dropdown.observe(lambda change: self._update_visibility(), names="value")
        self.draw_btn.on_click(self._on_draw)
        self.remove_btn.on_click(self._on_remove)
        self.added_config = None

    def _update_visibility(self):
        mode = self.mode_dropdown.value
        if mode == "Standard":
            self.custom_text.layout.display = "none"
            self.standard_dropdown.layout.display = ""
            self.ri_widget.container.layout.display = "none"
        elif mode == "Custom":
            self.custom_text.layout.display = ""
            self.standard_dropdown.layout.display = "none"
            self.ri_widget.container.layout.display = "none"
        elif mode == "RefractiveIndex":
            self.custom_text.layout.display = "none"
            self.standard_dropdown.layout.display = "none"
            self.ri_widget.container.layout.display = ""
    
    def _on_draw(self, b):
        self.added_config = self.get_config()
    
    def _on_remove(self, b):
        if self.remove_callback is not None:
            self.remove_callback(self)

    def get_config(self):
        mode = self.mode_dropdown.value
        if mode == "Standard":
            return {"type": "Standard", "material": self.standard_dropdown.value}
        elif mode == "Custom":
            return {"type": "Custom", "expression": self.custom_text.value.strip()}
        elif mode == "RefractiveIndex":
            sel = self.ri_widget.get_selection()
            if sel is None:
                return {"type": "None"}
            return {
                "type": "RefractiveIndex",
                "shelf": sel["shelf"],
                "book": sel["book"],
                "page": sel["page"],
                "data": sel["data"]
            }
        else:
            return {"type": "None"}

# ============================================================================
# Widget principal pour la configuration et le tracé interactif d'un rôle de matériau
# ============================================================================
class MaterialRoleWidget:
    def __init__(self, role_name, library, standard_list):
        self.role_name = role_name
        self.library = library
        self.standard_list = standard_list

        self.mode_dropdown = widgets.Dropdown(
            options=["None", "Custom", "Standard", "RefractiveIndex"],
            description="Mode:"
        )
        self.custom_text = widgets.Text(placeholder="Enter expression", description="Expr:")
        self.standard_dropdown = widgets.Dropdown(options=self.standard_list, description="Standard:")
        self.ri_widget = RefractiveIndexArboWidget(self.library)

        # Groupe le menu "Plot" avec le bouton "Draw" (remplace anciennement "Plott")
        self.plot_type_dropdown = widgets.Dropdown(
            options=[("ε(λ)", "epsilon"), ("n(λ)", "n"), ("k(λ)", "k"), ("n & k", "nk")],
            value="epsilon",
            description="Plot:"
        )
        self.draw_btn = widgets.Button(description="Draw", button_style="info")
        self.plot_type_and_btn = widgets.HBox([self.plot_type_dropdown, self.draw_btn])
        
        self.lam_min_slider = widgets.FloatSlider(value=200, min=0, max=100000, step=1, description="λ min (nm):")
        self.lam_max_slider = widgets.FloatSlider(value=1500, min=0, max=100000, step=1, description="λ max (nm):")
        # Suppression des observateurs pour que le tracé ne se mette à jour que lorsque l'utilisateur clique sur Draw
        self.plot_output = widgets.Output()

        # Zone de comparaison placée sous le graphique
        self.comparison_widgets = []
        self.comparison_vbox = widgets.VBox([])
        self.add_comparison_btn = widgets.Button(description="Add materials", button_style="info")
        self.add_comparison_btn.on_click(self.add_comparison)
        self.comparison_area = widgets.VBox([widgets.HTML("<b>Comparaison:</b>"), self.comparison_vbox, self.add_comparison_btn])
        self.plot_area = widgets.VBox([self.plot_output, self.comparison_area])

        self.container = widgets.VBox([
            self.mode_dropdown,
            self.custom_text,
            self.standard_dropdown,
            self.ri_widget.container,
            widgets.HTML(f"<hr><b>Trace pour {role_name}</b>"),
            self.plot_type_and_btn,
            widgets.HBox([self.lam_min_slider, self.lam_max_slider]),
            self.plot_area
        ])

        self._update_visibility()
        self.mode_dropdown.observe(lambda change: self._update_visibility(), names="value")
        # AUCUNE mise à jour automatique sur changement de valeurs des sliders ou menus
        self.draw_btn.on_click(lambda b: self.update_plot())
        self.update_plot()  # Possibilité de déclencher un premier tracé si désiré ; sinon, vous pouvez commenter cette ligne

    def add_comparison(self, b):
        new_comp = ComparisonMaterialWidget(self.standard_list, self.library, remove_callback=self.remove_comparison)
        self.comparison_widgets.append(new_comp)
        self.comparison_vbox.children = tuple(comp.container for comp in self.comparison_widgets)
        # Ne déclenche pas update_plot() automatiquement; l'utilisateur devra cliquer sur Draw

    def remove_comparison(self, comp_widget):
        self.comparison_widgets = [c for c in self.comparison_widgets if c is not comp_widget]
        self.comparison_vbox.children = tuple(c.container for c in self.comparison_widgets)
        # Pas de mise à jour automatique

    def _update_visibility(self):
        mode = self.mode_dropdown.value
        if mode == "None":
            self.custom_text.layout.display = "none"
            self.standard_dropdown.layout.display = "none"
            self.ri_widget.container.layout.display = "none"
        elif mode == "Custom":
            self.custom_text.layout.display = ""
            self.standard_dropdown.layout.display = "none"
            self.ri_widget.container.layout.display = "none"
        elif mode == "Standard":
            self.custom_text.layout.display = "none"
            self.standard_dropdown.layout.display = ""
            self.ri_widget.container.layout.display = "none"
        elif mode == "RefractiveIndex":
            self.custom_text.layout.display = "none"
            self.standard_dropdown.layout.display = "none"
            self.ri_widget.container.layout.display = ""

    def get_config(self):
        mode = self.mode_dropdown.value
        if mode == "None":
            return {"type": "None"}
        elif mode == "Custom":
            expr = self.custom_text.value.strip() or "None"
            return {"type": "Custom", "expression": expr}
        elif mode == "Standard":
            return {"type": "Standard", "material": self.standard_dropdown.value}
        elif mode == "RefractiveIndex":
            sel = self.ri_widget.get_selection()
            if sel is None:
                return {"type": "None"}
            return {"type": "RefractiveIndex",
                    "shelf": sel["shelf"],
                    "book": sel["book"],
                    "page": sel["page"],
                    "data": sel["data"]}

    def update_plot(self):
        with self.plot_output:
            clear_output(wait=True)
            num_points = 500
            default_bounds = (200, 1000)
            bounds_list = []

            config = self.get_config()
            mode = config.get("type", "None")
            if mode == "Standard":
                mat = config.get("material", "").strip()
                try:
                    bounds = get_lambda_bounds(mat, JSON_COMBINED_PATH, DATA_DIR)
                except Exception as e:
                    print(e)
                    bounds = None
                bounds_list.append(bounds if bounds else default_bounds)
            elif mode == "RefractiveIndex":
                try:
                    bounds = get_lambda_bounds_refractiveindex(config, DATA_DIR)
                except Exception as e:
                    print(e)
                    bounds = None
                bounds_list.append(bounds if bounds else default_bounds)
            elif mode == "Custom":
                bounds_list.append(default_bounds)
            elif mode == "None":
                print("Aucun matériau défini pour le tracé.")
                return

            for comp in self.comparison_widgets:
                comp_conf = comp.added_config  # Seules les configurations validées via le bouton Draw
                if comp_conf is None:
                    continue
                if comp_conf["type"] == "Standard":
                    try:
                        bnds = get_lambda_bounds(comp_conf.get("material", "").strip(), JSON_COMBINED_PATH, DATA_DIR)
                    except Exception as e:
                        print(e)
                        bnds = None
                    bounds_list.append(bnds if bnds else default_bounds)
                elif comp_conf["type"] == "Custom":
                    bounds_list.append(default_bounds)
                elif comp_conf["type"] == "RefractiveIndex":
                    try:
                        bnds = get_lambda_bounds_refractiveindex(comp_conf, DATA_DIR)
                    except Exception as e:
                        print(e)
                        bnds = None
                    bounds_list.append(bnds if bnds else default_bounds)
                else:
                    bounds_list.append(default_bounds)

            lam_min_global = min(b[0] for b in bounds_list)
            lam_max_global = max(b[1] for b in bounds_list)
            self.lam_min_slider.min = lam_min_global
            self.lam_min_slider.max = lam_max_global
            self.lam_max_slider.min = lam_min_global
            self.lam_max_slider.max = lam_max_global
            if self.lam_min_slider.value < lam_min_global:
                self.lam_min_slider.value = lam_min_global
            if self.lam_max_slider.value > lam_max_global:
                self.lam_max_slider.value = lam_max_global

            lam_min = self.lam_min_slider.value
            lam_max = self.lam_max_slider.value
            lam_range = np.linspace(lam_min, lam_max, num_points)

            if mode == "Custom":
                expr = config.get("expression", "")
                try:
                    if "lam" in expr:
                        n_func = lambda lam: np.array([float(eval(expr, {"lam": l})) for l in lam])
                    else:
                        eps_val = float(eval(expr))
                        if eps_val < 0:
                            raise ValueError("ε négatif")
                        n_val = np.sqrt(eps_val)
                        n_func = lambda lam: n_val * np.ones_like(lam)
                    k_func = lambda lam: np.zeros_like(lam)
                    eps_arr = compute_epsilon(n_func, k_func, lam_range)
                except Exception as e:
                    print(f"Erreur dans l'expression : {expr} ({e})")
                    return
            elif mode == "Standard":
                try:
                    from Material_Configuration import get_material_permittivity
                    n_arr, k_arr, eps_list = [], [], []
                    for l in lam_range:
                        perm_val = get_material_permittivity(config.get("material", "").strip(), l, JSON_COMBINED_PATH, DATA_DIR)
                        eps_list.append(perm_val)
                        sqrt_val = np.sqrt(perm_val)
                        n_arr.append(np.real(sqrt_val))
                        k_arr.append(np.imag(sqrt_val))
                    n_arr = np.array(n_arr)
                    k_arr = np.array(k_arr)
                    eps_arr = np.array(eps_list)
                except Exception as e:
                    print(f"Erreur dans get_material_permittivity: {e}")
                    return
            elif mode == "RefractiveIndex":
                try:
                    from Material_Configuration import build_material_configuration_dynamic
                    n_arr, k_arr, eps_list = [], [], []
                    for l in lam_range:
                        df = pd.DataFrame([{"key": self.role_name, "material": self.get_config()}])
                        eps_val = build_material_configuration_dynamic(df, l, JSON_COMBINED_PATH, None)[self.role_name]
                        eps_list.append(eps_val)
                        sqrt_val = np.sqrt(eps_val)
                        n_arr.append(np.real(sqrt_val))
                        k_arr.append(np.imag(sqrt_val))
                    n_arr = np.array(n_arr)
                    k_arr = np.array(k_arr)
                    eps_arr = np.array(eps_list)
                except Exception as e:
                    print(f"Erreur dans build_material_configuration_dynamic: {e}")
                    return
            else:
                print("Mode de configuration inconnu.")
                return

            plot_type = self.plot_type_dropdown.value
            fig, ax = plt.subplots(figsize=(8, 4))
            if mode in ["Standard", "RefractiveIndex"]:
                if plot_type == "epsilon":
                    ax.plot(lam_range, np.real(eps_arr), label=f"{self.role_name}")
                    ax.plot(lam_range, np.imag(eps_arr), label=f"{self.role_name} (Im)", linestyle="--")
                    ax.set_ylabel("ε")
                elif plot_type == "n":
                    ax.plot(lam_range, n_arr, label=f"{self.role_name}")
                    ax.set_ylabel("n")
                elif plot_type == "k":
                    ax.plot(lam_range, k_arr, label=f"{self.role_name}")
                    ax.set_ylabel("k")
                elif plot_type == "nk":
                    ax.plot(lam_range, n_arr, label=f"{self.role_name}")
                    ax.plot(lam_range, k_arr, label=f"{self.role_name} (k)", linestyle="--")
                    ax.set_ylabel("n et k")
            else:
                eps = compute_epsilon(n_func, k_func, lam_range)
                if plot_type == "epsilon":
                    ax.plot(lam_range, np.real(eps), label=f"{self.role_name}")
                    ax.plot(lam_range, np.imag(eps), label=f"{self.role_name} (Im)", linestyle="--")
                    ax.set_ylabel("ε")
                elif plot_type == "n":
                    ax.plot(lam_range, n_func(lam_range), label=f"{self.role_name}")
                    ax.set_ylabel("n")
                elif plot_type == "k":
                    ax.plot(lam_range, k_func(lam_range), label=f"{self.role_name}")
                    ax.set_ylabel("k")
                elif plot_type == "nk":
                    ax.plot(lam_range, n_func(lam_range), label=f"{self.role_name}")
                    ax.plot(lam_range, k_func(lam_range), label=f"{self.role_name} (k)", linestyle="--")
                    ax.set_ylabel("n et k")
            for comp in self.comparison_widgets:
                if comp.added_config is None:
                    continue
                comp_conf = comp.added_config
                if comp_conf["type"] == "Standard":
                    try:
                        from Material_Configuration import get_material_permittivity
                        mat_comp = comp_conf.get("material", "").strip()
                        n_comp, k_comp = [], []
                        for l in lam_range:
                            perm_comp = get_material_permittivity(mat_comp, l, JSON_COMBINED_PATH, DATA_DIR)
                            sqrt_comp = np.sqrt(perm_comp)
                            n_comp.append(np.real(sqrt_comp))
                            k_comp.append(np.imag(sqrt_comp))
                        n_comp = np.array(n_comp)
                        k_comp = np.array(k_comp)
                        if plot_type == "epsilon":
                            eps_comp = (n_comp + 1j*k_comp)**2
                            ax.plot(lam_range, np.real(eps_comp), label=f"{mat_comp} (Re)")
                            ax.plot(lam_range, np.imag(eps_comp), label=f"{mat_comp} (Im)", linestyle="--")
                        elif plot_type == "n":
                            ax.plot(lam_range, n_comp, label=mat_comp)
                        elif plot_type == "k":
                            ax.plot(lam_range, k_comp, label=mat_comp)
                        elif plot_type == "nk":
                            ax.plot(lam_range, n_comp, label=mat_comp)
                            ax.plot(lam_range, k_comp, label=f"{mat_comp} (k)", linestyle="--")
                    except Exception as e:
                        print(f"Erreur pour le matériau comparatif (Standard) {mat_comp}: {e}")
                elif comp_conf["type"] == "Custom":
                    try:
                        expr = comp_conf.get("expression", "")
                        if "lam" in expr:
                            n_comp = np.array([float(eval(expr, {"lam": l})) for l in lam_range])
                        else:
                            eps_comp_val = float(eval(expr))
                            n_val_comp = np.sqrt(eps_comp_val)
                            n_comp = n_val_comp * np.ones_like(lam_range)
                        k_comp = np.zeros_like(lam_range)
                        if plot_type == "epsilon":
                            eps_comp = (n_comp + 1j*k_comp)**2
                            ax.plot(lam_range, np.real(eps_comp), label="Custom (Re)")
                            ax.plot(lam_range, np.imag(eps_comp), label="Custom (Im)", linestyle="--")
                        elif plot_type == "n":
                            ax.plot(lam_range, n_comp, label="Custom")
                        elif plot_type == "k":
                            ax.plot(lam_range, k_comp, label="Custom")
                        elif plot_type == "nk":
                            ax.plot(lam_range, n_comp, label="Custom")
                            ax.plot(lam_range, k_comp, label="Custom (k)", linestyle="--")
                    except Exception as e:
                        print(f"Erreur pour le matériau comparatif (Custom): {e}")
                elif comp_conf["type"] == "RefractiveIndex":
                    try:
                        from Material_Configuration import build_material_configuration_dynamic
                        n_comp, k_comp = [], []
                        for l in lam_range:
                            df = pd.DataFrame([{"key": "comp", "material": comp_conf}])
                            eps_val = build_material_configuration_dynamic(df, l, JSON_COMBINED_PATH, None)["comp"]
                            sqrt_val = np.sqrt(eps_val)
                            n_comp.append(np.real(sqrt_val))
                            k_comp.append(np.imag(sqrt_val))
                        n_comp = np.array(n_comp)
                        k_comp = np.array(k_comp)
                        if plot_type == "epsilon":
                            eps_comp = (n_comp + 1j*k_comp)**2
                            ax.plot(lam_range, np.real(eps_comp), label="RI (Re)")
                            ax.plot(lam_range, np.imag(eps_comp), label="RI (Im)", linestyle="--")
                        elif plot_type == "n":
                            ax.plot(lam_range, n_comp, label="RI")
                        elif plot_type == "k":
                            ax.plot(lam_range, k_comp, label="RI")
                        elif plot_type == "nk":
                            ax.plot(lam_range, n_comp, label="RI")
                            ax.plot(lam_range, k_comp, label="RI (k)", linestyle="--")
                    except Exception as e:
                        print(f"Erreur pour le matériau comparatif (RefractiveIndex): {e}")
            ax.set_xlabel("λ (nm)")
            ax.set_title(f"{self.role_name} : {mode} - {plot_type}")
            ax.legend()
            ax.grid(True)
            ax.set_xlim(lam_min_global, lam_max_global)
            fig.tight_layout()
            display(fig)
            plt.close(fig)

# ============================================================================
# Widget principal regroupant les rôles en onglets
# ============================================================================
class MaterialSelectorTabbedNotebook:
    def __init__(self, roles):
        self.CONFIGURATIONS_dir = CONFIGURATIONS_DIR
        self.library = load_catalog_full(CATALOG_PATH)
        self.standard_list = get_standard_materials(JSON_COMBINED_PATH, DATA_DIR)
        self.roles = roles
        self.output = widgets.Output()
        self.role_widgets = {}
        children = []
        for role in roles:
            widget_role = MaterialRoleWidget(role, self.library, self.standard_list)
            self.role_widgets[role] = widget_role
            children.append(widget_role.container)
        self.tab = widgets.Tab(children=children)
        for i, role in enumerate(roles):
            self.tab.set_title(i, role)
        self.preconfigs = {
            "Preconfig Structure 1": {
                "perm_env": {"type": "None"},
                "perm_gap": {"type": "Custom", "expression": "1.45**2"},
                "perm_diel": {"type": "Custom", "expression": "1.45**2"},
                "perm_func": {"type": "None"},
                "perm_mol": {"type": "None"},
                "perm_sub": {"type": "Standard", "material": "ITO"},
                "perm_reso": {"type": "Standard", "material": "Silver"},
                "perm_metalliclayer": {"type": "Standard", "material": "Gold"},
                "perm_XIAOYI": {"type": "Standard", "material": "Si"},
                "perm_accroche": {"type": "None"}
            },
            "Preconfig Structure 2": {
                "perm_env": {"type": "None"},
                "perm_gap": {"type": "Custom", "expression": "1.45**2"},
                "perm_diel": {"type": "Custom", "expression": "1.45**2"},
                "perm_func": {"type": "None"},
                "perm_mol": {"type": "None"},
                "perm_reso": {"type": "Standard", "material": "Silver"},
                "perm_metalliclayer": {"type": "Standard", "material": "Gold"},
                "perm_XIAOYI": {"type": "Standard", "material": "Si"},
                "perm_accroche": {"type": "Standard", "material": "Aluminium"},
                "perm_sub": {"type": "Custom", "expression": "1.50**2"}
            }
        }
        self.load_preconfigs()
        self.preconfig_dropdown = widgets.Dropdown(options=self._get_preconfig_options(), description="Preconfig:")
        self.preconfig_name_text = widgets.Text(description="Preconfig Name:", placeholder="Enter a name...")
        self.add_preconfig_btn = widgets.Button(description="Add Preconfig", button_style="info")
        self.update_preconfig_btn = widgets.Button(description="Update Preconfig")
        self.delete_preconfig_btn = widgets.Button(description="Delete Preconfig", button_style="danger")
        self.preconfig_dropdown.observe(self.on_preconfig_change, names="value")
        self.add_preconfig_btn.on_click(self.on_add_preconfig)
        self.update_preconfig_btn.on_click(self.on_update_preconfig)
        self.delete_preconfig_btn.on_click(self.on_delete_preconfig)
        self.preconfig_control_box = widgets.HBox([
            self.preconfig_dropdown, self.preconfig_name_text,
            self.add_preconfig_btn, self.update_preconfig_btn, self.delete_preconfig_btn
        ])
        self.config_name_text = widgets.Text(description="Configuration Name:", placeholder="Enter the config name")
        self.add_config_btn = widgets.Button(description="Add Material config")
        self.save_quit_btn = widgets.Button(description="Save & Quit", button_style="success")
        self.add_config_btn.on_click(self.on_add_config)
        self.save_quit_btn.on_click(self.on_save_quit)
        self.config_dropdown = widgets.Dropdown(options=[], description="Saved Configs:", style={"description_width": "initial"})
        self.load_config_btn = widgets.Button(description="Load Config")
        self.update_config_btn = widgets.Button(description="Update Config")
        self.delete_config_btn = widgets.Button(description="Delete Config", button_style="danger")
        self.load_config_btn.on_click(self.on_load_config)
        self.update_config_btn.on_click(self.on_update_config)
        self.delete_config_btn.on_click(self.on_delete_config)
        self.container = widgets.VBox([
            self.preconfig_control_box, self.tab, self.config_name_text,
            widgets.HBox([self.add_config_btn, self.save_quit_btn]),
            widgets.HBox([self.config_dropdown, self.load_config_btn, self.update_config_btn, self.delete_config_btn]),
            self.output
        ])
        self.all_configs = []
        self.load_saved_configs()

    def _get_preconfig_options(self):
        options = [("None", "")]
        for key in self.preconfigs:
            options.append((key, key))
        return options

    def on_preconfig_change(self, change):
        preconfig_id = change["new"]
        if preconfig_id == "":
            for role, widget_role in self.role_widgets.items():
                widget_role.mode_dropdown.value = "None"
                widget_role.custom_text.value = ""
        else:
            self.apply_preconfig(preconfig_id)

    def apply_preconfig(self, preconfig_id):
        mapping = self.preconfigs.get(preconfig_id, {})
        for role, widget_role in self.role_widgets.items():
            if role in mapping:
                config = mapping[role]
                widget_role.mode_dropdown.value = config["type"]
                if config["type"] == "Custom":
                    widget_role.custom_text.value = config.get("expression", "")
                elif config["type"] == "Standard":
                    widget_role.standard_dropdown.value = config.get("material", "")
                elif config["type"] == "RefractiveIndex":
                    if hasattr(widget_role.ri_widget, "set_selection"):
                        selection = {
                            "shelf": config.get("shelf", ""),
                            "book": config.get("book", ""),
                            "page": config.get("page", ""),
                            "data": config.get("data", "")
                        }
                        widget_role.ri_widget.set_selection(selection)
            else:
                widget_role.mode_dropdown.value = "None"

    def on_add_preconfig(self, b):
        name = self.preconfig_name_text.value.strip()
        if not name:
            with self.output:
                print("Please enter a name for the preconfiguration.")
            return
        new_preconfig = {}
        for role, widget_role in self.role_widgets.items():
            new_preconfig[role] = widget_role.get_config()
        self.preconfigs[name] = new_preconfig
        self.preconfig_dropdown.options = self._get_preconfig_options()
        self.preconfig_dropdown.value = ""
        self.save_preconfigs()
        with self.output:
            print(f"Preconfiguration '{name}' added.")
        self.preconfig_name_text.value = ""

    def on_update_preconfig(self, b):
        selected = self.preconfig_dropdown.value
        if not selected:
            with self.output:
                print("No preconfiguration selected for update.")
            return
        updated_preconfig = {}
        for role, widget_role in self.role_widgets.items():
            updated_preconfig[role] = widget_role.get_config()
        self.preconfigs[selected] = updated_preconfig
        self.preconfig_dropdown.options = self._get_preconfig_options()
        self.preconfig_dropdown.value = ""
        self.save_preconfigs()
        with self.output:
            print(f"Preconfiguration '{selected}' updated.")

    def on_delete_preconfig(self, b):
        selected = self.preconfig_dropdown.value
        if not selected:
            with self.output:
                print("No preconfiguration selected for deletion.")
            return
        if selected in self.preconfigs:
            del self.preconfigs[selected]
            self.preconfig_dropdown.options = self._get_preconfig_options()
            self.preconfig_dropdown.value = ""
            self.save_preconfigs()
            with self.output:
                print(f"Preconfiguration '{selected}' deleted.")
        else:
            with self.output:
                print("Selected preconfiguration does not exist.")

    def load_preconfigs(self):
        preconfig_file = os.path.join(self.CONFIGURATIONS_dir, "preconfigs.json")
        if os.path.isfile(preconfig_file):
            try:
                with open(preconfig_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                self.preconfigs = loaded.get("PRECONFIGS", self.preconfigs)
                with self.output:
                    print(f"Preconfigurations loaded from {preconfig_file}")
            except Exception as e:
                with self.output:
                    print(f"Error loading preconfigurations from {preconfig_file}: {e}")
        else:
            with self.output:
                print("No preconfiguration file found; using default values.")

    def save_preconfigs(self):
        preconfig_file = os.path.join(self.CONFIGURATIONS_dir, "preconfigs.json")
        try:
            with open(preconfig_file, "w", encoding="utf-8") as f:
                json.dump({"PRECONFIGS": self.preconfigs}, f, indent=2)
            with self.output:
                print(f"Preconfigurations saved in:\n{preconfig_file}")
        except Exception as e:
            with self.output:
                print(f"Error saving preconfigurations in {preconfig_file}: {e}")

    def load_saved_configs(self):
        config_file = os.path.join(CONFIGURATIONS_DIR, "material_config.json")
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                final_dict = json.load(f)
            self.all_configs = final_dict.get("ALL_CONFIGS", [])
            self.update_config_dropdown()
            with self.output:
                print(f"Configurations loaded from {config_file}")
        except Exception as e:
            with self.output:
                print(f"Unable to load configurations from {config_file}: {e}")

    def update_config_dropdown(self):
        options = [(cfg["config_name"], cfg) for cfg in self.all_configs]
        self.config_dropdown.options = options if options else [("None", None)]

    def on_add_config(self, b):
        config_list = []
        ri_overrides = {}
        for role, widget_role in self.role_widgets.items():
            mat_info = widget_role.get_config()
            config_list.append({"key": role, "material": mat_info})
            if mat_info.get("type") == "RefractiveIndex":
                ri_overrides[role] = mat_info
        config_name = self.config_name_text.value.strip()
        if not config_name:
            config_name = f"Config_{len(self.all_configs)+1}"
        config_dict = {"config_name": config_name, "MATERIALS_CONFIG": config_list, "RI_OVERRIDES": ri_overrides}
        self.all_configs.append(config_dict)
        self.update_config_dropdown()
        with self.output:
            print(f"Configuration '{config_name}' added. You can add more or click Save & Quit.")
        self.config_name_text.value = ""
        for widget_role in self.role_widgets.values():
            widget_role.mode_dropdown.value = "None"
            widget_role.custom_text.value = ""

    def on_save_quit(self, b):
        if not self.all_configs:
            with self.output:
                print("No configuration has been added.")
            return
        final_dict = {"ALL_CONFIGS": self.all_configs}
        if not os.path.exists(CONFIGURATIONS_DIR):
            os.makedirs(CONFIGURATIONS_DIR)
        config_file = os.path.join(CONFIGURATIONS_DIR, "material_config.json")
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(final_dict, f, indent=2)
        with self.output:
            print(f"All configurations saved in:\n{config_file}")

    def on_load_config(self, b):
        selected = self.config_dropdown.value
        if selected is None:
            with self.output:
                print("No configuration selected for loading.")
            return
        for entry in selected["MATERIALS_CONFIG"]:
            role = entry["key"]
            mat_config = entry["material"]
            if role in self.role_widgets:
                widget_role = self.role_widgets[role]
                widget_role.mode_dropdown.value = mat_config.get("type", "None")
                if mat_config.get("type", "").lower() == "custom":
                    widget_role.custom_text.value = mat_config.get("expression", "")
                elif mat_config.get("type", "").lower() == "standard":
                    widget_role.standard_dropdown.value = mat_config.get("material", "")
                elif mat_config.get("type", "").lower() == "refractiveindex":
                    if hasattr(widget_role.ri_widget, "set_selection"):
                        sel = {
                            "shelf": mat_config.get("shelf", ""),
                            "book": mat_config.get("book", ""),
                            "page": mat_config.get("page", ""),
                            "data": mat_config.get("data", "")
                        }
                        widget_role.ri_widget.set_selection(sel)
        self.config_name_text.value = selected["config_name"]
        with self.output:
            print(f"Configuration '{selected['config_name']}' loaded into the tabs.")

    def on_update_config(self, b):
        selected = self.config_dropdown.value
        if selected is None:
            with self.output:
                print("No configuration selected for update.")
            return
        updated_config_list = []
        ri_overrides = {}
        for role, widget_role in self.role_widgets.items():
            mat_info = widget_role.get_config()
            updated_config_list.append({"key": role, "material": mat_info})
            if mat_info.get("type") == "RefractiveIndex":
                ri_overrides[role] = mat_info
        for cfg in self.all_configs:
            if cfg["config_name"] == selected["config_name"]:
                cfg["MATERIALS_CONFIG"] = updated_config_list
                cfg["RI_OVERRIDES"] = ri_overrides
                break
        self.update_config_dropdown()
        with self.output:
            print(f"Configuration '{selected['config_name']}' updated.")

    def on_delete_config(self, b):
        selected = self.config_dropdown.value
        if selected is None:
            with self.output:
                print("No configuration selected for deletion.")
            return
        self.all_configs = [cfg for cfg in self.all_configs if cfg["config_name"] != selected["config_name"]]
        self.update_config_dropdown()
        with self.output:
            print(f"Configuration '{selected['config_name']}' deleted.")

    def display(self):
        display(self.container)

# ============================================================================
# Rôles par défaut et lancement du Material Selector
# ============================================================================
DEFAULT_ROLES = [
    "perm_env", "perm_gap", "perm_diel", "perm_func", "perm_mol",
    "perm_reso", "perm_metalliclayer", "perm_XIAOYI", "perm_accroche", "perm_sub"
]

selector = MaterialSelectorTabbedNotebook(DEFAULT_ROLES)
