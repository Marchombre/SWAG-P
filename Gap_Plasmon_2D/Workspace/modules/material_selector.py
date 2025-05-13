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
    """Convertit les balises HTML <sub>...</sub> en indices Unicode."""
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
    """Charge le fichier YAML du catalogue complet."""
    with open(catalog_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_combined_materials(json_combined_path):
    """Charge le fichier JSON contenant la configuration combinée des matériaux."""
    with open(json_combined_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_standard_materials(json_combined_path, data_directory):
    """Retourne la liste triée des matériaux standards."""
    found = set()
    if os.path.isfile(json_combined_path):
        try:
            materials_data = load_combined_materials(json_combined_path)
            found.update(materials_data.keys())
        except Exception as e:
            print(f"[WARNING] {e}")
    if os.path.isdir(data_directory):
        for root, dirs, files in os.walk(data_directory):
            for fn in files:
                if fn.lower().endswith(".txt"):
                    name = os.path.splitext(fn)[0]
                    found.add(name)
    return sorted(found)

def get_lambda_range_from_txt(material_name, data_dir):
    """
    Recherche un fichier texte pour material_name dans data_dir et retourne
    (lambda_min, lambda_max) en nm (les valeurs du fichier étant en µm).
    """
    pattern = os.path.join(data_dir, f"{material_name}.txt")
    txt_files = glob.glob(pattern)
    if not txt_files:
        pattern = os.path.join(data_dir, f"*{material_name}*.txt")
        txt_files = glob.glob(pattern)
        if not txt_files:
            raise ValueError(f"Fichier pour '{material_name}' introuvable.")
    txt_file = txt_files[0]
    with open(txt_file, "r") as f:
        lines = f.readlines()
    wl_data = []
    for idx in range(2, len(lines) - 2):
        line = lines[idx].strip()
        if line:
            try:
                vals = [float(v) for v in line.split()]
                if len(vals) >= 3:
                    wl_data.append(vals[0])
            except ValueError:
                continue
    if not wl_data:
        raise ValueError(f"Aucune donnée de λ pour '{material_name}'.")
    wl_data = np.array(wl_data)
    return float(np.min(wl_data)*1000), float(np.max(wl_data)*1000)

def get_lambda_bounds(material_name, json_combined_path, data_dir):
    """Retourne la plage (lambda_min, lambda_max) en nm pour material_name."""
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
            if "wavelength_range" in material and isinstance(material["wavelength_range"], list):
                return float(material["wavelength_range"][0]), float(material["wavelength_range"][1])
            elif "wavelength_list" in material and isinstance(material["wavelength_list"], list) and material["wavelength_list"]:
                wl = material["wavelength_list"]
                return float(min(wl)), float(max(wl))
            else:
                return get_lambda_range_from_txt(found, data_dir)
        elif model == "brendelbormann":
            if "wavelength_range" in material and isinstance(material["wavelength_range"], list):
                return float(material["wavelength_range"][0]), float(material["wavelength_range"][1])
            else:
                return None
        else:
            raise ValueError(f"Modèle non supporté pour '{found}'.")
    else:
        return get_lambda_range_from_txt(material_name, data_dir)

def get_lambda_bounds_refractiveindex(config_identifier, data_dir):
    """Retourne la plage (lambda_min, lambda_max) en nm pour un matériau RefractiveIndex."""
    from refractiveindexINFO import RefractiveIndex, Material
    if not isinstance(config_identifier, dict):
        raise ValueError("Config dict attendue pour RefractiveIndex.")
    shelf = config_identifier.get("shelf", "").strip()
    book  = config_identifier.get("book", "").strip()
    page  = config_identifier.get("page", "").strip()
    data_field = config_identifier.get("data", "").strip()
    if data_field:
        filename = os.path.join(data_dir, data_field)
        if not os.path.exists(filename):
            raise ValueError(f"'{filename}' introuvable.")
    else:
        RI_instance = RefractiveIndex()
        filename = RI_instance.getMaterialFilename(shelf, book, page)
        if not filename:
            raise ValueError("Fichier introuvable pour RefractiveIndex.")
    mat_instance = Material(filename)
    if hasattr(mat_instance, "originalData") and isinstance(mat_instance.originalData, dict):
        if "wavelength (um)" in mat_instance.originalData:
            wavelengths = mat_instance.originalData["wavelength (um)"]
            if len(wavelengths) == 0:
                raise ValueError("Aucune donnée de λ dans le matériau.")
            return float(np.min(wavelengths)*1000), float(np.max(wavelengths)*1000)
    if hasattr(mat_instance, "getWavelengthBounds"):
        bounds = mat_instance.getWavelengthBounds()
        if isinstance(bounds, (tuple, list)) and len(bounds) == 2:
            return float(bounds[0]), float(bounds[1])
    raise ValueError("Plage de λ introuvable pour RefractiveIndex.")

def compute_epsilon(n_func, k_func, lam_range):
    """Calcule ε(λ) à partir de n(λ) et k(λ)."""
    return (n_func(lam_range) + 1j * k_func(lam_range))**2



def resolve_lambda_bounds(get_bounds_fn, config, JSON_COMBINED_PATH, DATA_DIR,
                          default_bounds=(200.0, 1000.0)):
    """
    Appelle get_bounds_fn(config, JSON_COMBINED_PATH, DATA_DIR)
    qui renvoie soit (min, max) soit None,
    gère l’override dans config['override'] et fournit toujours
    un tuple (min, max) valide.
    """
    try:
        raw = get_bounds_fn(config, JSON_COMBINED_PATH, DATA_DIR)
    except Exception:
        raw = None

    ov_min, ov_max = config.get("override", (None, None))

    # pas de bornes intrinsèques ?
    if raw is None or raw[0] is None or raw[1] is None:
        if ov_min is not None and ov_max is not None:
            return ov_min, ov_max
        return default_bounds

    # bornes intrinsèques -> on applique override et clamp
    low  = ov_min if ov_min is not None else raw[0]
    high = ov_max if ov_max is not None else raw[1]
    return max(raw[0], low), min(raw[1], high)





# ============================================================================
# Widget d'exploration du catalogue RefractiveIndex
# ============================================================================

class RefractiveIndexArboWidget:
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
                options.append((html_sub_to_unicode(entry.get("name", entry["SHELF"])), i))
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
        options = []
        for j, bk in enumerate(shelf_item.get("content", [])):
            if "BOOK" in bk:
                options.append((html_sub_to_unicode(bk.get("name", bk["BOOK"])), j))
            elif "DIVIDER" in bk:
                options.append((f"—— {bk['DIVIDER']} ——", None))
        self.book_dropdown.options = options
        self.page_dropdown.options = []
    
    def on_book_changed(self, change):
        book_val = change["new"]
        shelf_val = self.shelf_dropdown.value
        if shelf_val is None or book_val is None:
            self.page_dropdown.options = []
            return
        shelf_item = self.library[shelf_val]
        content = shelf_item.get("content", [])
        if not (0 <= book_val < len(content)):
            self.page_dropdown.options = []
            return
        book_dict = content[book_val]
        options = []
        for pg in book_dict.get("content", []):
            if "PAGE" in pg:
                options.append((html_sub_to_unicode(pg.get("name", pg["PAGE"])), pg["PAGE"]))
            elif "DIVIDER" in pg:
                options.append((f"—— {pg['DIVIDER']} ——", None))
        self.page_dropdown.options = options
        if options:
            for opt in options:
                if opt[1] is not None:
                    self.page_dropdown.value = opt[1]
                    break
    
    def get_selection(self):
        shelf_val = self.shelf_dropdown.value
        if shelf_val is None:
            return None
        shelf_item = self.library[shelf_val]
        book_val = self.book_dropdown.value
        if book_val is None:
            return None
        content = shelf_item.get("content", [])
        if not (0 <= book_val < len(content)):
            return None
        book_dict = content[book_val]
        if "BOOK" not in book_dict:
            return None
        page_val = self.page_dropdown.value
        if page_val is None:
            return None
        return {
            "shelf": shelf_item["SHELF"],
            "book": book_dict["BOOK"],
            "page": page_val,
            "data": next((pg.get("data", "") for pg in book_dict.get("content", []) if "PAGE" in pg and pg["PAGE"] == page_val), "")
        }
    
    def set_selection(self, selection):
        """
        Attend un dict {shelf, book, page, data} et positionne
        shelf_dropdown, book_dropdown et page_dropdown.
        """
        # 1) shelf
        shelf_index = next(
            (i for i, entry in enumerate(self.library)
             if entry.get("SHELF","") == selection.get("shelf","")),
            None
        )
        if shelf_index is not None:
            self.shelf_dropdown.value = shelf_index

        # 2) book (index dans le content de la shelf)
        if shelf_index is not None and selection.get("book","") != "":
            shelf_item = self.library[shelf_index].get("content", [])
            book_index = next(
                (j for j, bk in enumerate(shelf_item)
                 if bk.get("BOOK","") == selection.get("book","")),
                None
            )
            if book_index is not None:
                self.book_dropdown.value = book_index

        # 3) page (valeur égale au code PAGE)
        if selection.get("page", "") != "":
            self.page_dropdown.value = selection.get("page")

# ============================================================================
# Widget pour configurer un matériau comparé (avec override dynamique)
# ============================================================================
class ComparisonMaterialWidget:
    def __init__(self, standard_list, library, remove_callback=None):
        self.standard_list = standard_list
        self.library = library
        self.remove_callback = remove_callback

        self.mode_dropdown = widgets.Dropdown(
            options=["Standard", "Custom", "RefractiveIndex"],
            value="Standard", description="Mode:"
        )
        self.custom_text = widgets.Text(placeholder="Enter expression", description="Expr:")
        self.standard_dropdown = widgets.Dropdown(options=self.standard_list, description="Standard:")
        self.ri_widget = RefractiveIndexArboWidget(self.library)
        self.draw_btn = widgets.Button(description="Add")
        self.remove_btn = widgets.Button(description="Delete materials", button_style="danger")
        self.config_box = widgets.HBox([self.mode_dropdown, self.custom_text, self.standard_dropdown, self.ri_widget.container])
        
        # Pour Standard/Custom, override via Text (affiche valeurs si disponibles, sinon vide)
        self.standard_override_min = widgets.Text(value="", description="λ min override:")
        self.standard_override_max = widgets.Text(value="", description="λ max override:")
        self.standard_override_box = widgets.HBox([self.standard_override_min, self.standard_override_max])
        
        # Pour RefractiveIndex, override via FloatSliders
        self.refrac_override_min = widgets.FloatSlider(value=200, min=0, max=100000, step=1, description="λ min:")
        self.refrac_override_max = widgets.FloatSlider(value=1000, min=0, max=100000, step=1, description="λ max:")
        self.refrac_override_box = widgets.HBox([self.refrac_override_min, self.refrac_override_max])
        
        self.override_box = widgets.VBox([])
        self.button_box = widgets.HBox([self.draw_btn, self.remove_btn])
        self.container = widgets.VBox([self.config_box, self.override_box, self.button_box])
        self.added_config = None

        self.mode_dropdown.observe(lambda change: self._update_visibility(), names="value")
        self.standard_dropdown.observe(lambda change: self._update_override_standard(), names="value")
        self.ri_widget.shelf_dropdown.observe(lambda change: self._update_override_refrac(), names="value")
        self.ri_widget.book_dropdown.observe(lambda change: self._update_override_refrac(), names="value")
        self.ri_widget.page_dropdown.observe(lambda change: self._update_override_refrac(), names="value")
        self.draw_btn.on_click(self._on_draw)
        self.remove_btn.on_click(self._on_remove)
        self._update_visibility()
    
    def _update_override_standard(self):
        try:
            bounds = get_lambda_bounds(self.standard_dropdown.value, JSON_COMBINED_PATH, DATA_DIR)
            if bounds is not None:
                self.standard_override_min.value = str(bounds[0])
                self.standard_override_max.value = str(bounds[1])
            else:
                self.standard_override_min.value = ""
                self.standard_override_max.value = ""
        except Exception:
            self.standard_override_min.value = ""
            self.standard_override_max.value = ""
    
    def _update_override_refrac(self):
        try:
            config = self.get_config()
            bounds = get_lambda_bounds_refractiveindex(config, DATA_DIR)
            self.refrac_override_min.min = bounds[0]
            self.refrac_override_min.max = bounds[1]
            self.refrac_override_max.min = bounds[0]
            self.refrac_override_max.max = bounds[1]
            self.refrac_override_min.value = bounds[0]
            self.refrac_override_max.value = bounds[1]
        except Exception:
            self.refrac_override_min.value = 200
            self.refrac_override_max.value = 1000
    
    def _update_visibility(self):
        mode = self.mode_dropdown.value
        if mode == "Standard":
            self.custom_text.layout.display = "none"
            self.standard_dropdown.layout.display = ""
            self.ri_widget.container.layout.display = "none"
            self._update_override_standard()
            self.override_box.children = (self.standard_override_box,)
        elif mode == "Custom":
            self.custom_text.layout.display = ""
            self.standard_dropdown.layout.display = "none"
            self.ri_widget.container.layout.display = "none"
            self._update_override_standard()
            self.override_box.children = (self.standard_override_box,)
        elif mode == "RefractiveIndex":
            self.custom_text.layout.display = "none"
            self.standard_dropdown.layout.display = "none"
            self.ri_widget.container.layout.display = ""
            self._update_override_refrac()
            self.override_box.children = (self.refrac_override_box,)
    
    def _on_draw(self, b):
        self.added_config = self.get_config()
    
    def _on_remove(self, b):
        if self.remove_callback is not None:
            self.remove_callback(self)
    
    def get_config(self):
        mode = self.mode_dropdown.value
        if mode == "Standard":
            return {"type": "Standard",
                    "material": self.standard_dropdown.value,
                    "override": (self._parse_text(self.standard_override_min.value),
                                 self._parse_text(self.standard_override_max.value))}
        elif mode == "Custom":
            return {"type": "Custom",
                    "expression": self.custom_text.value.strip(),
                    "override": (self._parse_text(self.standard_override_min.value),
                                 self._parse_text(self.standard_override_max.value))}
        elif mode == "RefractiveIndex":
            sel = self.ri_widget.get_selection()
            if sel is None:
                return {"type": "None"}
            return {"type": "RefractiveIndex",
                    "shelf": sel["shelf"],
                    "book": sel["book"],
                    "page": sel["page"],
                    "data": sel["data"],
                    "override": (self.refrac_override_min.value, self.refrac_override_max.value)}
        else:
            return {"type": "None"}
    
    def _parse_text(self, text_value):
        try:
            return float(text_value.strip()) if text_value.strip() != "" else None
        except Exception:
            return None

# ============================================================================
# Widget principal de configuration et tracé pour un matériau
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

        self.plot_type_dropdown = widgets.Dropdown(
            options=[("ε(λ)", "epsilon"), ("n(λ)", "n"), ("k(λ)", "k"), ("n & k", "nk")],
            value="epsilon", description="Plot:"
        )
        self.draw_btn = widgets.Button(description="Draw", button_style="info")
        self.plot_type_and_btn = widgets.HBox([self.plot_type_dropdown, self.draw_btn])
        
        # Aucun slider global n'est utilisé
        self.plot_output = widgets.Output()

        # Override pour le matériau principal
        self.standard_override_min = widgets.Text(value="", description="λ min override:")
        self.standard_override_max = widgets.Text(value="", description="λ max override:")
        self.standard_override_box = widgets.HBox([self.standard_override_min, self.standard_override_max])
        
        self.refrac_override_min = widgets.FloatSlider(value=200, min=0, max=100000, step=1, description="λ min:")
        self.refrac_override_max = widgets.FloatSlider(value=1000, min=0, max=100000, step=1, description="λ max:")
        self.refrac_override_box = widgets.HBox([self.refrac_override_min, self.refrac_override_max])
        
        self.override_box = widgets.VBox([])
        
        # Zone de comparaison
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
            self.override_box,
            widgets.HTML(f"<hr><b>Trace pour {role_name}</b>"),
            self.plot_type_and_btn,
            self.plot_area
        ])

        self.mode_dropdown.observe(lambda change: self._update_visibility(), names="value")
        self.standard_dropdown.observe(lambda change: self._update_override_standard(), names="value")
        self.ri_widget.shelf_dropdown.observe(lambda change: self._update_override_refrac(), names="value")
        self.ri_widget.book_dropdown.observe(lambda change: self._update_override_refrac(), names="value")
        self.ri_widget.page_dropdown.observe(lambda change: self._update_override_refrac(), names="value")
        self.draw_btn.on_click(lambda b: self.update_plot())
        self._update_visibility()
        # Ne pas appeler update_plot() automatiquement ici.
    
    def _update_override_standard(self):
        try:
            bounds = get_lambda_bounds(self.standard_dropdown.value, JSON_COMBINED_PATH, DATA_DIR)
            if bounds is not None:
                self.standard_override_min.value = str(bounds[0])
                self.standard_override_max.value = str(bounds[1])
            else:
                self.standard_override_min.value = ""
                self.standard_override_max.value = ""
        except Exception:
            self.standard_override_min.value = ""
            self.standard_override_max.value = ""
    
    def _update_override_refrac(self):
        try:
            config = self.get_config()
            bounds = get_lambda_bounds_refractiveindex(config, DATA_DIR)
            self.refrac_override_min.min = bounds[0]
            self.refrac_override_min.max = bounds[1]
            self.refrac_override_max.min = bounds[0]
            self.refrac_override_max.max = bounds[1]
            self.refrac_override_min.value = bounds[0]
            self.refrac_override_max.value = bounds[1]
        except Exception:
            self.refrac_override_min.value = 200
            self.refrac_override_max.value = 1000
    
    def _update_visibility(self):
        mode = self.mode_dropdown.value
        if mode == "None":
            self.custom_text.layout.display = "none"
            self.standard_dropdown.layout.display = "none"
            self.ri_widget.container.layout.display = "none"
            self.override_box.children = []
        elif mode == "Custom":
            self.custom_text.layout.display = ""
            self.standard_dropdown.layout.display = "none"
            self.ri_widget.container.layout.display = "none"
            self._update_override_standard()
            self.override_box.children = (self.standard_override_box,)
        elif mode == "Standard":
            self.custom_text.layout.display = "none"
            self.standard_dropdown.layout.display = ""
            self.ri_widget.container.layout.display = "none"
            self._update_override_standard()
            self.override_box.children = (self.standard_override_box,)
        elif mode == "RefractiveIndex":
            self.custom_text.layout.display = "none"
            self.standard_dropdown.layout.display = "none"
            self.ri_widget.container.layout.display = ""
            self._update_override_refrac()
            self.override_box.children = (self.refrac_override_box,)
    
    def add_comparison(self, b):
        new_comp = ComparisonMaterialWidget(self.standard_list, self.library, remove_callback=self.remove_comparison)
        # Ne validez pas automatiquement : l'utilisateur doit appuyer sur Draw pour le comparatif.
        self.comparison_widgets.append(new_comp)
        self.comparison_vbox.children = tuple(comp.container for comp in self.comparison_widgets)
    
    def remove_comparison(self, comp_widget):
        self.comparison_widgets = [c for c in self.comparison_widgets if c is not comp_widget]
        self.comparison_vbox.children = tuple(c.container for c in self.comparison_widgets)
    
    def get_config(self):
        mode = self.mode_dropdown.value
        if mode == "None":
            return {"type": "None"}
        elif mode == "Custom":
            expr = self.custom_text.value.strip() or "None"
            try:
                ov_min = float(self.standard_override_min.value.strip()) if self.standard_override_min.value.strip() != "" else None
            except Exception:
                ov_min = None
            try:
                ov_max = float(self.standard_override_max.value.strip()) if self.standard_override_max.value.strip() != "" else None
            except Exception:
                ov_max = None
            return {"type": "Custom",
                    "expression": expr,
                    "override": (ov_min, ov_max)}
        elif mode == "Standard":
            try:
                ov_min = float(self.standard_override_min.value.strip()) if self.standard_override_min.value.strip() != "" else None
            except Exception:
                ov_min = None
            try:
                ov_max = float(self.standard_override_max.value.strip()) if self.standard_override_max.value.strip() != "" else None
            except Exception:
                ov_max = None
            return {"type": "Standard",
                    "material": self.standard_dropdown.value,
                    "override": (ov_min, ov_max)}
        elif mode == "RefractiveIndex":
            sel = self.ri_widget.get_selection()
            if sel is None:
                return {"type": "None"}
            return {"type": "RefractiveIndex",
                    "shelf": sel["shelf"],
                    "book": sel["book"],
                    "page": sel["page"],
                    "data": sel["data"],
                    "override": (self.refrac_override_min.value, self.refrac_override_max.value)}
    
    
    
    def update_plot(self):
        with self.plot_output:
            clear_output(wait=True)
            num_points = 500
            config = self.get_config()
            mode = config.get("type", "None")
            if mode == "None":
                print("Aucun matériau défini pour le tracé.")
                return

            # --- calcul des bornes du matériau principal sans aucun fallback ---
            # --- calcul des bornes du matériau principal, via resolve_lambda_bounds ---
            if mode == "RefractiveIndex":
                bounds_fn = lambda cfg, json_path, data_dir: get_lambda_bounds_refractiveindex(cfg, data_dir)
            else:
                bounds_fn = lambda cfg, json_path, data_dir: get_lambda_bounds(
                    cfg.get("material",""), json_path, data_dir)

            main_bounds = resolve_lambda_bounds(bounds_fn, config, JSON_COMBINED_PATH, DATA_DIR)
            local_range = np.linspace(main_bounds[0], main_bounds[1], num_points)


            # --- calcul des valeurs du matériau principal ---
            if mode == "Custom":
                expr = config.get("expression", "")
                try:
                    if "lam" in expr:
                        n_func = lambda lam: np.array([float(eval(expr, {"lam": l})) for l in lam])
                    else:
                        eps_val = float(eval(expr))
                        n_val = np.sqrt(eps_val)
                        n_func = lambda lam: n_val * np.ones_like(lam)
                    k_func = lambda lam: np.zeros_like(lam)
                    eps_vals = compute_epsilon(n_func, k_func, local_range)
                    n_vals = n_func(local_range)
                    k_vals = k_func(local_range)
                except Exception as e:
                    print(f"Erreur dans l'expression : {expr} ({e})")
                    return

            elif mode == "Standard":
                try:
                    from Material_Configuration import get_material_permittivity
                    n_vals, k_vals, eps_list = [], [], []
                    for l in local_range:
                        perm_val = get_material_permittivity(config.get("material", "").strip(),
                                                            l, JSON_COMBINED_PATH, DATA_DIR)
                        eps_list.append(perm_val)
                        sqrt_val = np.sqrt(perm_val)
                        n_vals.append(np.real(sqrt_val))
                        k_vals.append(np.imag(sqrt_val))
                    n_vals = np.array(n_vals)
                    k_vals = np.array(k_vals)
                    eps_vals = np.array(eps_list)
                except Exception as e:
                    print(f"Erreur dans get_material_permittivity: {e}")
                    return

            elif mode == "RefractiveIndex":
                try:
                    from Material_Configuration import build_material_configuration_dynamic
                    n_vals, k_vals, eps_list = [], [], []
                    for l in local_range:
                        df = pd.DataFrame([{"key": self.role_name, "material": config}])
                        eps_val = build_material_configuration_dynamic(
                            df, l, JSON_COMBINED_PATH, None)[self.role_name]
                        eps_list.append(eps_val)
                        sqrt_val = np.sqrt(eps_val)
                        n_vals.append(np.real(sqrt_val))
                        k_vals.append(np.imag(sqrt_val))
                    n_vals = np.array(n_vals)
                    k_vals = np.array(k_vals)
                    eps_vals = np.array(eps_list)
                except Exception as e:
                    print(f"Erreur dans build_material_configuration_dynamic: {e}")
                    return

            else:
                print("Mode de configuration inconnu.")
                return

            # --- tracé du matériau principal ---
            fig, ax = plt.subplots(figsize=(8, 4))
            plot_type = self.plot_type_dropdown.value
            if plot_type == "epsilon":
                ax.plot(local_range, np.real(eps_vals), label=f"{self.role_name} (Re)")
                ax.plot(local_range, np.imag(eps_vals), label=f"{self.role_name} (Im)")
                ax.set_ylabel("ε")
            elif plot_type == "n":
                ax.plot(local_range, n_vals, label=f"{self.role_name}")
                ax.set_ylabel("n")
            elif plot_type == "k":
                ax.plot(local_range, k_vals, label=f"{self.role_name}")
                ax.set_ylabel("k")
            elif plot_type == "nk":
                ax.plot(local_range, n_vals, label=f"{self.role_name} (n)")
                ax.plot(local_range, k_vals, label=f"{self.role_name} (k)")
                ax.set_ylabel("n et k")

            # === tracé des matériaux de comparaison ===
            for comp in self.comparison_widgets:
                comp_conf = comp.get_config()
                if comp_conf.get("type", "None") == "None":
                    continue

                # bornes
                # bornes et plage comparée via resolve_lambda_bounds
                if comp_conf["type"] == "RefractiveIndex":
                    bounds_fn_c = lambda cfg, json_path, data_dir: get_lambda_bounds_refractiveindex(cfg, data_dir)
                else:
                    bounds_fn_c = lambda cfg, json_path, data_dir: get_lambda_bounds(
                        cfg.get("material",""), json_path, data_dir)

                comp_bounds = resolve_lambda_bounds(bounds_fn_c, comp_conf,
                                                    JSON_COMBINED_PATH, DATA_DIR)
                comp_range  = np.linspace(comp_bounds[0], comp_bounds[1], num_points)


                # calcul des valeurs
                if comp_conf["type"] == "Custom":
                    expr = comp_conf.get("expression", "")
                    if "lam" in expr:
                        n_comp = lambda lam: np.array([float(eval(expr, {"lam": l})) for l in lam])
                    else:
                        val = float(eval(expr))
                        n_comp = lambda lam: val * np.ones_like(lam)
                    k_comp = lambda lam: np.zeros_like(lam)
                    eps_comp = compute_epsilon(n_comp, k_comp, comp_range)
                    n_vals_c = n_comp(comp_range)
                    k_vals_c = k_comp(comp_range)

                elif comp_conf["type"] == "Standard":
                    from Material_Configuration import get_material_permittivity
                    n_vals_c, k_vals_c, eps_list_c = [], [], []
                    for l in comp_range:
                        epsv = get_material_permittivity(comp_conf.get("material", ""), l,
                                                        JSON_COMBINED_PATH, DATA_DIR)
                        eps_list_c.append(epsv)
                        sq = np.sqrt(epsv)
                        n_vals_c.append(np.real(sq))
                        k_vals_c.append(np.imag(sq))
                    n_vals_c = np.array(n_vals_c)
                    k_vals_c = np.array(k_vals_c)
                    eps_comp = np.array(eps_list_c)

                elif comp_conf["type"] == "RefractiveIndex":
                    from Material_Configuration import build_material_configuration_dynamic
                    n_vals_c, k_vals_c, eps_list_c = [], [], []
                    for l in comp_range:
                        df = pd.DataFrame([{"key": self.role_name, "material": comp_conf}])
                        epsv = build_material_configuration_dynamic(
                            df, l, JSON_COMBINED_PATH, None)[self.role_name]
                        eps_list_c.append(epsv)
                        sq = np.sqrt(epsv)
                        n_vals_c.append(np.real(sq))
                        k_vals_c.append(np.imag(sq))
                    n_vals_c = np.array(n_vals_c)
                    k_vals_c = np.array(k_vals_c)
                    eps_comp = np.array(eps_list_c)

                # préparation du label
                if comp_conf["type"] in ["Standard", "Custom"]:
                    label_base = comp_conf.get("material", comp_conf.get("expression", ""))
                else:
                    label_base = f"RI {comp_conf.get('shelf')}:{comp_conf.get('page')}"

                # tracé comparison
                if plot_type == "epsilon":
                    ax.plot(comp_range, np.real(eps_comp), label=f"{label_base} (Re)", linestyle=":")
                    ax.plot(comp_range, np.imag(eps_comp), label=f"{label_base} (Im)", linestyle=":")
                elif plot_type == "n":
                    ax.plot(comp_range, n_vals_c, label=label_base, linestyle=":")
                elif plot_type == "k":
                    ax.plot(comp_range, k_vals_c, label=label_base, linestyle=":")
                elif plot_type == "nk":
                    ax.plot(comp_range, n_vals_c, label=f"{label_base} (n)", linestyle=":")
                    ax.plot(comp_range, k_vals_c, label=f"{label_base} (k)", linestyle=":")

            # --- fin comparaison ---

            ax.set_xlabel("λ (nm)")
            ax.set_title(f"{self.role_name} : {mode} - {plot_type}")
            ax.legend()
            ax.grid(True)
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
                        sel = {
                            "shelf": config.get("shelf", ""),
                            "book": config.get("book", ""),
                            "page": config.get("page", ""),
                            "data": config.get("data", "")
                        }
                        widget_role.ri_widget.set_selection(sel)
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
                    print(f"Error loading preconfigurations: {e}")
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
                print(f"Error saving preconfigurations: {e}")
    
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
                print(f"Unable to load configurations: {e}")
    
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
        config_name = self.config_name_text.value.strip() or f"Config_{len(self.all_configs)+1}"
        config_dict = {"config_name": config_name, "MATERIALS_CONFIG": config_list, "RI_OVERRIDES": ri_overrides}
        self.all_configs.append(config_dict)
        self.update_config_dropdown()
        with self.output:
            print(f"Configuration '{config_name}' added.")
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

        # (1) Pour chaque rôle, on positionne mode + valeur
        for entry in selected["MATERIALS_CONFIG"]:
            role = entry["key"]
            mat_config = entry["material"]
            widget_role = self.role_widgets.get(role)
            if widget_role is None:
                continue

            # a) mode (None, Custom, Standard, RefractiveIndex)
            widget_role.mode_dropdown.value = mat_config.get("type", "None")

            # b) contenu
            mtype = widget_role.mode_dropdown.value.lower()
            if mtype == "custom":
                widget_role.custom_text.value = mat_config.get("expression", "")
            elif mtype == "standard":
                widget_role.standard_dropdown.value = mat_config.get("material", "")
            elif mtype == "refractiveindex":
                sel = {
                    "shelf": mat_config.get("shelf", ""),
                    "book":  mat_config.get("book", ""),
                    "page":  mat_config.get("page", ""),
                    "data":  mat_config.get("data", "")
                }
                widget_role.ri_widget.set_selection(sel)
                # on force le recalcul des sliders λ-min/λ-max
                widget_role._update_override_refrac()

        # (2) On rafraîchit l’affichage de chaque onglet
        for widget_role in self.role_widgets.values():
            widget_role._update_visibility()

        # (3) Mise à jour du nom et message
        self.config_name_text.value = selected["config_name"]
        with self.output:
            print(f"Configuration '{selected['config_name']}' loaded.")

    
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
