# material_selector.py


from gap_plasmon_2d import paths
import os
import yaml
import json
import glob
import ipywidgets as widgets
from IPython.display import display, clear_output
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functools import lru_cache


from gap_plasmon_2d.ui.geometry_settings import get_geometry_save_path
from gap_plasmon_2d.utils.file_watchers import start_watcher
from gap_plasmon_2d.ui.ui_events import subscribe_geometry_changed
from gap_plasmon_2d.ui.ui_material_events import notify_material_config_changed

# ============================================================================
# Définition des chemins globaux
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
WORKSPACE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(str(paths.DATA_DIR))
CONFIGURATIONS_DIR = os.path.join(str(paths.CONFIGS_DIR))
CATALOG_PATH = os.path.join(WORKSPACE_DIR, paths.CATALOG_NK)
JSON_COMBINED_PATH = os.path.join(DATA_DIR, "combined_materials.json")


# ============================================================================
# Rôles par défaut et lancement du Material Selector
# ============================================================================
DEFAULT_ROLES = [
    "perm_env", "perm_reso", "perm_mol", "perm_func", "perm_diel", "perm_gap",
    "perm_metalliclayer", "perm_XIAOYI", "perm_accroche", "perm_sub"
]


# ============================================================================
# Fonctions utilitaires
# ============================================================================


def get_bounds_function_for_mode(mode):
    """
    Retourne une fonction adaptée au mode matériau pour calculer les bornes λ.
    """
    if mode == "RefractiveIndex":
        return lambda cfg, json_path, data_dir: get_lambda_bounds_refractiveindex(cfg, data_dir)
    elif mode == "Standard":
        return lambda cfg, json_path, data_dir: get_lambda_bounds(cfg.get("material", ""), json_path, data_dir)
    elif mode == "Custom":
        return lambda cfg, json_path, data_dir: None
    else:
        return lambda cfg, json_path, data_dir: None




def safe_eval_custom_expression(expr, lam=None):
    """
    Évalue une expression custom dans un contexte restreint.
    Autorise np et éventuellement lam.
    """
    safe_globals = {"__builtins__": {}}
    safe_locals = {"np": np}

    if lam is not None:
        safe_locals["lam"] = lam

    return eval(expr, safe_globals, safe_locals)



def _load_all_geometry_configs():
    """
    Charge toutes les géométries sauvegardées.
    Retourne une liste de dictionnaires.
    """
    path = get_geometry_save_path()
    if not os.path.exists(path):
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_cfgs = data.get("ALL_GEOMETRY_CONFIGS", [])
        return all_cfgs if isinstance(all_cfgs, list) else []
    except Exception:
        return []



def _load_geometry_names():
    all_cfgs = _load_all_geometry_configs()
    return [cfg["config_name"] for cfg in all_cfgs if "config_name" in cfg]



def _geometry_to_display_roles(geom):
    """
    Transforme un dict de géométrie en liste ordonnée de rôles matériaux
    à afficher, uniquement pour les couches dont l'épaisseur est > 0.

    On conserve l'ordre physique voulu pour l'UI matériaux.
    """
    if not isinstance(geom, dict):
        return DEFAULT_ROLES.copy()

    display_roles = []

    # --- avant ---
    if geom.get("thick_super", 0) > 0:
        display_roles.append("perm_env")

    before_map = [
        ("thick_reso",             "perm_reso"),
        ("thick_gap",              "perm_gap"),
        ("thick_mol",              "perm_mol"),
        ("thick_func",             "perm_func"),
        ("thick_diel",             "perm_diel"),
        ("thick_metalliclayer",    "perm_metalliclayer"),
    ]
    for key, role in before_map:
        if geom.get(key, 0) > 0:
            display_roles.append(role)

    # --- couches homo dynamiques ---
    for thick_key, val in geom.items():
        if thick_key.startswith("thick_homo_") and val > 0:
            suffix = thick_key[len("thick_"):]
            display_roles.append(f"perm_{suffix}")

    # --- après ---
    after_map = [
        ("thick_XIAOYI",   "perm_XIAOYI"),
        ("thick_accroche", "perm_accroche"),
        ("thick_sub",      "perm_sub"),
    ]
    for key, role in after_map:
        if geom.get(key, 0) > 0:
            display_roles.append(role)

    return display_roles


def _load_geometry_by_name(config_name):
    """
    Charge une géométrie sauvegardée à partir de son nom.
    Retourne l'entrée complète {config_name, compartment, geometry} ou None.
    """
    all_cfgs = _load_all_geometry_configs()
    return next((cfg for cfg in all_cfgs if cfg.get("config_name") == config_name), None)


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

@lru_cache(maxsize=1)
def load_combined_materials(json_combined_path):
    """
    Charge le fichier JSON contenant la configuration combinée des matériaux.
    Le résultat est mis en cache pour éviter de relire le fichier à chaque appel.
    """
    with open(json_combined_path, "r", encoding="utf-8") as f:
        return json.load(f)

@lru_cache(maxsize=1)
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
    materials_data = load_combined_materials(json_combined_path)

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
    from gap_plasmon_2d.materials.refractiveindex_info import RefractiveIndex, Material
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



def resolve_lambda_bounds(get_bounds_fn, config, json_combined_path, data_dir,
                          default_bounds=(200.0, 1000.0)):
    """
    Appelle get_bounds_fn(config, json_combined_path, data_dir)
    qui renvoie soit (min, max) soit None,
    gère l’override dans config['override'] et fournit toujours
    un tuple (min, max) valide.
    """
    try:
        raw = get_bounds_fn(config, json_combined_path, data_dir)
    except Exception:
        raw = None

    ov_min, ov_max = config.get("override", (None, None))

    if raw is None or raw[0] is None or raw[1] is None:
        if ov_min is not None and ov_max is not None:
            return ov_min, ov_max
        return default_bounds

    low = ov_min if ov_min is not None else raw[0]
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
        self._is_applying_config = False

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
        
        # Pour Standard/Custom, override via Text
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

        self.mode_dropdown.observe(lambda change: self._update_visibility(refresh_override=not self._is_applying_config), names="value")
        self.standard_dropdown.observe(lambda change: self._update_override_standard(), names="value")
        self.ri_widget.shelf_dropdown.observe(lambda change: self._update_override_refrac(), names="value")
        self.ri_widget.book_dropdown.observe(lambda change: self._update_override_refrac(), names="value")
        self.ri_widget.page_dropdown.observe(lambda change: self._update_override_refrac(), names="value")
        self.draw_btn.on_click(self._on_draw)
        self.remove_btn.on_click(self._on_remove)
        self._update_visibility()

    def _override_pair_from_config(self, config):
        override = config.get("override", (None, None))
        if not isinstance(override, (list, tuple)) or len(override) != 2:
            return None, None
        return override[0], override[1]

    def _set_standard_override_fields(self, override_min, override_max):
        self.standard_override_min.value = "" if override_min is None else str(override_min)
        self.standard_override_max.value = "" if override_max is None else str(override_max)

    def _set_refractiveindex_override_fields(self, override_min, override_max):
        try:
            current_min_limit = self.refrac_override_min.min
            current_max_limit = self.refrac_override_max.max

            if override_min is None:
                override_min = self.refrac_override_min.value
            if override_max is None:
                override_max = self.refrac_override_max.value

            override_min = float(override_min)
            override_max = float(override_max)

            override_min = max(current_min_limit, min(current_max_limit, override_min))
            override_max = max(current_min_limit, min(current_max_limit, override_max))

            if override_max < override_min:
                override_max = override_min

            self.refrac_override_min.value = override_min
            self.refrac_override_max.value = override_max

        except Exception:
            pass

    def apply_config(self, config):
        """
        Applique une configuration complète au widget de comparaison,
        y compris les overrides sauvegardés, sans les écraser ensuite.
        """
        if not isinstance(config, dict):
            self.mode_dropdown.value = "Standard"
            self._update_visibility(refresh_override=True)
            return

        mode = config.get("type", "Standard")

        if mode not in ("Standard", "Custom", "RefractiveIndex"):
            mode = "Standard"

        self._is_applying_config = True
        try:
            self.mode_dropdown.value = mode

            if mode == "Standard":
                material_name = config.get("material", "")
                available_values = [opt[1] if isinstance(opt, tuple) else opt for opt in self.standard_dropdown.options]
                if material_name in available_values:
                    self.standard_dropdown.value = material_name

                self._update_visibility(refresh_override=False)

                override_min, override_max = self._override_pair_from_config(config)
                self._set_standard_override_fields(override_min, override_max)

            elif mode == "Custom":
                self.custom_text.value = config.get("expression", "").strip()

                self._update_visibility(refresh_override=False)

                override_min, override_max = self._override_pair_from_config(config)
                self._set_standard_override_fields(override_min, override_max)

            elif mode == "RefractiveIndex":
                sel = {
                    "shelf": config.get("shelf", ""),
                    "book": config.get("book", ""),
                    "page": config.get("page", ""),
                    "data": config.get("data", "")
                }
                self.ri_widget.set_selection(sel)

                # D'abord on met à jour les limites intrinsèques
                self._update_override_refrac()

                # Puis on affiche le bon mode sans réécraser
                self._update_visibility(refresh_override=False)

                # Enfin on applique les overrides sauvegardés
                override_min, override_max = self._override_pair_from_config(config)
                self._set_refractiveindex_override_fields(override_min, override_max)
        finally:
            self._is_applying_config = False

    def _update_override_standard(self):
        if self._is_applying_config:
            return
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
        if self._is_applying_config:
            return
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

    def _update_visibility(self, refresh_override=True):
        mode = self.mode_dropdown.value
        if mode == "Standard":
            self.custom_text.layout.display = "none"
            self.standard_dropdown.layout.display = ""
            self.ri_widget.container.layout.display = "none"
            self.override_box.children = (self.standard_override_box,)
            if refresh_override:
                self._update_override_standard()
        elif mode == "Custom":
            self.custom_text.layout.display = ""
            self.standard_dropdown.layout.display = "none"
            self.ri_widget.container.layout.display = "none"
            self.override_box.children = (self.standard_override_box,)
        elif mode == "RefractiveIndex":
            self.custom_text.layout.display = "none"
            self.standard_dropdown.layout.display = "none"
            self.ri_widget.container.layout.display = ""
            self.override_box.children = (self.refrac_override_box,)
            if refresh_override:
                self._update_override_refrac()

    def _on_draw(self, b):
        self.added_config = self.get_config()

    def _on_remove(self, b):
        if self.remove_callback is not None:
            self.remove_callback(self)

    def get_config(self):
        mode = self.mode_dropdown.value
        if mode == "Standard":
            return {
                "type": "Standard",
                "material": self.standard_dropdown.value,
                "override": (
                    self._parse_text(self.standard_override_min.value),
                    self._parse_text(self.standard_override_max.value)
                )
            }
        elif mode == "Custom":
            return {
                "type": "Custom",
                "expression": self.custom_text.value.strip(),
                "override": (
                    self._parse_text(self.standard_override_min.value),
                    self._parse_text(self.standard_override_max.value)
                )
            }
        elif mode == "RefractiveIndex":
            sel = self.ri_widget.get_selection()
            if sel is None:
                return {"type": "None"}
            return {
                "type": "RefractiveIndex",
                "shelf": sel["shelf"],
                "book": sel["book"],
                "page": sel["page"],
                "data": sel["data"],
                "override": (self.refrac_override_min.value, self.refrac_override_max.value)
            }
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
        self._is_applying_config = False

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

        self.plot_output = widgets.Output()

        self.standard_override_min = widgets.Text(value="", description="λ min override:")
        self.standard_override_max = widgets.Text(value="", description="λ max override:")
        self.standard_override_box = widgets.HBox([self.standard_override_min, self.standard_override_max])

        self.refrac_override_min = widgets.FloatSlider(value=200, min=0, max=100000, step=1, description="λ min:")
        self.refrac_override_max = widgets.FloatSlider(value=1000, min=0, max=100000, step=1, description="λ max:")
        self.refrac_override_box = widgets.HBox([self.refrac_override_min, self.refrac_override_max])

        self.override_box = widgets.VBox([])

        self.comparison_widgets = []
        self.comparison_vbox = widgets.VBox([])
        self.add_comparison_btn = widgets.Button(description="Add materials", button_style="info")
        self.add_comparison_btn.on_click(self.add_comparison)
        self.comparison_area = widgets.VBox([widgets.HTML("<b>Compare with...:</b>"), self.comparison_vbox, self.add_comparison_btn])
        self.plot_area = widgets.VBox([self.plot_output, self.comparison_area])

        self.container = widgets.VBox([
            self.mode_dropdown,
            self.custom_text,
            self.standard_dropdown,
            self.ri_widget.container,
            self.override_box,
            widgets.HTML(f"<hr><b>Plot spectrum for {role_name}</b>"),
            self.plot_type_and_btn,
            self.plot_area
        ])

        self.mode_dropdown.observe(lambda change: self._update_visibility(refresh_override=not self._is_applying_config), names="value")
        self.standard_dropdown.observe(lambda change: self._update_override_standard(), names="value")
        self.ri_widget.shelf_dropdown.observe(lambda change: self._update_override_refrac(), names="value")
        self.ri_widget.book_dropdown.observe(lambda change: self._update_override_refrac(), names="value")
        self.ri_widget.page_dropdown.observe(lambda change: self._update_override_refrac(), names="value")
        self.draw_btn.on_click(lambda b: self.update_plot())
        self._update_visibility()



    def reset_widget(self):
        """
        Remet le widget dans un état propre.
        """
        self.mode_dropdown.value = "None"
        self.custom_text.value = ""
        self.standard_override_min.value = ""
        self.standard_override_max.value = ""
        self.comparison_widgets = []
        self.comparison_vbox.children = tuple()
        self._update_visibility(refresh_override=False)


    def _override_pair_from_config(self, config):
        override = config.get("override", (None, None))
        if not isinstance(override, (list, tuple)) or len(override) != 2:
            return None, None
        return override[0], override[1]

    def _set_standard_override_fields(self, override_min, override_max):
        self.standard_override_min.value = "" if override_min is None else str(override_min)
        self.standard_override_max.value = "" if override_max is None else str(override_max)

    def _set_refractiveindex_override_fields(self, override_min, override_max):
        try:
            current_min_limit = self.refrac_override_min.min
            current_max_limit = self.refrac_override_max.max

            if override_min is None:
                override_min = self.refrac_override_min.value
            if override_max is None:
                override_max = self.refrac_override_max.value

            override_min = float(override_min)
            override_max = float(override_max)

            override_min = max(current_min_limit, min(current_max_limit, override_min))
            override_max = max(current_min_limit, min(current_max_limit, override_max))

            if override_max < override_min:
                override_max = override_min

            self.refrac_override_min.value = override_min
            self.refrac_override_max.value = override_max

        except Exception:
            pass

    def apply_config(self, config):
        """
        Applique une configuration complète au widget de rôle,
        y compris les overrides sauvegardés, sans les écraser ensuite.
        """
        if not isinstance(config, dict):
            self.mode_dropdown.value = "None"
            self._update_visibility(refresh_override=True)
            return

        mode = config.get("type", "None")

        if mode not in ("None", "Custom", "Standard", "RefractiveIndex"):
            mode = "None"

        self._is_applying_config = True
        try:
            self.mode_dropdown.value = mode

            if mode == "None":
                self.custom_text.value = ""
                self._update_visibility(refresh_override=False)
                return

            if mode == "Standard":
                material_name = config.get("material", "")
                available_values = [opt[1] if isinstance(opt, tuple) else opt for opt in self.standard_dropdown.options]
                if material_name in available_values:
                    self.standard_dropdown.value = material_name

                self._update_visibility(refresh_override=False)

                override_min, override_max = self._override_pair_from_config(config)
                self._set_standard_override_fields(override_min, override_max)

            elif mode == "Custom":
                self.custom_text.value = config.get("expression", "").strip()

                self._update_visibility(refresh_override=False)

                override_min, override_max = self._override_pair_from_config(config)
                self._set_standard_override_fields(override_min, override_max)

            elif mode == "RefractiveIndex":
                sel = {
                    "shelf": config.get("shelf", ""),
                    "book": config.get("book", ""),
                    "page": config.get("page", ""),
                    "data": config.get("data", "")
                }
                self.ri_widget.set_selection(sel)

                # D'abord on met à jour les limites intrinsèques
                self._update_override_refrac()

                # Puis on affiche le bon mode sans écrasement
                self._update_visibility(refresh_override=False)

                # Enfin on applique les overrides sauvegardés
                override_min, override_max = self._override_pair_from_config(config)
                self._set_refractiveindex_override_fields(override_min, override_max)
        finally:
            self._is_applying_config = False

    def _update_override_standard(self):
        if self._is_applying_config:
            return
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
        if self._is_applying_config:
            return
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

    def _update_visibility(self, refresh_override=True):
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
            self.override_box.children = (self.standard_override_box,)
        elif mode == "Standard":
            self.custom_text.layout.display = "none"
            self.standard_dropdown.layout.display = ""
            self.ri_widget.container.layout.display = "none"
            self.override_box.children = (self.standard_override_box,)
            if refresh_override:
                self._update_override_standard()
        elif mode == "RefractiveIndex":
            self.custom_text.layout.display = "none"
            self.standard_dropdown.layout.display = "none"
            self.ri_widget.container.layout.display = ""
            self.override_box.children = (self.refrac_override_box,)
            if refresh_override:
                self._update_override_refrac()

    def add_comparison(self, b):
        new_comp = ComparisonMaterialWidget(self.standard_list, self.library, remove_callback=self.remove_comparison)
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
            ov_min = self._parse_text(self.standard_override_min.value)
            ov_max = self._parse_text(self.standard_override_max.value)

            return {
                "type": "Custom",
                "expression": expr,
                "override": (ov_min, ov_max)
            }

        elif mode == "Standard":
            ov_min = self._parse_text(self.standard_override_min.value)
            ov_max = self._parse_text(self.standard_override_max.value)

            return {
                "type": "Standard",
                "material": self.standard_dropdown.value,
                "override": (ov_min, ov_max)
            }

        elif mode == "RefractiveIndex":
            sel = self.ri_widget.get_selection()
            if sel is None:
                return {"type": "None"}

            return {
                "type": "RefractiveIndex",
                "shelf": sel["shelf"],
                "book": sel["book"],
                "page": sel["page"],
                "data": sel["data"],
                "override": (self.refrac_override_min.value, self.refrac_override_max.value)
            }



    def _parse_text(self, text_value):
        try:
            return float(text_value.strip()) if text_value.strip() != "" else None
        except Exception:
            return None



    def update_plot(self):
        with self.plot_output:
            clear_output(wait=True)
            num_points = 500
            config = self.get_config()
            mode = config.get("type", "None")
            if mode == "None":
                print("Aucun matériau défini pour le tracé.")
                return

            bounds_fn = get_bounds_function_for_mode(mode)

            main_bounds = resolve_lambda_bounds(bounds_fn, config, JSON_COMBINED_PATH, DATA_DIR)
            local_range = np.linspace(main_bounds[0], main_bounds[1], num_points)

            if mode == "Custom":
                expr = config.get("expression", "")
                try:
                    if "lam" in expr:
                        n_func = lambda lam: np.array([float(safe_eval_custom_expression(expr, lam=l)) for l in lam])
                    else:
                        eps_val = float(safe_eval_custom_expression(expr))
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
                    from gap_plasmon_2d.materials.material__configuration import get_material_permittivity
                    n_vals, k_vals, eps_list = [], [], []
                    for l in local_range:
                        perm_val = get_material_permittivity(
                            config.get("material", "").strip(),
                            l,
                            JSON_COMBINED_PATH,
                            DATA_DIR
                        )
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
                    from gap_plasmon_2d.materials.material__configuration import build_material_configuration_dynamic

                    df = pd.DataFrame([{"key": self.role_name, "material": config}])

                    n_vals = []
                    k_vals = []
                    eps_list = []

                    for l in local_range:
                        eps_val = build_material_configuration_dynamic(
                            df, l, JSON_COMBINED_PATH, None
                        )[self.role_name]
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

            for comp in self.comparison_widgets:
                comp_conf = comp.get_config()
                if comp_conf.get("type", "None") == "None":
                    continue

                bounds_fn_c = get_bounds_function_for_mode(comp_conf["type"])

                comp_bounds = resolve_lambda_bounds(bounds_fn_c, comp_conf, JSON_COMBINED_PATH, DATA_DIR)
                comp_range = np.linspace(comp_bounds[0], comp_bounds[1], num_points)

                if comp_conf["type"] == "Custom":
                    expr = comp_conf.get("expression", "")
                    if "lam" in expr:
                        n_comp = lambda lam: np.array([float(safe_eval_custom_expression(expr, lam=l)) for l in lam])
                    else:
                        val = float(safe_eval_custom_expression(expr))
                        n_comp = lambda lam: val * np.ones_like(lam)
                    k_comp = lambda lam: np.zeros_like(lam)
                    eps_comp = compute_epsilon(n_comp, k_comp, comp_range)
                    n_vals_c = n_comp(comp_range)
                    k_vals_c = k_comp(comp_range)

                elif comp_conf["type"] == "Standard":
                    from gap_plasmon_2d.materials.material__configuration import get_material_permittivity
                    n_vals_c, k_vals_c, eps_list_c = [], [], []
                    for l in comp_range:
                        epsv = get_material_permittivity(
                            comp_conf.get("material", ""),
                            l,
                            JSON_COMBINED_PATH,
                            DATA_DIR
                        )
                        eps_list_c.append(epsv)
                        sq = np.sqrt(epsv)
                        n_vals_c.append(np.real(sq))
                        k_vals_c.append(np.imag(sq))
                    n_vals_c = np.array(n_vals_c)
                    k_vals_c = np.array(k_vals_c)
                    eps_comp = np.array(eps_list_c)

                elif comp_conf["type"] == "RefractiveIndex":
                    from gap_plasmon_2d.materials.material__configuration import build_material_configuration_dynamic

                    df = pd.DataFrame([{"key": self.role_name, "material": comp_conf}])

                    n_vals_c = []
                    k_vals_c = []
                    eps_list_c = []

                    for l in comp_range:
                        epsv = build_material_configuration_dynamic(
                            df, l, JSON_COMBINED_PATH, None
                        )[self.role_name]
                        eps_list_c.append(epsv)

                        sq = np.sqrt(epsv)
                        n_vals_c.append(np.real(sq))
                        k_vals_c.append(np.imag(sq))

                    n_vals_c = np.array(n_vals_c)
                    k_vals_c = np.array(k_vals_c)
                    eps_comp = np.array(eps_list_c)

                if comp_conf["type"] in ["Standard", "Custom"]:
                    label_base = comp_conf.get("material", comp_conf.get("expression", ""))
                else:
                    label_base = f"RI {comp_conf.get('shelf')}:{comp_conf.get('page')}"

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
        
        self.geometry_dropdown = widgets.Dropdown(
            options=[""] + _load_geometry_names(),
            description="Geometry:",
            layout=widgets.Layout(width="300px")
        )
        self.geometry_dropdown.observe(self._on_geometry_change, names="value")

        # ──────────────────────────────────────────────────────────────────────
        # Mécanisme principal : notification directe en mémoire
        # ──────────────────────────────────────────────────────────────────────
        self._unsubscribe_geometry_event = subscribe_geometry_changed(
            self._on_geometry_event
        )

        # ──────────────────────────────────────────────────────────────────────
        # Fallback : watcher si le JSON change hors de cette UI
        # ──────────────────────────────────────────────────────────────────────
        self._geom_watcher, self._geom_handler = start_watcher(
            path=str(get_geometry_save_path()),
            callback=self._on_geom_fs_event,
            extensions=[".json"],
            recursive=False,
            debounce_interval=0.2,
        )


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
        self.current_roles = DEFAULT_ROLES.copy()

        for i, role in enumerate(roles):
            self.tab.set_title(i, role)
        self.preconfigs = {
            "Preconfig Structure 1": {
                "perm_env": {"type": "None"},
                "perm_reso": {"type": "Standard", "material": "Silver"},
                "perm_mol": {"type": "None"},
                "perm_func": {"type": "None"},
                "perm_diel": {"type": "Custom", "expression": "1.45**2"},
                "perm_gap": {"type": "Custom", "expression": "1.45**2"},
                "perm_metalliclayer": {"type": "Standard", "material": "Gold"},
                "perm_XIAOYI": {"type": "Standard", "material": "Si"},
                "perm_accroche": {"type": "None"},
                "perm_sub": {"type": "Standard", "material": "ITO"}
            },
            "Preconfig Structure 2": {
                "perm_env": {"type": "None"},
                "perm_reso": {"type": "Standard", "material": "Silver"},
                "perm_mol": {"type": "None"},
                "perm_func": {"type": "None"},
                "perm_diel": {"type": "Custom", "expression": "1.45**2"},
                "perm_gap": {"type": "Custom", "expression": "1.45**2"},
                "perm_metalliclayer": {"type": "Standard", "material": "Gold"},
                "perm_XIAOYI": {"type": "Standard", "material": "Si"},
                "perm_accroche": {"type": "Standard", "material": "Aluminium"},
                "perm_sub": {"type": "Custom", "expression": "1.50**2"}
            }
        }
        self.load_preconfigs()
        self.preconfig_dropdown = widgets.Dropdown(options=self._get_preconfig_options(), description="Preconfig:")
        self.preconfig_name_text = widgets.Text(description="Preconfig Name:", placeholder="Enter a name...")
        self.add_preconfig_btn = widgets.Button(description="Add Preconfig", button_style="warning")
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
        
        
        self.add_save_btn = widgets.Button(
            description="Add Material config(s)",
            button_style="success",
            layout=widgets.Layout(width='200px')
        )
        
        def _on_add_and_save(_):
            self.on_add_config(_)    # ajoute la config
            self.on_save_quit(_)     # puis sauve et quitte
        self.add_save_btn.on_click(_on_add_and_save)        
            
            
        self.config_dropdown = widgets.Dropdown(options=[], description="Saved Configs:", style={"description_width": "initial"})
        self.load_config_btn = widgets.Button(description="Load Config")
        self.update_config_btn = widgets.Button(description="Update Config")
        self.delete_config_btn = widgets.Button(description="Delete Config", button_style="danger")
        self.load_config_btn.on_click(self.on_load_config)
        self.update_config_btn.on_click(self.on_update_config)
        self.delete_config_btn.on_click(self.on_delete_config)
        self.container = widgets.VBox([
            self.geometry_dropdown,
            self.preconfig_control_box, self.tab, self.config_name_text,
            self.add_save_btn,
            widgets.HBox([self.config_dropdown, self.load_config_btn, self.update_config_btn, self.delete_config_btn]),
            self.output
        ])
        self.all_configs = []
        self.load_saved_configs()
        

    def close(self):
        """
        Nettoyage optionnel de l'instance.
        Désabonne l'event bus et arrête le timer du handler watcher.
        """
        try:
            if hasattr(self, "_unsubscribe_geometry_event") and self._unsubscribe_geometry_event is not None:
                self._unsubscribe_geometry_event()
                self._unsubscribe_geometry_event = None
        except Exception as e:
            print(f"[MaterialSelector] erreur unsubscribe geometry event : {e}")

        try:
            if hasattr(self, "_geom_handler") and self._geom_handler is not None:
                self._geom_handler.stop()
        except Exception as e:
            print(f"[MaterialSelector] erreur arrêt handler watcher : {e}")


    def _on_geometry_change(self, change):
        name = change["new"]

        if not name:
            display_roles = DEFAULT_ROLES.copy()
        else:
            geom_entry = _load_geometry_by_name(name)
            if geom_entry is None:
                with self.output:
                    clear_output()
                    print(f"[WARN] Géométrie '{name}' introuvable, affichage par défaut.")
                display_roles = DEFAULT_ROLES.copy()
            else:
                geom = geom_entry.get("geometry", {})
                display_roles = _geometry_to_display_roles(geom)

        self._rebuild_tabs(display_roles)

        for role, widget_role in self.role_widgets.items():
            if role not in display_roles:
                widget_role.mode_dropdown.value = "None"

        # IMPORTANT : mémoriser uniquement les rôles réellement affichés
        self.current_roles = display_roles.copy()



    # ──────────────────────────────────────────────────────────────────────
    #  Callbacks liés au watcher de géométrie
    # ──────────────────────────────────────────────────────────────────────
    def _refresh_geometry_dropdown(self):
        """
        Recharge la liste des géométries disponibles tout en conservant
        la sélection courante si elle existe encore.
        """
        current_value = self.geometry_dropdown.value
        new_options = [""] + _load_geometry_names()

        old_options = list(self.geometry_dropdown.options)
        if old_options == new_options:
            return

        target_value = current_value if current_value in new_options else ""

        # On change d'abord les options
        self.geometry_dropdown.options = new_options

        # Puis on ne modifie la valeur que si nécessaire
        if self.geometry_dropdown.value != target_value:
            self.geometry_dropdown.value = target_value

    def _on_geometry_event(self):
        """
        Callback principal déclenché par l'event bus en mémoire après une
        sauvegarde / mise à jour / suppression de géométrie.
        """
        self._refresh_geometry_dropdown()

    def _on_geom_fs_event(self):
        """
        Callback fallback appelé par le watcher fichier.
        Ici, le handler watchdog a déjà rebasculé dans la boucle Jupyter.
        """
        self._refresh_geometry_dropdown()







    def _rebuild_tabs(self, roles_list):
        """
        Reconstruit self.tab.children et titres en ne gardant
        que les widgets pour les rôles passés.
        """
        children = []
        for role in roles_list:
            # créer à la volée s’il n’existe pas encore
            if role not in self.role_widgets:
                self.role_widgets[role] = MaterialRoleWidget(role, self.library, self.standard_list)
            children.append(self.role_widgets[role].container)
        self.tab.children = tuple(children)
        for i, role in enumerate(roles_list):
            self.tab.set_title(i, role)



    def _get_preconfig_options(self):
        options = [("None", "")]
        for key in self.preconfigs:
            options.append((key, key))
        return options
    
    def on_preconfig_change(self, change):
        preconfig_id = change["new"]
        if preconfig_id == "":
            for role, widget_role in self.role_widgets.items():
                widget_role.reset_widget()
        else:
            self.apply_preconfig(preconfig_id)
    
    def apply_preconfig(self, preconfig_id):
        mapping = self.preconfigs.get(preconfig_id, {})
        for role, widget_role in self.role_widgets.items():
            if role in mapping:
                widget_role.apply_config(mapping[role])
            else:
                widget_role.apply_config({"type": "None"})
    
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
                # with self.output:
                #     print(f"Preconfigurations loaded from {preconfig_file}")
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
            # with self.output:
            #     print(f"Configurations loaded from {config_file}")
        except Exception as e:
            with self.output:
                print(f"Unable to load configurations: {e}")
    
    def update_config_dropdown(self):
        options = [(cfg["config_name"], cfg) for cfg in self.all_configs]
        self.config_dropdown.options = options if options else [("None", None)]
    
    def on_add_config(self, b):
        # 1) Définir l'ordre des rôles d'après la géométrie sélectionnée
        geom_name = self.geometry_dropdown.value
        if geom_name:
            geom_entry = _load_geometry_by_name(geom_name)
            if geom_entry is not None:
                ordered_roles = _geometry_to_display_roles(geom_entry.get("geometry", {}))
            else:
                ordered_roles = self.current_roles.copy()
        else:
            ordered_roles = self.current_roles.copy()

        # 2) Construire MATERIALS_CONFIG et RI_OVERRIDES
        config_list = []
        ri_overrides = {}
        for role in ordered_roles:
            widget = self.role_widgets.get(role)
            if widget is not None:
                mat_cfg = widget.get_config()
            else:
                mat_cfg = {"type": "None"}
            # Forcer à {"type":"None"} si besoin
            if not isinstance(mat_cfg, dict) or mat_cfg.get("type") == "None":
                mat_cfg = {"type": "None"}
            config_list.append({"key": role, "material": mat_cfg})
            if mat_cfg.get("type") == "RefractiveIndex":
                ri_overrides[role] = mat_cfg

        # 3) Préparer le nom et éviter les doublons
        name = self.config_name_text.value.strip() or f"Config_{len(self.all_configs)+1}"
        # On retire l'ancien éventuel
        self.all_configs = [cfg for cfg in self.all_configs if cfg["config_name"] != name]

        # 4) Créer et enregistrer la nouvelle config
        new_cfg = {
            "config_name": name,
            "MATERIALS_CONFIG": config_list,
            "RI_OVERRIDES": ri_overrides
        }
        self.all_configs.append(new_cfg)
        self.update_config_dropdown()
        with self.output:
            clear_output()
            print(f"'{name}' configuration added.")

        # 5) Persister tout de suite
        self._persist_configs()
        notify_material_config_changed()
        self.config_name_text.value = ""

        # 6) Réinitialiser l’UI
        for w in self.role_widgets.values():
            w.reset_widget()

    
    def on_save_quit(self, b):
        if not self.all_configs:
            with self.output:
                print("No configuration has been added.")
            return
        self._persist_configs()
        with self.output:
            print(f"All configurations saved in:\n{os.path.join(CONFIGURATIONS_DIR, 'material_config.json')}")

            
    def on_load_config(self, b):
        
        # 0) Ne pas tenir compte de la géométrie précédente
        self.geometry_dropdown.value = ""

        selected = self.config_dropdown.value
        if selected is None:
            with self.output:
                clear_output()
                print("No configuration selected for loading.")
            return

        # 1) Déterminer la liste des rôles à afficher d'après MATERIALS_CONFIG
        roles_to_show = [entry["key"] for entry in selected["MATERIALS_CONFIG"]]

        # 2) Reconstruire les onglets pour n'afficher que ces rôles
        self._rebuild_tabs(roles_to_show)
        self.current_roles = roles_to_show.copy()

        # 3) Initialiser chaque widget de rôle avec sa config complète,
        #    y compris les overrides sauvegardés
        for role in self.role_widgets:
            w = self.role_widgets[role]

            if role in roles_to_show:
                mat_cfg = next(
                    (e["material"] for e in selected["MATERIALS_CONFIG"] if e["key"] == role),
                    None
                )
                w.apply_config(mat_cfg if isinstance(mat_cfg, dict) else {"type": "None"})
            else:
                w.apply_config({"type": "None"})

        # 4) Mettre à jour le nom de la config et informer l'utilisateur
        self.config_name_text.value = selected["config_name"]
        with self.output:
            clear_output()
            print(f"Configuration '{selected['config_name']}' loaded.")

            


    def on_update_config(self, b):
        selected = self.config_dropdown.value
        if selected is None:
            with self.output:
                clear_output()
                print("No configuration selected for update.")
            return

        # Récupérer l'ancien nom via le label du dropdown
        old_label = next((lbl for lbl, obj in self.config_dropdown.options if obj is selected), None)
        old_name  = old_label or selected.get("config_name", "")

        # 1) Même logique que pour on_add_config pour l'ordre des rôles
        geom_name = self.geometry_dropdown.value
        if geom_name:
            geom_entry = _load_geometry_by_name(geom_name)
            if geom_entry is not None:
                ordered_roles = _geometry_to_display_roles(geom_entry.get("geometry", {}))
            else:
                ordered_roles = self.current_roles.copy()
        else:
            ordered_roles = self.current_roles.copy()

        # 2) Recomposer MATERIALS_CONFIG et RI_OVERRIDES
        updated_config_list = []
        ri_overrides = {}
        for role in ordered_roles:
            widget = self.role_widgets.get(role)
            if widget is not None:
                mat_cfg = widget.get_config()
            else:
                mat_cfg = {"type": "None"}
            if not isinstance(mat_cfg, dict) or mat_cfg.get("type") == "None":
                mat_cfg = {"type": "None"}
            updated_config_list.append({"key": role, "material": mat_cfg})
            if mat_cfg.get("type") == "RefractiveIndex":
                ri_overrides[role] = mat_cfg

        # 3) Appliquer dans self.all_configs
        new_name = self.config_name_text.value.strip() or old_name
        cfg = None
        for entry in self.all_configs:
            if entry["config_name"] == old_name:
                entry["MATERIALS_CONFIG"] = updated_config_list
                entry["RI_OVERRIDES"]    = ri_overrides
                entry["config_name"]     = new_name
                cfg = entry
                break

        # 4) Patch convergence_results.json si renommage
        convergence_json = os.path.join(WORKSPACE_DIR, "Convergence", "convergence_results.json")
        if new_name != old_name and os.path.exists(convergence_json):
            with open(convergence_json, "r", encoding="utf-8") as f:
                master = json.load(f)
            configs = master.get("configs", {})
            if old_name in configs:
                configs[new_name] = configs.pop(old_name)
                with open(convergence_json, "w", encoding="utf-8") as f:
                    json.dump(master, f, indent=2)
                with self.output:
                    print(f"[PATCH] convergence_results.json : {old_name} → {new_name}")
            else:
                with self.output:
                    print(f"[INFO] Pas d'entrée à renommer dans convergence_results.json pour {old_name}.")

        # 5) Rafraîchir l'UI et re-sélectionner
        self.update_config_dropdown()
        self.config_dropdown.value = cfg
        with self.output:
            clear_output()
            print(f"'{old_name}' configuration  updated (new name : '{new_name}').")

        # 6) Persister les changements
        self._persist_configs()
        notify_material_config_changed()

    def on_delete_config(self, b):
        selected = self.config_dropdown.value
        if selected is None:
            with self.output:
                print("No configuration selected for deletion.")
            return

        name = selected["config_name"]
        # 1) Retirer de la liste
        self.all_configs = [
            cfg for cfg in self.all_configs
            if cfg["config_name"] != name
        ]
        # 2) Mettre à jour le dropdown
        self.update_config_dropdown()
        with self.output:
            print(f"'{name}' configuration  deleted.")

        # 3) Persister immédiatement
        self._persist_configs()
        notify_material_config_changed()


    def _persist_configs(self):
        """Écrit self.all_configs dans material_config.json"""
        if not os.path.exists(CONFIGURATIONS_DIR):
            os.makedirs(CONFIGURATIONS_DIR)
        path = os.path.join(CONFIGURATIONS_DIR, "material_config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"ALL_CONFIGS": self.all_configs}, f, indent=2)


    def display(self):
        display(self.container)

