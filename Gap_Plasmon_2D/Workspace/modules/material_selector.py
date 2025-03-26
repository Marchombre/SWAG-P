import os
import yaml
import json
import ipywidgets as widgets
from IPython.display import display

# --- Utility Functions ---
def load_catalog_full(catalog_file):
    with open(catalog_file, "r", encoding="utf-8") as f:
        lib = yaml.safe_load(f)
    return lib

def load_combined_materials(json_combined_path):
    with open(json_combined_path, 'r', encoding='utf-8') as f:
        materials_data = json.load(f)
    return materials_data

def get_standard_materials(json_combined_path, data_directory):
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
        print(f"[WARNING] Data directory '{data_directory}' does not exist or is not a directory.")
    return sorted(found)

# --- Class RefractiveIndexArboWidget ---
class RefractiveIndexArboWidget:
    def __init__(self, library):
        self.library = library
        self.shelf_dropdown = widgets.Dropdown(description="Shelf:")
        self.book_dropdown = widgets.Dropdown(description="Book:")
        self.page_dropdown = widgets.Dropdown(description="Page:")
        self.container = widgets.HBox([self.shelf_dropdown, self.book_dropdown, self.page_dropdown])
        self._populate_shelf()
        self.shelf_dropdown.observe(self.on_shelf_changed, names='value')
        self.book_dropdown.observe(self.on_book_changed, names='value')

    def _populate_shelf(self):
        options = []
        for i, entry in enumerate(self.library):
            if "SHELF" in entry:
                disp = entry.get("name", entry["SHELF"])
                options.append((disp, i))
            elif "DIVIDER" in entry:
                disp = f"—— {entry['DIVIDER']} ——"
                options.append((disp, None))
        self.shelf_dropdown.options = options

    def on_shelf_changed(self, change):
        val = change['new']
        if val is None:
            self.book_dropdown.options = []
            self.page_dropdown.options = []
            return
        shelf_item = self.library[val]
        content = shelf_item.get("content", [])
        book_options = []
        for j, bk in enumerate(content):
            if "BOOK" in bk:
                disp = bk.get("name", bk["BOOK"])
                book_options.append((disp, j))
            elif "DIVIDER" in bk:
                disp = f"—— {bk['DIVIDER']} ——"
                book_options.append((disp, None))
        self.book_dropdown.options = book_options
        self.page_dropdown.options = []

    def on_book_changed(self, change):
        book_val = change['new']
        shelf_val = self.shelf_dropdown.value
        if shelf_val is None or book_val is None:
            self.page_dropdown.options = []
            return
        shelf_item = self.library[shelf_val]
        shelf_content = shelf_item.get("content", [])
        if book_val < 0 or book_val >= len(shelf_content):
            self.page_dropdown.options = []
            return
        book_dict = shelf_content[book_val]
        page_options = []
        for k, pg in enumerate(book_dict.get("content", [])):
            if "PAGE" in pg:
                disp = pg.get("name", pg["PAGE"])
                page_options.append((disp, k))
            elif "DIVIDER" in pg:
                disp = f"—— {pg['DIVIDER']} ——"
                page_options.append((disp, None))
        self.page_dropdown.options = page_options

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
        if book_val < 0 or book_val >= len(shelf_content):
            return None
        book_dict = shelf_content[book_val]
        if "BOOK" not in book_dict:
            return None
        book_key = book_dict["BOOK"]

        page_val = self.page_dropdown.value
        if page_val is None:
            return None
        book_content = book_dict.get("content", [])
        if page_val < 0 or page_val >= len(book_content):
            return None
        page_dict = book_content[page_val]
        if "PAGE" not in page_dict:
            return None
        page_key = page_dict["PAGE"]
        data_field = page_dict.get("data", "")

        return {
            "shelf": shelf_key,
            "book": book_key,
            "page": page_key,
            "data": data_field
        }

    def set_selection(self, selection):
        for i, entry in enumerate(self.library):
            if entry.get("SHELF", "") == selection.get("shelf", ""):
                self.shelf_dropdown.value = i
                break
        if not hasattr(self, "_set_book_and_page_handler"):
            def set_book_and_page(change):
                for option in self.book_dropdown.options:
                    if option[0] == selection.get("book", "") or (
                        option[1] is not None and 
                        self.library[self.shelf_dropdown.value].get("content", [])[option[1]].get("BOOK", "") == selection.get("book", "")
                    ):
                        self.book_dropdown.value = option[1]
                        break
                if not hasattr(self, "_set_page_handler"):
                    def set_page(change2):
                        for option in self.page_dropdown.options:
                            if option[0] == selection.get("page", ""):
                                self.page_dropdown.value = option[1]
                                break
                    self._set_page_handler = set_page
                    self.book_dropdown.observe(self._set_page_handler, names="value")
            self._set_book_and_page_handler = set_book_and_page
            self.shelf_dropdown.observe(self._set_book_and_page_handler, names="value")
        else:
            try:
                self.shelf_dropdown.unobserve(self._set_book_and_page_handler, names="value")
            except ValueError:
                pass
            self.shelf_dropdown.observe(self._set_book_and_page_handler, names="value")

# --- Class MaterialRoleWidget ---
class MaterialRoleWidget:
    def __init__(self, role_name, library, standard_list):
        self.role_name = role_name
        self.library = library
        self.standard_list = standard_list

        self.mode_dropdown = widgets.Dropdown(
            options=["None", "Custom", "Standard", "RefractiveIndex"],
            description="Mode:"
        )
        self.custom_text = widgets.Text(placeholder="Enter expression")
        self.standard_dropdown = widgets.Dropdown(
            options=standard_list,
            description="Standard:"
        )
        self.ri_widget = RefractiveIndexArboWidget(library)

        self.container = widgets.VBox([
            self.mode_dropdown,
            self.custom_text,
            self.standard_dropdown,
            self.ri_widget.container
        ])

        self._update_visibility()
        self.mode_dropdown.observe(self._on_mode_change, names='value')

    def _on_mode_change(self, change):
        self._update_visibility()

    def _update_visibility(self):
        mode = self.mode_dropdown.value
        if mode == "None":
            self.custom_text.layout.display = 'none'
            self.standard_dropdown.layout.display = 'none'
            self.ri_widget.container.layout.display = 'none'
        elif mode == "Custom":
            self.custom_text.layout.display = ''
            self.standard_dropdown.layout.display = 'none'
            self.ri_widget.container.layout.display = 'none'
        elif mode == "Standard":
            self.custom_text.layout.display = 'none'
            self.standard_dropdown.layout.display = ''
            self.ri_widget.container.layout.display = 'none'
        elif mode == "RefractiveIndex":
            self.custom_text.layout.display = 'none'
            self.standard_dropdown.layout.display = 'none'
            self.ri_widget.container.layout.display = ''

    def get_config(self):
        mode = self.mode_dropdown.value
        if mode == "None":
            return {"type": "None"}
        elif mode == "Custom":
            expr = self.custom_text.value.strip()
            if not expr:
                expr = "None"
            return {"type": "Custom", "expression": expr}
        elif mode == "Standard":
            mat = self.standard_dropdown.value
            return {"type": "Standard", "material": mat}
        elif mode == "RefractiveIndex":
            sel = self.ri_widget.get_selection()
            if sel is None:
                return {"type": "None"}
            return {
                "type": "RefractiveIndex",
                "shelf": sel["shelf"],
                "book":  sel["book"],
                "page":  sel["page"],
                "data":  sel["data"]
            }

# --- Class MaterialSelectorTabbedNotebook ---
class MaterialSelectorTabbedNotebook:
    def __init__(self, roles):
        # Determine paths
        script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
        workspace_dir = os.path.abspath(os.path.join(script_dir, ".."))
        catalog_path = os.path.join(workspace_dir, "catalog_nk.yml")
        data_dir = os.path.join(workspace_dir, "data")
        json_combined_path = os.path.join(data_dir, "combined_materials.json")
        # CONFIGURATIONS directory for persistent configurations
        self.CONFIGURATIONS_dir = os.path.join(workspace_dir, "CONFIGURATIONS")

        self.library = load_catalog_full(catalog_path)
        self.standard_list = get_standard_materials(json_combined_path, data_dir)
        self.roles = roles

        self.output = widgets.Output()

        # Create a tab with one MaterialRoleWidget per role
        self.role_widgets = {}
        children = []
        for role in roles:
            widget_role = MaterialRoleWidget(role, self.library, self.standard_list)
            self.role_widgets[role] = widget_role
            children.append(widget_role.container)
        self.tab = widgets.Tab(children=children)
        for i, role in enumerate(roles):
            self.tab.set_title(i, role)

        # Updated default preconfigurations for new geometry:
        # - "perm_gap" corresponds to gap polymer (central gap)
        # - "perm_diel" corresponds to the lateral dielectric layer
        # - "perm_func" and "perm_mol" remain for functionalisation and molecule respectively.
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
                "perm_accroche": {"type": "None"},
                "perm_sub": {"type": "Standard", "material": "ITO"}

            },
            "Preconfig Structure 2": {
                "perm_env": {"type": "None"},
                "perm_gap": {"type": "Custom", "expression": "1.45**2"},
                "perm_diel": {"type": "Custom", "expression": "1.45**2"},
                "perm_func": {"type": "None"},
                "perm_mol": {"type": "None"},
                "perm_reso": {"type": "Standard", "material": "Silver"},
                "perm_metalliclayer": {"type": "Standard", "material": "Gold"},
                "perm_accroche": {"type": "Standard", "material": "Aluminium"},
                "perm_sub": {"type": "Custom", "expression": "1.50**2"}
            }
        }
        self.load_preconfigs()

        self.preconfig_dropdown = widgets.Dropdown(
            options=self._get_preconfig_options(),
            description="Preconfig:"
        )
        self.preconfig_name_text = widgets.Text(
            description="Preconfig Name:",
            placeholder="Enter a name..."
        )
        self.add_preconfig_btn = widgets.Button(description="Add Preconfig", button_style="info")
        self.update_preconfig_btn = widgets.Button(description="Update Preconfig")
        self.delete_preconfig_btn = widgets.Button(description="Delete Preconfig", button_style="danger")
        self.preconfig_dropdown.observe(self.on_preconfig_change, names="value")
        self.add_preconfig_btn.on_click(self.on_add_preconfig)
        self.update_preconfig_btn.on_click(self.on_update_preconfig)
        self.delete_preconfig_btn.on_click(self.on_delete_preconfig)

        self.preconfig_control_box = widgets.HBox([
            self.preconfig_dropdown,
            self.preconfig_name_text,
            self.add_preconfig_btn,
            self.update_preconfig_btn,
            self.delete_preconfig_btn
        ])

        self.config_name_text = widgets.Text(
            description="Configuration Name:",
            placeholder="Enter the config name"
        )

        self.add_config_btn = widgets.Button(description="Add Material config")
        self.save_quit_btn = widgets.Button(description="Save & Quit", button_style='success')
        self.add_config_btn.on_click(self.on_add_config)
        self.save_quit_btn.on_click(self.on_save_quit)

        self.config_dropdown = widgets.Dropdown(
            options=[],
            description="Saved Configs:",
            style={'description_width': 'initial'}
        )
        self.load_config_btn = widgets.Button(description="Load Config")
        self.update_config_btn = widgets.Button(description="Update Config")
        self.delete_config_btn = widgets.Button(description="Delete Config", button_style='danger')
        self.load_config_btn.on_click(self.on_load_config)
        self.update_config_btn.on_click(self.on_update_config)
        self.delete_config_btn.on_click(self.on_delete_config)

        self.container = widgets.VBox([
            self.preconfig_control_box,
            self.tab,
            self.config_name_text,
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
                            "book":  config.get("book", ""),
                            "page":  config.get("page", ""),
                            "data":  config.get("data", "")
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
                print(f"Preconfigurations saved in {preconfig_file}")
        except Exception as e:
            with self.output:
                print(f"Error saving preconfigurations in {preconfig_file}: {e}")

    def load_saved_configs(self):
        module_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_dir = os.path.dirname(module_dir)
        CONFIGURATIONS_dir = os.path.join(workspace_dir, "CONFIGURATIONS")
        config_file = os.path.join(CONFIGURATIONS_dir, "material_config.json")
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
        config_dict = {
            "config_name": config_name,
            "MATERIALS_CONFIG": config_list,
            "RI_OVERRIDES": ri_overrides
        }
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
        module_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_dir = os.path.dirname(module_dir)
        CONFIGURATIONS_dir = os.path.join(workspace_dir, "CONFIGURATIONS")
        if not os.path.exists(CONFIGURATIONS_dir):
            os.makedirs(CONFIGURATIONS_dir)
        config_file = os.path.join(CONFIGURATIONS_dir, "material_config.json")
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
                        selection = {
                            "shelf": mat_config.get("shelf", ""),
                            "book":  mat_config.get("book", ""),
                            "page":  mat_config.get("page", ""),
                            "data":  mat_config.get("data", "")
                        }
                        widget_role.ri_widget.set_selection(selection)
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

# --- Default roles updated for new geometry ---
# New keys:
# - "perm_gap" represents the gap polymer (central gap)
# - "perm_diel" represents the lateral dielectric layer
# Other keys remain as before.
DEFAULT_ROLES = [
    "perm_env",
    "perm_gap",
    "perm_diel",
    "perm_func",
    "perm_mol",
    "perm_reso",
    "perm_metalliclayer",
    "perm_accroche",
    "perm_sub"
]

# --- Instantiation and display ---
selector = MaterialSelectorTabbedNotebook(DEFAULT_ROLES)
