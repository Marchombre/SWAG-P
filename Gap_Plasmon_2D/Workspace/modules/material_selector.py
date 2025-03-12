#!/usr/bin/env python3
"""
material_selector_tabbed.py

Exemple PyQt6 qui gère plusieurs rôles (perm_env, etc.) dans des onglets (QTabWidget).
Pour chaque rôle, 4 modes:
 - None
 - Custom (valeur/expr)
 - Standard (JSON + .txt)
 - RefractiveIndex (navigation nkexplorer-like)

On affiche le champ 'name' pour shelf/book/page (ex: "Al (Aluminium)"),
tout en stockant en interne shelf="Al", book=..., page=... tel que défini
dans catalog_nk.yml, sans omission ni simplification.

A la validation, on enregistre dans __main__:
 - MATERIALS_CONFIG : DataFrame [ {key:role, material:...}, ... ]
 - RI_OVERRIDES     : dict { role: {...}, ... } pour RefractiveIndex

Ensuite, dans un Notebook, on peut récupérer:
   import __main__
   df_config = __main__.MATERIALS_CONFIG
   ri_over = __main__.RI_OVERRIDES
   ...

Usage:
   python material_selector_tabbed.py
"""

import sys
import os
import yaml
import json

import pandas as pd

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QLabel, QLineEdit, QComboBox, QPushButton, QMessageBox,
    QVBoxLayout, QHBoxLayout, QGroupBox
)
from PyQt6.QtCore import Qt

##############################################################################
# 0) CONFIG & CHEMINS
##############################################################################

script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.abspath(os.path.join(script_dir, ".."))

catalog_path = os.path.join(workspace_dir, "catalog_nk.yml")
data_dir = os.path.join(workspace_dir, "data")
json_combined_path = os.path.join(data_dir, "combined_materials.json")

DEFAULT_ROLES = [
    "perm_env",
    "perm_dielec",
    "perm_sub",
    "perm_reso",
    "perm_metalliclayer",
    "perm_accroche",
    "perm_func",
    "perm_mol"
]


##############################################################################
# 1) CHARGER CATALOG_NK.YML COMME NKEXPLORER (SANS OMISSIONS)
##############################################################################

def load_catalog_full(catalog_file):
    """
    Charge catalog_nk.yml complet,
    renvoie une liste library = [ {SHELF:..., name:..., content:[...]}, {DIVIDER:...}, ... ]
    """
    with open(catalog_file, "r", encoding="utf-8") as f:
        lib = yaml.safe_load(f)
    return lib


##############################################################################
# 2) LISTER LES MATERIAUX STANDARD (JSON + .txt)
##############################################################################

def load_combined_materials(json_combined_path):
    """
    Loads the combined JSON file containing ExpData and BrendelBormann data.
    Returns a dictionary of materials.
    """
    with open(json_combined_path, 'r') as f:
        materials_data = json.load(f)
    return materials_data

def get_standard_materials(json_combined_path, data_directory):
    """
    Parcourt le JSON combiné (ExpData, Brendel-Bormann) et le répertoire data_directory
    pour en extraire la liste de tous les noms de matériaux disponibles.

    1) Charge le JSON combiné (via load_combined_materials) et récupère la liste de ses clés.
    2) Parcourt le dossier data_directory pour y chercher tous les fichiers .txt.
    3) Retourne la liste triée des noms de matériaux trouvés (sans doublons).

    Parameters
    ----------
    json_combined_path : str
        Chemin vers le fichier JSON combiné contenant ExpData/BrendelBormann.
    data_directory : str
        Chemin vers le dossier contenant d'éventuels fichiers .txt

    Returns
    -------
    list
        Liste triée des noms de matériaux (str).
    """
    found = set()

    # 1) Lecture du JSON via load_combined_materials
    if os.path.isfile(json_combined_path):
        try:
            materials_data = load_combined_materials(json_combined_path)
            # Les clés du dictionnaire JSON représentent les noms de matériaux
            found.update(materials_data.keys())
        except Exception as e:
            print(f"[AVERTISSEMENT] Impossible de lire/parsing le JSON '{json_combined_path}': {e}")
    else:
        print(f"[AVERTISSEMENT] Le fichier JSON '{json_combined_path}' est introuvable.")

    # 2) Recherche des fichiers .txt dans data_directory
    if os.path.isdir(data_directory):
        for root, dirs, files in os.walk(data_directory):
            for fn in files:
                if fn.lower().endswith(".txt"):
                    # On ajoute le nom du fichier (sans extension) à l'ensemble found
                    name = os.path.splitext(fn)[0]
                    found.add(name)
    else:
        print(f"[AVERTISSEMENT] Le dossier data '{data_directory}' n’existe pas ou n’est pas un répertoire.")

    # 3) Retourne la liste triée
    return sorted(found)



##############################################################################
# 3) WIDGET REFRACTIVEINDEX (SHELF->BOOK->PAGE), AFFICHANT LE CHAMP "NAME"
##############################################################################

class RefractiveIndexArboWidget(QWidget):
    """
    Trois QComboBox : shelf, book, page. On affiche le champ 'name' (nkexplorer-like).
    En interne, on stocke .SHELF, .BOOK, .PAGE (ex: shelf="main", book="Al", page="Rakic-BB").
    """
    def __init__(self, library, parent=None):
        super().__init__(parent)
        self.library = library  # [ {SHELF:"main", name:"MAIN - simple inorganic materials", content:[...] }, {DIVIDER:...}, ... ]

        layout = QHBoxLayout(self)

        self.shelf_combo = QComboBox()
        self.book_combo  = QComboBox()
        self.page_combo  = QComboBox()

        # Populate shelf_combo
        for i, entry in enumerate(self.library):
            if "SHELF" in entry:
                disp_shelf = entry.get("name", entry["SHELF"])  # ex: "Al (Aluminium)"
                self.shelf_combo.addItem(disp_shelf, i)
            elif "DIVIDER" in entry:
                divtxt = f"—— {entry['DIVIDER']} ——"
                idx = self.shelf_combo.count()
                self.shelf_combo.addItem(divtxt, None)
                model_item = self.shelf_combo.model().item(idx)
                model_item.setEnabled(False)

        self.shelf_combo.currentIndexChanged.connect(self.on_shelf_changed)
        self.book_combo.currentIndexChanged.connect(self.on_book_changed)

        layout.addWidget(QLabel("Shelf:"))
        layout.addWidget(self.shelf_combo)
        layout.addWidget(QLabel("Book:"))
        layout.addWidget(self.book_combo)
        layout.addWidget(QLabel("Page:"))
        layout.addWidget(self.page_combo)

    def on_shelf_changed(self, idx):
        self.book_combo.clear()
        self.page_combo.clear()

        shelf_data = self.shelf_combo.itemData(idx)
        if shelf_data is None:
            # c'est un divider
            return

        shelf_item = self.library[shelf_data]
        content = shelf_item.get("content", [])
        for j, bk in enumerate(content):
            if "BOOK" in bk:
                disp_book = bk.get("name", bk["BOOK"])
                self.book_combo.addItem(disp_book, j)
            elif "DIVIDER" in bk:
                divb = f"—— {bk['DIVIDER']} ——"
                idxb = self.book_combo.count()
                self.book_combo.addItem(divb, None)
                itb = self.book_combo.model().item(idxb)
                itb.setEnabled(False)

    def on_book_changed(self, idx):
        self.page_combo.clear()

        shelf_data = self.shelf_combo.itemData(self.shelf_combo.currentIndex())
        if shelf_data is None:
            return
        shelf_item = self.library[shelf_data]
        shelf_content = shelf_item.get("content", [])

        book_data = self.book_combo.itemData(idx)
        if book_data is None:
            return

        if book_data<0 or book_data>=len(shelf_content):
            return
        book_dict = shelf_content[book_data]

        for k, pg in enumerate(book_dict.get("content", [])):
            if "PAGE" in pg:
                disp_page = pg.get("name", pg["PAGE"])
                self.page_combo.addItem(disp_page, k)
            elif "DIVIDER" in pg:
                divp = f"—— {pg['DIVIDER']} ——"
                idxp = self.page_combo.count()
                self.page_combo.addItem(divp, None)
                itp = self.page_combo.model().item(idxp)
                itp.setEnabled(False)





    def get_selection(self):
        """
        Retourne un dict => {"shelf":..., "book":..., "page":..., "data":...} ou None si un divider.
        On récupère .SHELF, .BOOK, .PAGE, et le champ 'data' (chemin vers le fichier YAML).
        """
        shelf_data = self.shelf_combo.itemData(self.shelf_combo.currentIndex())
        if shelf_data is None:
            return None
        shelf_item = self.library[shelf_data]
        shelf_key = shelf_item["SHELF"]

        book_data = self.book_combo.itemData(self.book_combo.currentIndex())
        if book_data is None:
            return None
        shelf_content = shelf_item.get("content", [])
        if book_data < 0 or book_data >= len(shelf_content):
            return None
        book_dict = shelf_content[book_data]
        if "BOOK" not in book_dict:
            return None
        book_key = book_dict["BOOK"]

        page_data = self.page_combo.itemData(self.page_combo.currentIndex())
        if page_data is None:
            return None
        book_content = book_dict.get("content", [])
        if page_data < 0 or page_data >= len(book_content):
            return None
        page_dict = book_content[page_data]
        if "PAGE" not in page_dict:
            return None
        page_key = page_dict["PAGE"]
        data_field = page_dict.get("data", "")  # Récupère le chemin associé (ex: "main/Ag/nk/Johnson.yml")

        return {
            "shelf": shelf_key,
            "book":  book_key,
            "page":  page_key,
            "data":  data_field
        }


##############################################################################
# 4) MaterialRoleWidget : Gère un rôle (mode= None, Custom, Standard, RefrIndex)
##############################################################################

class MaterialRoleWidget(QWidget):
    def __init__(self, role_name, library, standard_list, parent=None):
        super().__init__(parent)
        self.role_name = role_name
        self.library = library
        self.standard_list = standard_list

        layout = QVBoxLayout(self)

        # Mode: None/Custom/Standard/RefractiveIndex
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["None","Custom","Standard","RefractiveIndex"])
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)

        mode_line = QHBoxLayout()
        mode_line.addWidget(QLabel("Mode:"))
        mode_line.addWidget(self.mode_combo)
        layout.addLayout(mode_line)

        # Custom
        self.custom_edit = QLineEdit()
        layout.addWidget(self.custom_edit)

        # Standard
        self.standard_combo = QComboBox()
        self.standard_combo.addItems(self.standard_list)
        layout.addWidget(self.standard_combo)

        # RefractiveIndex
        self.rif_arbo = RefractiveIndexArboWidget(self.library)
        layout.addWidget(self.rif_arbo)

        self.on_mode_changed()

    def on_mode_changed(self, idx=None):
        mode = self.mode_combo.currentText()
        if mode == "None":
            self.custom_edit.setVisible(False)
            self.standard_combo.setVisible(False)
            self.rif_arbo.setVisible(False)
        elif mode == "Custom":
            self.custom_edit.setVisible(True)
            self.standard_combo.setVisible(False)
            self.rif_arbo.setVisible(False)
        elif mode == "Standard":
            self.custom_edit.setVisible(False)
            self.standard_combo.setVisible(True)
            self.rif_arbo.setVisible(False)
        else:
            # RefractiveIndex
            self.custom_edit.setVisible(False)
            self.standard_combo.setVisible(False)
            self.rif_arbo.setVisible(True)


    def get_config(self):
        """
        Retourne ex:
        - {"type": "None"}
        - {"type": "Custom", "expression": "1.5"}
        - {"type": "Standard", "material": "ITO"}
        - {"type": "RefractiveIndex", "shelf": "main", "book": "Al", "page": "Rakic-BB", "data": "main/Al/nk/Rakic-BB.yml"}
        """
        mode = self.mode_combo.currentText()
        if mode == "None":
            return {"type": "None"}
        elif mode == "Custom":
            expr = self.custom_edit.text().strip()
            if not expr:
                expr = "None"
            return {"type": "Custom", "expression": expr}
        elif mode == "Standard":
            mat = self.standard_combo.currentText()
            return {"type": "Standard", "material": mat}
        else:
            # RefractiveIndex
            sel = self.rif_arbo.get_selection()
            if sel is None:
                return {"type": "None"}
            return {
                "type": "RefractiveIndex",
                "shelf": sel["shelf"],
                "book":  sel["book"],
                "page":  sel["page"],
                "data":  sel["data"]  # Enregistrement du chemin associé
            }


##############################################################################
# 5) Fenêtre principale : QTabWidget pour chaque rôle
##############################################################################


class MaterialSelectorTabbed(QMainWindow):
    def __init__(self, roles, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Material Selector - nkexplorer style, multi-rôles")
        # Fenêtre un peu plus compacte
        self.resize(900, 600)

        # Charger la library et la liste standard
        self.library = load_catalog_full(catalog_path)
        self.standard_list = get_standard_materials(json_combined_path, data_dir)

        # Configuration du widget central
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Création d'un onglet par rôle
        self.tab_widget = QTabWidget()
        self.role_widgets = {}
        for role_name in roles:
            w = MaterialRoleWidget(role_name, self.library, self.standard_list)
            self.role_widgets[role_name] = w
            self.tab_widget.addTab(w, role_name)
        main_layout.addWidget(self.tab_widget)

        # Bouton Validate / Quit
        btn_box = QHBoxLayout()
        self.btn_validate = QPushButton("Validate / Quit")
        self.btn_validate.clicked.connect(self.on_validate)
        btn_box.addStretch()
        btn_box.addWidget(self.btn_validate)
        main_layout.addLayout(btn_box)

    def on_validate(self):
        """
        Récupère la configuration pour chaque rôle, la convertit en DataFrame et
        en dict RI_OVERRIDES, affiche la configuration, et l'enregistre dans un fichier JSON
        ("material_config.json") dans le workspace. Ensuite, ferme la fenêtre.
        """
        config_list = []
        ri_overrides = {}  # Pour stocker les configurations RefractiveIndex

        for role_name, widget_role in self.role_widgets.items():
            mat_info = widget_role.get_config()
            config_list.append({"key": role_name, "material": mat_info})
            if mat_info.get("type") == "RefractiveIndex":
                ri_overrides[role_name] = mat_info

        df_config = pd.DataFrame(config_list)

        # Affichage en console
        print("Selected materials configuration:")
        print(df_config)

        # Affichage dans une boîte de dialogue
        from pprint import pformat
        msg = pformat(config_list, indent=2)
        QMessageBox.information(self, "Selection done", msg)

        # Enregistrement dans un fichier JSON dans le workspace
        config_dict = {
            "MATERIALS_CONFIG": config_list,
            "RI_OVERRIDES": ri_overrides
        }
        config_file = os.path.join(workspace_dir, "material_config.json")
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)
        print(f"Configuration saved to {config_file}")

        self.close()


def main():
    from PyQt6.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    # DEFAULT_ROLES doit être défini (par exemple, importé depuis votre module)
    w = MaterialSelectorTabbed(DEFAULT_ROLES)
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()


