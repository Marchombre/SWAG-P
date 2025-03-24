# nkexplorer_widget.py
import os
import re
import yaml
import numpy as np
import matplotlib.pyplot as plt

import ipywidgets as widgets
from IPython.display import display, clear_output

# --- Variables globales pour la structure de la base de données ---
shelf_ids = []
shelf_names = []

book_ids = []
book_names = []

page_ids = []
page_names = []
page_paths = []

wl_n = []
wl_k = []
n = []
k = []
n_defined = []
k_defined = []

# Détermination du chemin courant
current_file_path = os.path.abspath(__file__) if '__file__' in globals() else os.getcwd()
# On suppose ici que la base de données se trouve à deux niveaux au-dessus (adaptable)
db_path = os.path.join(os.path.dirname(os.path.dirname(current_file_path)))
lib_path = os.path.join(db_path, "catalog_nk.yml")

# Chargement du fichier YAML de la bibliothèque
with open(lib_path, "r", encoding="utf-8") as file:
    library = yaml.safe_load(file)

# --- Fonctions utilitaires ---
def html2mathtext(text):
    text = re.sub(r"<sub>(.*?)</sub>", r"$_{\1}$", text)
    return f"{text}"

def stringify(d, indent=0):
    s = ""
    for i, (key, value) in enumerate(d.items()):
        if i > 0:
            s += "\n"
        s += " " * indent + f"{key}:"
        if isinstance(value, dict):
            s += "\n" + stringify(value, indent + 2)
        elif isinstance(value, list) and all(isinstance(ii, dict) for ii in value):
            for item in value:
                s += "\n  - " + stringify(item, indent + 4).lstrip()
        else:
            s += f" {value}"
    return s

# --- Classe de l'interface NK Explorer en ipywidgets ---
class NKExplorerWidget:
    def __init__(self):
        # Widgets pour la sélection de l'étagère et du livre
        self.shelf_dropdown = widgets.Dropdown(
            options=[], description="Shelf:", layout=widgets.Layout(width="300px")
        )
        self.book_dropdown = widgets.Dropdown(
            options=[], description="Book:", layout=widgets.Layout(width="300px")
        )
        
        # Options de tracé
        self.plot_checkbox_n = widgets.Checkbox(value=True, description="n")
        self.plot_checkbox_k = widgets.Checkbox(value=True, description="k")
        self.plot_checkbox_logx = widgets.Checkbox(value=False, description="LogX")
        self.plot_checkbox_logy = widgets.Checkbox(value=False, description="LogY")
        
        # Conteneurs pour la sélection des pages
        self.checkbox_container = widgets.VBox()
        self.radio_container = widgets.VBox()
        self.page_checkboxes = []
        self.page_radios = []
        
        # Figure matplotlib et zone d'affichage du graphique
        self.fig, self.ax = plt.subplots()
        self.plot_out = widgets.Output()
        
        # Widgets HTML pour les onglets Details et About
        self.details_html = widgets.HTML()
        self.about_html = widgets.HTML()
        
        # Organisation en onglets
        self.tab = widgets.Tab()
        data_explorer_tab = widgets.HBox([
            self.checkbox_container,
            widgets.VBox([
                self.plot_out,
                widgets.HBox([self.plot_checkbox_n, self.plot_checkbox_k,
                              self.plot_checkbox_logx, self.plot_checkbox_logy])
            ])
        ])
        details_tab = widgets.HBox([self.radio_container, self.details_html])
        about_tab = self.about_html
        
        self.tab.children = [data_explorer_tab, details_tab, about_tab]
        self.tab.set_title(0, "Data explorer")
        self.tab.set_title(1, "Details")
        self.tab.set_title(2, "About")
        
        # Assemblage de l'interface principale
        self.ui = widgets.VBox([self.shelf_dropdown, self.book_dropdown, self.tab])
        
        # Liaison des événements
        self.shelf_dropdown.observe(self.on_shelf_change, names="value")
        self.book_dropdown.observe(self.on_book_change, names="value")
        self.plot_checkbox_n.observe(self.on_plot_change, names="value")
        self.plot_checkbox_k.observe(self.on_plot_change, names="value")
        self.plot_checkbox_logx.observe(self.on_plot_change, names="value")
        self.plot_checkbox_logy.observe(self.on_plot_change, names="value")
        
        # Initialisation
        self.update_shelf_list()
    
    def update_shelf_list(self):
        global shelf_ids, shelf_names
        shelf_ids = []
        shelf_names = []
        options = []
        for shelf in library:
            if "SHELF" in shelf:
                shelf_ids.append(shelf.get("SHELF"))
                shelf_names.append(shelf.get("name"))
                options.append(shelf.get("name"))
            elif "DIVIDER" in shelf:
                shelf_ids.append("")
                shelf_names.append("")
                options.append("   " + shelf.get("DIVIDER"))
        self.shelf_dropdown.options = options
        # Sélectionne la première option non-indendée
        for opt in options:
            if not opt.startswith("   "):
                self.shelf_dropdown.value = opt
                break
        self.update_book_list()
    
    def update_book_list(self):
        global book_ids, book_names
        shelf_index = self.shelf_dropdown.options.index(self.shelf_dropdown.value)
        shelf = library[shelf_index].get("content")
        if not shelf:
            return
        options = []
        book_ids = []
        book_names = []
        for book in shelf:
            if "BOOK" in book:
                book_ids.append(book.get("BOOK"))
                book_names.append(book.get("name"))
                options.append(re.sub("<[^<]+?>", "", book.get("name")))
            elif "DIVIDER" in book:
                book_ids.append("")
                book_names.append("")
                options.append("   " + book.get("DIVIDER"))
        self.book_dropdown.options = options
        for opt in options:
            if not opt.startswith("   "):
                self.book_dropdown.value = opt
                break
        self.update_page_list()
    
    def update_page_list(self):
        global page_ids, page_names, page_paths
        shelf_index = self.shelf_dropdown.options.index(self.shelf_dropdown.value)
        shelf = library[shelf_index].get("content")
        if not shelf:
            return
        book_index = self.book_dropdown.options.index(self.book_dropdown.value)
        book = shelf[book_index].get("content")
        if not book:
            return
        page_ids = []
        page_names = []
        page_paths = []
        self.page_checkboxes = []
        self.page_radios = []
        checkbox_widgets = []
        radio_widgets = []
        for i, page in enumerate(book):
            if "PAGE" in page:
                page_ids.append(page.get("PAGE"))
                page_names.append(page.get("name"))
                page_paths.append(page.get("data"))
                is_first_enabled = (len(page_ids)==1 and page_ids[0]!="") or (len(page_ids)==2 and page_ids[0]=="")
                cb = widgets.Checkbox(value=is_first_enabled, description=html2mathtext(page.get("name")))
                cb.observe(self.on_plot_change, names="value")
                checkbox_widgets.append(cb)
                self.page_checkboxes.append(cb)
                rb = widgets.ToggleButton(value=is_first_enabled, description=html2mathtext(page.get("name")), layout=widgets.Layout(width="auto"))
                rb.observe(self.on_radio_change, names="value")
                radio_widgets.append(rb)
                self.page_radios.append(rb)
            if "DIVIDER" in page:
                page_ids.append("")
                page_names.append("")
                page_paths.append("")
                cb = widgets.Checkbox(value=False, description=html2mathtext(page.get("DIVIDER")), disabled=True)
                checkbox_widgets.append(cb)
                self.page_checkboxes.append(cb)
                rb = widgets.ToggleButton(value=False, description=html2mathtext(page.get("DIVIDER")), disabled=True, layout=widgets.Layout(width="auto"))
                radio_widgets.append(rb)
                self.page_radios.append(rb)
        self.checkbox_container.children = checkbox_widgets
        self.radio_container.children = radio_widgets
        self.update_data()
        self.on_plot_change(None)
        self.update_details()
        self.update_about()
    
    def update_data(self):
        global wl_n, wl_k, n, k, n_defined, k_defined
        wl_n = []
        wl_k = []
        n = []
        k = []
        n_defined = []
        k_defined = []
        for i in range(len(page_ids)):
            if page_ids[i] == "":
                n_defined.append(False)
                k_defined.append(False)
                wl_n.append(0)
                wl_k.append(0)
                n.append(0)
                k.append(0)
                continue
            data_path = os.path.join(db_path, "data", page_paths[i])
            data_path = os.path.normpath(data_path)
            if os.path.exists(data_path):
                tmp_wl_n = []
                tmp_wl_k = []
                tmp_n = []
                tmp_k = []
                tmp_n_defined = False
                tmp_k_defined = False
                with open(data_path, "r", encoding="utf-8") as file:
                    datafile = yaml.safe_load(file)
                for data in datafile.get("DATA"):
                    datatype = data.get("type").split()
                    if datatype[0] == "tabulated":
                        rows = data.get("data").split("\n")
                        splitrows = [c.split() for c in rows]
                        for s in splitrows:
                            if len(s) > 0:
                                if datatype[1] == "n":
                                    tmp_n_defined = True
                                    tmp_wl_n.append(float(s[0]))
                                    tmp_n.append(float(s[1]))
                                if datatype[1] == "k":
                                    tmp_k_defined = True
                                    tmp_wl_k.append(float(s[0]))
                                    tmp_k.append(float(s[1]))
                                if datatype[1] == "nk":
                                    tmp_n_defined = True
                                    tmp_k_defined = True
                                    tmp_wl_n.append(float(s[0]))
                                    tmp_wl_k.append(float(s[0]))
                                    tmp_n.append(float(s[1]))
                                    tmp_k.append(float(s[2]))
                    elif datatype[0] == "formula":
                        tmp_n_defined = True
                        wavelength_range = np.array(data.get("wavelength_range").split()).astype(float)
                        if wavelength_range[1]/wavelength_range[0] > 20:
                            wl_range = np.logspace(np.log10(wavelength_range[0]), np.log10(wavelength_range[1]), 101)
                        else:
                            wl_range = np.linspace(wavelength_range[0], wavelength_range[1], 101)
                        tmp_wl_n = wl_range
                        coefficients = np.array(data.get("coefficients").split()).astype(float)
                        num_coeff = coefficients.size
                        C1  = coefficients[0] if num_coeff > 0 else 0
                        C2  = coefficients[1] if num_coeff > 1 else 0
                        C3  = coefficients[2] if num_coeff > 2 else 0
                        C4  = coefficients[3] if num_coeff > 3 else 0
                        C5  = coefficients[4] if num_coeff > 4 else 0
                        C6  = coefficients[5] if num_coeff > 5 else 0
                        C7  = coefficients[6] if num_coeff > 6 else 0
                        C8  = coefficients[7] if num_coeff > 7 else 0
                        C9  = coefficients[8] if num_coeff > 8 else 0
                        C10 = coefficients[9] if num_coeff > 9 else 0
                        C11 = coefficients[10] if num_coeff > 10 else 0
                        C12 = coefficients[11] if num_coeff > 11 else 0
                        C13 = coefficients[12] if num_coeff > 12 else 0
                        C14 = coefficients[13] if num_coeff > 13 else 0
                        C15 = coefficients[14] if num_coeff > 14 else 0
                        C16 = coefficients[15] if num_coeff > 15 else 0
                        C17 = coefficients[16] if num_coeff > 16 else 0
                        if datatype[1] == "1":
                            tmp_n = (1 + C1 + C2/(1-(C3/wl_range)**2) + C4/(1-(C5/wl_range)**2) +
                                     C6/(1-(C7/wl_range)**2) + C8/(1-(C9/wl_range)**2) +
                                     C10/(1-(C11/wl_range)**2) + C12/(1-(C13/wl_range)**2) +
                                     C14/(1-(C15/wl_range)**2) + C16/(1-(C17/wl_range)**2))**0.5
                        elif datatype[1] == "2":
                            tmp_n = (1 + C1 + C2/(1-C3/wl_range**2) + C4/(1-C5/wl_range**2) +
                                     C6/(1-C7/wl_range**2) + C8/(1-C9/wl_range**2) +
                                     C10/(1-C11/wl_range**2) + C12/(1-C13/wl_range**2) +
                                     C14/(1-C15/wl_range**2) + C16/(1-C17/wl_range**2))**0.5
                        elif datatype[1] == "3":
                            tmp_n = (C1 + C2*wl_range**C3 + C4*wl_range**C5 +
                                     C6*wl_range**C7 + C8*wl_range**C9 + C10*wl_range**C11 +
                                     C12*wl_range**C13 + C14*wl_range**C15 + C16*wl_range**C17)**0.5
                        elif datatype[1] == "4":
                            tmp_n = (C1 + C2*wl_range**C3/(wl_range**2-C4**C5) +
                                     C6*wl_range**C7/(wl_range**2-C8**C9) +
                                     C10*wl_range**C11 + C12*wl_range**C13 +
                                     C14*wl_range**C15 + C16*wl_range**C17)**0.5
                        elif datatype[1] == "5":
                            tmp_n = C1 + C2*wl_range**C3 + C4*wl_range**C5 + C6*wl_range**C7 + C8*wl_range**C9 + C10*wl_range**C11
                        elif datatype[1] == "6":
                            tmp_n = 1 + C1 + C2/(C3-wl_range**-2) + C4/(C5-wl_range**-2) + C6/(C7-wl_range**-2) + C8/(C9-wl_range**-2) + C10/(C11-wl_range**-2)
                        elif datatype[1] == "7":
                            tmp_n = C1 + C2/(wl_range**2-0.028) + C3/(wl_range**2-0.028)**2 + C4*wl_range**2 + C5*wl_range**4 + C6*wl_range**6
                        elif datatype[1] == "8":
                            tmp = C1 + C2*wl_range**2/(wl_range**2-C3) + C4*wl_range**2
                            tmp_n = ((2*tmp+1)/(1-tmp))**0.5
                        elif datatype[1] == "9":
                            tmp_n = (C1 + C2/(wl_range**2-C3) + C4*(wl_range-C5)/((wl_range-C5)**2 + C6))**0.5
                n_defined.append(tmp_n_defined)
                k_defined.append(tmp_k_defined)
                wl_n.append(tmp_wl_n)
                wl_k.append(tmp_wl_k)
                n.append(tmp_n)
                k.append(tmp_k)
    
    def draw_plot(self):
        with self.plot_out:
            clear_output(wait=True)
            self.ax.clear()
            for i in range(len(page_ids)):
                if self.page_checkboxes[i].value:
                    first_curve_plotted = False
                    if n_defined[i] and self.plot_checkbox_n.value:
                        line, = self.ax.plot(wl_n[i], n[i], label=page_ids[i])
                        first_curve_plotted = True
                    if k_defined[i] and self.plot_checkbox_k.value:
                        if first_curve_plotted:
                            self.ax.plot(wl_k[i], k[i], linestyle='--', color=line.get_color())
                        else:
                            self.ax.plot(wl_k[i], k[i], linestyle='--', label=page_ids[i])
            self.ax.set_xscale('log' if self.plot_checkbox_logx.value else 'linear')
            self.ax.set_yscale('log' if self.plot_checkbox_logy.value else 'linear')
            book_index = self.book_dropdown.options.index(self.book_dropdown.value)
            self.ax.set_title(html2mathtext(book_names[book_index]))
            self.ax.set_xlabel("Wavelength (μm)")
            if self.plot_checkbox_n.value and not self.plot_checkbox_k.value:
                self.ax.set_ylabel("n")
            elif self.plot_checkbox_k.value and not self.plot_checkbox_n.value:
                self.ax.set_ylabel("k")
            else:
                self.ax.set_ylabel("n (solid), k (dashed)")
            self.ax.grid()
            self.ax.legend()
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
            plt.show()
    
    def on_plot_change(self, change):
        self.draw_plot()
    
    def on_shelf_change(self, change):
        self.update_book_list()
    
    def on_book_change(self, change):
        self.update_page_list()
    
    def on_radio_change(self, change):
        if change['new']:
            for rb in self.page_radios:
                if rb is not change['owner']:
                    rb.value = False
            self.update_details()
            self.update_about()
    
    def update_details(self):
        text = ""
        selected_index = None
        for i, rb in enumerate(self.page_radios):
            if rb.value:
                selected_index = i
                break
        if selected_index is None:
            for i, rb in enumerate(self.page_radios):
                if not rb.disabled:
                    rb.value = True
                    selected_index = i
                    break
        if selected_index is not None:
            data_path = os.path.join(db_path, "data", page_paths[selected_index])
            data_path = os.path.normpath(data_path)
            if os.path.exists(data_path):
                with open(data_path, "r", encoding="utf-8") as file:
                    datafile = yaml.safe_load(file)
                ref = datafile.get("REFERENCES", "")
                com = datafile.get("COMMENTS", "")
                dat = ""
                for data in datafile.get("DATA"):
                    datatype = data.get("type").split()
                    dat += "<b>type: " + datatype[0] + " " + datatype[1] + "</b><br>"
                    if datatype[0] == "tabulated":
                        dat += data.get("data").strip().replace("\n", "<br>")
                    if datatype[0] == "formula":
                        dat += "wavelength_range: " + data.get("wavelength_range").strip() + "<br>"
                        dat += "coefficients: " + data.get("coefficients").strip()
                    dat += "<br><br>"
                con = stringify(datafile.get("CONDITIONS", {}))
                pro = stringify(datafile.get("PROPERTIES", {}))
                if con:
                    text += "<h4>CONDITIONS</h4><pre>" + con + "</pre>"
                if pro:
                    text += "<h4>PROPERTIES</h4><pre>" + pro + "</pre>"
                if com:
                    text += "<h4>COMMENTS</h4><p>" + com + "</p>"
                if ref:
                    text += "<h4>REFERENCES</h4><p>" + ref + "</p>"
                if dat:
                    text += "<h4>DATA</h4><pre>" + dat + "</pre>"
            else:
                text += f"<p> Missing file: {data_path} </p>"
        self.details_html.value = text
    
    def update_about(self):
        text = ""
        selected_index = None
        for i, rb in enumerate(self.page_radios):
            if rb.value:
                selected_index = i
                break
        if selected_index is None:
            for i, rb in enumerate(self.page_radios):
                if not rb.disabled:
                    rb.value = True
                    selected_index = i
                    break
        if selected_index is not None:
            data_path = os.path.join(db_path, "data", page_paths[selected_index])
            data_path = os.path.normpath(data_path)
            datadir = os.path.dirname(page_paths[selected_index])
            dir_1up = os.path.dirname(datadir)
            dir_2up = os.path.dirname(dir_1up)
            about_path1 = os.path.join(db_path, "data", dir_1up, "about.yml")
            about_path2 = os.path.join(db_path, "data", dir_2up, "about.yml")
            about1 = ''
            names1 = []
            links1 = []
            about2 = ''
            names2 = []
            links2 = []
            if os.path.exists(about_path1):
                with open(about_path1, "r", encoding="utf-8") as file:
                    aboutfile = yaml.safe_load(file)
                raw_read = aboutfile.get("NAMES", [[{}]])
                names1 = [str(item) for item in raw_read]
                about1 = aboutfile.get("ABOUT", {})
                raw_links = aboutfile.get("LINKS", [])
                for link in raw_links:
                    if 'url' in link and 'text' in link:
                        links1.append(f'<a href="{link["url"]}">{link["text"]}</a>')
            if os.path.exists(about_path2):
                with open(about_path2, "r", encoding="utf-8") as file:
                    aboutfile = yaml.safe_load(file)
                raw_read = aboutfile.get("NAMES", [[{}]])
                names2 = [str(item) for item in raw_read]
                about2 = aboutfile.get("ABOUT", {})
                raw_links = aboutfile.get("LINKS", [])
                for link in raw_links:
                    if 'url' in link and 'text' in link:
                        links2.append(f'<a href="{link["url"]}">{link["text"]}</a>')
            if about1 or names1 or links1:
                text += '<h3>About'
                if names1:
                    text += f' {names1[0]}'
                text += '</h3>'
                text += '<div style="margin:0 10px 10px 10px;">'
                if about1:
                    text += f'<p>{about1}</p>'
                if names1 and len(names1) > 1:
                    text += f'<h4>Other names and variants of {names1[0]}</h4><ul>'
                    for name in names1[1:]:
                        text += f'<li>{name}</li>'
                    text += '</ul>'
                if links1:
                    text += '<h4>Links</h4><ul>'
                    for link in links1:
                        text += f'<li>{link}</li>'
                    text += '</ul>'
                text += '</div>'
            if about2 or names2 or links2:
                text += '<h3>About'
                if names2:
                    text += f' {names2[0]}'
                text += '</h3>'
                text += '<div style="margin:0 10px 10px 10px;">'
                if about2:
                    text += f'<p>{about2}</p>'
                if names2 and len(names2) > 1:
                    text += f'<h4>Other names and variants of {names2[0]}</h4><ul>'
                    for name in names2[1:]:
                        text += f'<li>{name}</li>'
                    text += '</ul>'
                if links2:
                    text += '<h4>Links</h4><ul>'
                    for link in links2:
                        text += f'<li>{link}</li>'
                    text += '</ul>'
                text += '</div>'
        self.about_html.value = text
    
    def display(self):
        display(self.ui)
