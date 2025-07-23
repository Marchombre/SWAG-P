from gap_plasmon_2d import paths
#!/usr/bin/env python3
# geometry_settings.py

import os, json
import ipywidgets as widgets
from IPython.display import clear_output, display
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.axes import Axes
import numpy as np

# Configuration par défaut (les valeurs réelles utilisées pour les calculs de réflectance restent inchangées)
geometry_config = {
    "thick_super": 200,        # Épaisseur réelle du Superstrate
    "thick_reso": 30,          # Hauteur réelle du Nanocube
    "width_reso": 30,          # Largeur réelle du Nanocube
    "thick_gap": 3,            # Épaisseur réelle du Gap (polymère)
    "thick_mol": 0.5,          # Épaisseur réelle de la Molécule
    "thick_func": 1,           # Épaisseur réelle de la Fonctionnalisation
    "thick_diel": 1,           # Épaisseur réelle du Polymère
    "thick_metalliclayer": 10, # Épaisseur réelle du Métallique
    "thick_XIAOYI": 0,         # Épaisseur réelle du XIAOYI
    "thick_accroche": 1,       # Épaisseur réelle de l'Accroche
    "thick_sub": 200,          # Épaisseur réelle du Substrate
    "period": 100.2153         # Cellule RCWA (carrée)
}

geometry_limits = {
    "thick_super": (0, 2000),
    "thick_reso": (0, 2000),
    "width_reso": (0, 2000),
    "thick_gap": (0, 2000),
    "thick_mol": (0, 2000),
    "thick_func": (0, 2000),
    "thick_diel": (0, 2000),
    "thick_metalliclayer": (0, 2000),
    "thick_XIAOYI": (0, 2000),
    "thick_accroche": (0, 2000),
    "thick_sub": (0, 2000),
    "period": (50, 300)
}

def displayed_thickness(t):
    """
    Retourne 1/10 de l'épaisseur réelle pour l'affichage du Substrate et du Superstrate.
    """
    return t / 10


def _adjust_sub_super(d_sub, d_sup, p, *, min_central_ratio=0.70):
    """
    Réduit proportionnellement Substrate & Superstrate quand ils prennent
    trop de place, de sorte que la « zone centrale » garde au moins
    `min_central_ratio · p` de hauteur.
    """
    max_subsuper = p * (1.0 - min_central_ratio)
    total = d_sub + d_sup
    if total <= 0 or total <= max_subsuper:         # rien à faire
        return d_sub, d_sup
    k = max_subsuper / total                       # facteur < 1
    return d_sub * k, d_sup * k


def _rescale_with_min(thicknesses, H_avail, h_min):
    """
    Ramène la liste *thicknesses* dans la hauteur disponible *H_avail*
    en imposant une hauteur mini *h_min* à chaque couche non nulle.
    """
    h = np.asarray(thicknesses, float)
    if h.sum() == 0:
        return np.zeros_like(h)

    # 1) mise à l’échelle linéaire
    h *= H_avail / h.sum()

    # 2) contrainte h ≥ h_min
    small  = h < h_min
    if not small.any():
        return h

    deficit = (h_min - h[small]).sum()
    large   = ~small
    surplus = (h[large] - h_min).sum()

    if surplus <= 0:                     # cas extrême : tout est min
        h[:] = h_min
        return h * (H_avail / h.sum())

    factor     = 1 - deficit / surplus
    h[large]   = h_min + (h[large] - h_min) * factor
    h[small]   = h_min
    return h



def get_geometry_save_path():
    """
    Retourne le chemin vers le fichier JSON utilisé pour stocker les configurations géométriques.
    """
    module_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    workspace_dir = os.path.dirname(module_dir)
    CONFIG_DIR = os.path.join(str(paths.CONFIGS_DIR))
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR)
    return os.path.join(CONFIG_DIR, "geometry_configurations.json")

GEOMETRY_CONFIGS = []

def load_geometry_configs():
    """
    Charge les configurations géométriques sauvegardées depuis le fichier JSON, s'il existe.
    """
    global GEOMETRY_CONFIGS
    save_path = get_geometry_save_path()
    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            GEOMETRY_CONFIGS = data.get("ALL_GEOMETRY_CONFIGS", [])
    else:
        GEOMETRY_CONFIGS = []

def save_geometry_configs():
    """
    Sauvegarde l'ensemble des configurations géométriques actuelles dans le fichier JSON.
    """
    save_path = get_geometry_save_path()
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({"ALL_GEOMETRY_CONFIGS": GEOMETRY_CONFIGS}, f, indent=2)
    return save_path

def draw_layer(ax, x, y, w, h, color, label, hatch=None):
    """
    Dessine une couche rectangulaire sur l'objet Axes 'ax',
    avec un libellé optionnel (affiché au centre) et éventuellement un motif de hachures.
    Si h <= 0, la couche n'est pas dessinée.
    """
    if h <= 0:
        return
    rect = patches.Rectangle((x, y), w, h, edgecolor=None, facecolor=color, hatch=hatch)
    ax.add_patch(rect)
    if label and h > 0:
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=9)




def create_geometry_widget():
    """
    Crée une interface basée sur ipywidgets pour ajuster les paramètres de la géométrie,
    sauvegarder/charger les configurations, et afficher le schéma.
    """
    ordered_params = [
        ("thick_super", "Superstrate"),
        ("thick_reso", "Nanocube height"),
        ("width_reso", "Nanocube width"),
        ("thick_gap", "Gap (polymer)"),
        ("thick_mol", "Molecule"),
        ("thick_func", "Functionalisation"),
        ("thick_diel", "Dielectric"),
        ("thick_metalliclayer", "Metallic"),
        ("thick_XIAOYI", "XIAOYI"),
        ("thick_accroche", "Accroche"),
        ("thick_sub", "Substrate"),
        ("period", "Period")
    ]
    
    geometry_sliders = {}
    slider_widgets = []
    
    for key, label in ordered_params:
        default = geometry_config.get(key, 0)
        min_val, max_val = geometry_limits.get(key, (0, 200))
        
        if key == "period":
            description_str = label + ":"
        else:
            description_str = label + " (nm):"
        # Création d'un slider avec un FloatText lié pour chaque paramètre
        slider = widgets.FloatSlider(
            value=default, min=min_val, max=max_val, step=0.1,
            description=description_str,
            continuous_update=False,
            layout=widgets.Layout(width='350px'),
            style={'description_width': '180px'}
        )
        float_text = widgets.FloatText(value=default, layout=widgets.Layout(width='100px'))
        widgets.jslink((slider, 'value'), (float_text, 'value'))
        
        # Empêche les valeurs négatives
        def validate_positive(change):
            if change['new'] < 0:
                change['owner'].value = 0
        float_text.observe(validate_positive, names='value')
        
        geometry_sliders[key] = slider
        slider_widgets.append(widgets.HBox([slider, float_text]))
    
    # 3) === AJOUT DYNAMIQUE : couches homogènes “homo_XXX” ===
    extra_layer_keys = []  # stocke les clés "homo_nom"
    layers_box = widgets.VBox(
        [], layout=widgets.Layout(
            border='1px dashed lightgray',
            padding='5px', margin='10px 0', width='100%'
        )
    )
    button_add_layer = widgets.Button(
        description="Add layer", button_style='info',
        layout=widgets.Layout(width='150px', margin='10px 0 0 0')
    )


    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    extra_layer_colors = {}      # mapping layer_key -> couleur
    next_color_idx = 0           # index dans color_cycle


    # compteur pour générer un nom par défaut unique
    layer_counter = {'count': 0}


    def on_add_layer(_):
        nonlocal next_color_idx
        # incrémente et construit un nom par défaut
        layer_counter['count'] += 1
        default_name = f"layer{layer_counter['count']}"


        # Création du Text pour le nom de layer et du slider pour l'épaisseur
        name_w = widgets.Text(
            value=default_name, placeholder='Layer name',
            layout=widgets.Layout(width='120px')
        )
        sl = widgets.FloatSlider(
            value=1.0, min=0, max=50, step=0.1,
            description="thick (nm):", continuous_update=False,
            layout=widgets.Layout(width='250px'),
            style={'description_width':'100px'}
        )
        ft = widgets.FloatText(value=1.0, layout=widgets.Layout(width='80px'))
        widgets.jslink((sl, 'value'), (ft, 'value'))

        # Bouton de suppression
        btn_del = widgets.Button(description="✕", layout=widgets.Layout(width='30px'))
        container = widgets.HBox([name_w, sl, ft, btn_del])


        # 1) Gestion du renommage du Text en clé thick_homo_<name>
        def _rename(change):
            nonlocal next_color_idx
            old_key = getattr(container, 'layer_key', None)
            base_key = f"thick_homo_{change['new']}"
            # générer un key unique (comme dans votre code)
            candidate = base_key
            i = 1
            while candidate in geometry_sliders and candidate != old_key:
                candidate = f"{base_key}_{i}"
                i += 1

            # on enlève l’ancienne si besoin
            if old_key in geometry_sliders:
                geometry_sliders.pop(old_key); extra_layer_keys.remove(old_key)
                extra_layer_colors.pop(old_key, None)

            # on attribue la couleur si c’est un nouveau layer
            if candidate not in extra_layer_colors:
                extra_layer_colors[candidate] = color_cycle[next_color_idx % len(color_cycle)]
                next_color_idx += 1

            # on enregistre la clé
            geometry_sliders[candidate] = sl
            extra_layer_keys.append(candidate)
            container.layer_key = candidate

            draw_structure()

        name_w.observe(_rename, names='value')
        # déclencher _rename une première fois avec la valeur par défaut
        _rename({'new': default_name})




        # 2) Gestion de la suppression
        def _del(_):
            layers_box.children = tuple(c for c in layers_box.children if c is not container)
            key = getattr(container, 'layer_key', None)
            if key in geometry_sliders:
                geometry_sliders.pop(key)
                extra_layer_keys.remove(key)
            draw_structure()

        btn_del.on_click(_del)
        sl.observe(lambda *_: draw_structure(), names='value')

        layers_box.children = tuple(list(layers_box.children) + [container])

    button_add_layer.on_click(on_add_layer)

    slider_widgets.append(widgets.HTML("<b>Homogène Layers</b>"))
    slider_widgets.append(button_add_layer)
    slider_widgets.append(layers_box)




    # 4) Widgets de configuration existants (Add/Save/Load/Update/Delete) …
    config_name_text = widgets.Text(
        value='', placeholder='Configuration Name', description='Config Name :',
        layout=widgets.Layout(width='350px'), style={'description_width':'180px'}
    )
    compartment_text = widgets.Text(
        value='', placeholder='Nom du compartiment', description='Compartiment :',
        layout=widgets.Layout(width='350px'), style={'description_width':'180px'}
    )
    button_add    = widgets.Button(description="Add Config",    layout=widgets.Layout(width='150px'))
    button_save   = widgets.Button(description="Save & Quit", button_style='success',
                                  layout=widgets.Layout(width='200px'))
    config_dropdown = widgets.Dropdown(options=[], description="Saved Configs :",
                                       layout=widgets.Layout(width='350px'),
                                       style={'description_width':'180px'})
    button_load   = widgets.Button(description="Load",    layout=widgets.Layout(width='100px'))
    button_update = widgets.Button(description="Update",  layout=widgets.Layout(width='120px'))
    button_delete = widgets.Button(description="Delete",  button_style='danger',
                                  layout=widgets.Layout(width='120px'))

    compartment_filter = widgets.Dropdown(
        options=["Tous"], value="Tous", description="Filtrer Compart.:",
        layout=widgets.Layout(width='350px'),
        style={'description_width':'180px'}
    )
    output_area = widgets.Output(layout=widgets.Layout(padding='10px'))

    load_geometry_configs()

    def update_dropdown_options(change=None):
        filtre = compartment_filter.value
        if filtre == "Tous":
            filtered = GEOMETRY_CONFIGS
        else:
            filtered = [c for c in GEOMETRY_CONFIGS
                        if c.get("compartment","Défaut")==filtre]
        opts = [(f"{c['config_name']} ({c.get('compartment','Défaut')})", c)
                for c in filtered] or [("None", None)]
        config_dropdown.options = opts
        comps = sorted({c.get("compartment","Défaut") for c in GEOMETRY_CONFIGS})
        compartment_filter.options = ["Tous"] + comps

    update_dropdown_options()
    compartment_filter.observe(update_dropdown_options, names='value')


    # Mise à jour initiale du dropdown
    update_dropdown_options()

    # Observer le changement de sélection du compartiment pour rafraîchir la liste
    compartment_filter.observe(update_dropdown_options, names='value')
    
    def add_geometry_config(_):
        # 1) On récupère tous les sliders
        for k, s in geometry_sliders.items():
            geometry_config[k] = s.value

        # 2) On construit le nom de base
        raw_name = config_name_text.value.strip() or f"Geometry_{len(GEOMETRY_CONFIGS)+1}"
        existing = {c["config_name"] for c in GEOMETRY_CONFIGS}

        # 3) Si le nom existe déjà, on lui ajoute "_1", "_2", …
        name = raw_name
        if name in existing:
            i = 1
            while f"{raw_name}_{i}" in existing:
                i += 1
            name = f"{raw_name}_{i}"

        comp = compartment_text.value.strip() or "Défaut"

        # définition manuelle de l’ordre des clés
        before = [
            "thick_super","thick_reso","width_reso",
            "thick_gap","thick_mol","thick_func",
            "thick_diel","thick_metalliclayer"
        ]
        after = ["thick_XIAOYI", "thick_accroche","thick_sub","period"]
        # on reconstruit la géométrie dans l’ordre voulu :
        ordered_geom = {}
        for k in before:
            ordered_geom[k] = geometry_config[k]
        for k in extra_layer_keys:          # vos homo_XXX, dans l’ordre d’ajout
            ordered_geom[k] = geometry_sliders[k].value
        for k in after:
            ordered_geom[k] = geometry_config[k]

        new = {
            "config_name": name,
            "compartment": comp,
            "geometry": ordered_geom
        }

        GEOMETRY_CONFIGS.append(new)
        update_dropdown_options()
        save_geometry_configs()
        with output_area:
            clear_output(); print("Configuration ajoutée :", new)
        config_name_text.value = ''; compartment_text.value = ''




    def save_geometry_configs_btn(_):
        p = save_geometry_configs()
        with output_area:
            clear_output(); print(f"Configurations saved in {p}")



    def load_config(_):
        sel = config_dropdown.value
        if sel is None:
            with output_area:
                clear_output(); print("Aucune config sélectionnée.")
            return

        # 0) Vider les anciennes couches homo_* et l'UI
        for key in list(extra_layer_keys):
            geometry_sliders.pop(key, None)
        extra_layer_keys.clear()
        layers_box.children = ()

        # 1) Extraire **dans l’ordre** les clés thick_homo_* du JSON
        homo_keys_in_json = [
            k for k in sel["geometry"].keys()
            if k.startswith("thick_homo_") and sel["geometry"][k] > 0
        ]

        # 2) Pour chaque clé, recréer le slider AU BON ENDROIT
        for thick_key in homo_keys_in_json:
            thickness = sel["geometry"][thick_key]

            # 2.1) Ajouter un nouveau container (Text + slider)
            on_add_layer(None)
            container = layers_box.children[-1]

            # 2.2) Donner au Text le nom de base pour déclencher _rename()
            base_name = thick_key[len("thick_homo_"):]
            container.children[0].value = base_name

            # 2.3) Récupérer la clé **réelle** (qui peut avoir été uniquifiée)
            actual_key = container.layer_key

            # 2.4) Régler l'épaisseur
            geometry_sliders[actual_key].value = thickness

        # 3) Mettre à jour tous les sliders “classiques”
        for key, value in sel["geometry"].items():
            if key in geometry_sliders:
                geometry_sliders[key].value = value

        # 4) Remplir les champs de texte et afficher un message
        config_name_text.value = sel["config_name"]
        compartment_text.value = sel.get("compartment", "Défaut")
        with output_area:
            clear_output(); print(f"Configuration '{sel['config_name']}' chargée.")

        # 5) Redessiner la structure dans le bon ordre
        draw_structure()


            

    def update_config(_):
        sel = config_dropdown.value
        if sel is None:
            with output_area:
                clear_output(); print("Aucune config sélectionnée.")
            return

        # 1) Récupère toutes les valeurs depuis les sliders
        for k, s in geometry_sliders.items():
            geometry_config[k] = s.value

        # 2) Reconstruit le dict geometry dans l’ordre voulu
        before = [
            "thick_super","thick_reso","width_reso",
            "thick_gap","thick_mol","thick_func",
            "thick_diel","thick_metalliclayer"
        ]
        after = ["thick_XIAOYI", "thick_accroche","thick_sub","period"]

        ordered_geom = {}
        # insère d’abord les clés « before »
        for k in before:
            ordered_geom[k] = geometry_sliders[k].value

        # puis tes couches dynamiques homo layers, dans l’ordre stocké
        for k in extra_layer_keys:
            ordered_geom[k] = geometry_sliders[k].value

        # enfin les clés « after »
        for k in after:
            ordered_geom[k] = geometry_sliders[k].value

        # 3) Remplace entièrement sel["geometry"]
        sel["geometry"] = ordered_geom

        # 4) Gère le renommage éventuel (comme toi)
        old_label = next((lbl for lbl,obj in config_dropdown.options if obj is sel),
                        sel["config_name"])
        old_name = old_label.split(' (')[0]
        new_name = config_name_text.value.strip() or old_name
        sel["config_name"] = new_name
        sel["compartment"] = compartment_text.value.strip() or "Défaut"

        # 5) Patch convergence_results.json si on change de nom
        WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        conv_json = os.path.join(WORKSPACE_DIR, "Convergence", "convergence_results.json")
        if new_name != old_name and os.path.exists(conv_json):
            with open(conv_json, "r", encoding="utf-8") as f:
                master = json.load(f)
            cfgs = master.get("configs", {})
            if old_name in cfgs:
                cfgs[new_name] = cfgs.pop(old_name)
                with open(conv_json, "w", encoding="utf-8") as f:
                    json.dump(master, f, indent=2)
                with output_area:
                    print(f"[PATCH] convergence_results.json : {old_name} → {new_name}")

        # 6) Sauvegarde et remise à jour de l’UI
        save_geometry_configs()
        update_dropdown_options()
        with output_area:
            clear_output(); print(f"Configuration mise à jour : {sel}")


    def delete_config(_):
        sel = config_dropdown.value
        if sel is None:
            with output_area:
                clear_output(); print("Aucune config sélectionnée.")
            return
        global GEOMETRY_CONFIGS
        GEOMETRY_CONFIGS = [c for c in GEOMETRY_CONFIGS
                            if c["config_name"]!=sel["config_name"]]
        update_dropdown_options(); save_geometry_configs()
        with output_area:
            clear_output(); print(f"Configuration '{sel['config_name']}' supprimée.")

    button_add.on_click(add_geometry_config)
    button_save.on_click(save_geometry_configs_btn)
    button_load.on_click(load_config)
    button_update.on_click(update_config)
    button_delete.on_click(delete_config)

    # 5) Prépare label & zone dessin
    case_label_widget = widgets.Label(value="")
    # fig_output sans scroll, et taille auto pour pouvoir être centré
    fig_output = widgets.Output(layout=widgets.Layout(
        min_height='0px',      # pour que la sortie puisse réduire sa hauteur
        overflow='hidden',     # supprime les scrollbars
        padding='10px',
        width='auto'           # largeur auto pour ne pas prendre tout l'espace
    ))


    # 6) Fonction de dessin, appelée à chaque modif
    def draw_structure(_=None):
        with fig_output:
            clear_output(wait=True)
            # lecture des épaisseurs
            p        = geometry_sliders["period"].value
            t_super  = geometry_sliders["thick_super"].value
            t_reso   = geometry_sliders["thick_reso"].value
            w_reso   = geometry_sliders["width_reso"].value
            t_gap    = geometry_sliders["thick_gap"].value
            t_mol    = geometry_sliders["thick_mol"].value
            t_func   = geometry_sliders["thick_func"].value
            t_diel   = geometry_sliders["thick_diel"].value
            t_metal  = geometry_sliders["thick_metalliclayer"].value
            t_XIAOYI = geometry_sliders["thick_XIAOYI"].value
            t_acc    = geometry_sliders["thick_accroche"].value
            t_sub    = geometry_sliders["thick_sub"].value

            # dynamiques
            t_extras = [geometry_sliders[k].value for k in extra_layer_keys]

            # ---------- conversion Sub / Super + liste centrale -----------------
            disp_sub_raw   = displayed_thickness(t_sub)
            disp_super_raw = displayed_thickness(t_super)
            disp_sub, disp_super = _adjust_sub_super(
                disp_sub_raw, disp_super_raw, p, min_central_ratio=0.70
            )

            # ordre physique complet (on suit le JSON) ---------------------------
            central_pairs = [
                ("thick_accroche",       t_acc),
                ("thick_XIAOYI",         t_XIAOYI),
                *[(k, geometry_sliders[k].value) for k in extra_layer_keys],  # homo_*
                ("thick_metalliclayer",  t_metal),
                ("thick_gap",            t_gap),
                ("thick_reso",           t_reso),
                ("thick_diel",           t_diel),
                ("thick_func",           t_func),
                ("thick_mol",            t_mol),
            ]
            central_pairs = [(k, v) for k, v in central_pairs if v > 0]        # on vire les 0 nm
            central_keys, central_real = zip(*central_pairs) if central_pairs else ([], [])

            H_avail = p - (disp_sub + disp_super)
            h_min   = max(0.5, 0.02 * H_avail)         # ≥ 0,5 nm ou 2 % de la zone
            disp_all = _rescale_with_min(central_real, H_avail, h_min)
            disp_map = dict(zip(central_keys, disp_all))

            # ---------------- redistribution explicite --------------------------
            disp_acc        = disp_map.get("thick_accroche",      0.0)
            disp_x          = disp_map.get("thick_XIAOYI",        0.0)
            disp_metal      = disp_map.get("thick_metalliclayer", 0.0)
            disp_gap        = disp_map.get("thick_gap",           0.0)
            disp_reso       = disp_map.get("thick_reso",          0.0)
            disp_dielectric = disp_map.get("thick_diel",          0.0)
            disp_func       = disp_map.get("thick_func",          0.0)
            disp_mol        = disp_map.get("thick_mol",           0.0)
            disp_extras     = [disp_map[k] for k in extra_layer_keys]

            # Case A/B
            case_str = "Case A" if (t_diel+t_func+t_mol)<t_gap else "Case B"
            case_label_widget.value = f"Case: {case_str}"

            # ------------------------------------------------------------------
            #  Création de la figure et calcul de la largeur affichée du cube
            # ------------------------------------------------------------------
            fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
            ax.set_title(f"Schematics – {case_label_widget.value}", fontsize=10, pad=5)

            # — largeur affichée pour rendre le nanocube carré —
            if t_reso > 0:
                scale_h = disp_reso / t_reso          # facteur d’échelle vertical appliqué au cube
                w_reso_disp = w_reso * scale_h        # largeur affichée
            else:
                w_reso_disp = w_reso                  # cas « cube » absent

            central_x = (p - w_reso_disp) / 2         # position x du cube centrée
            lat_width = central_x                     # largeur des zones latérales

            # ------------------------------------------------------------------
            #  Substrate – Accroche – XIAOYI – couches homo_* déjà dans y_cursor
            # ------------------------------------------------------------------
            bande = min(0.05 * p, disp_sub, disp_super)
            draw_layer(ax, 0, 0, p, disp_sub, "brown", "Substrate")
            draw_layer(ax, 0, 0, p, bande, "none", "", hatch='///')

            y = disp_sub
            draw_layer(ax, 0, y, p, disp_acc, "orange", "Accroche"); y += disp_acc
            draw_layer(ax, 0, y, p, disp_x, "purple", "XIAOYI");     y += disp_x

            for key, h in zip(extra_layer_keys, disp_extras):
                color = extra_layer_colors.get(key, "#888")
                draw_layer(ax, 0, y, p, h, color, key.replace("thick_homo_", ""))
                y += h

            # ------------------------------------------------------------------
            #  Metallic layer, Gap, Nanocube
            # ------------------------------------------------------------------
            draw_layer(ax, 0, y, p, disp_metal, "gold", "Metallic layer")
            y_metal_top = y + disp_metal
            draw_layer(ax, central_x, y_metal_top,           w_reso_disp, disp_gap,  "lightgreen", "Gap")
            draw_layer(ax, central_x, y_metal_top + disp_gap, w_reso_disp, disp_reso, "silver",     "Nanocube")
            y_cube_top = y_metal_top + disp_gap + disp_reso

            # ------------------------------------------------------------------
            #  Parois latérales
            # ------------------------------------------------------------------
            y_lat = y_metal_top
            draw_layer(ax, 0,              y_lat, lat_width, disp_dielectric, "green",  "Photopolymer")
            draw_layer(ax, central_x+w_reso_disp, y_lat, lat_width, disp_dielectric, "green",  "")
            y_lat += disp_dielectric

            draw_layer(ax, 0,              y_lat, lat_width, disp_func, "pink",  "Functionalisation")
            draw_layer(ax, central_x+w_reso_disp, y_lat, lat_width, disp_func, "pink",  "")
            y_lat += disp_func

            draw_layer(ax, 0,              y_lat, lat_width, disp_mol, "violet", "Molecule")
            draw_layer(ax, central_x+w_reso_disp, y_lat, lat_width, disp_mol, "violet", "")
            y_lat += disp_mol

            # éventuel remplissage
            if y_cube_top > y_lat:
                h_fill = y_cube_top - y_lat
                draw_layer(ax, 0,              y_lat, lat_width, h_fill, "lightblue", "")
                draw_layer(ax, central_x+w_reso_disp, y_lat, lat_width, h_fill, "lightblue", "")

            # ------------------------------------------------------------------
            #  Superstrate
            # ------------------------------------------------------------------
            draw_layer(ax, 0, y_cube_top, p, p - y_cube_top, "lightblue", "Superstrate")
            draw_layer(ax, 0, p - bande, p, bande, "none", "", hatch='///')

            ax.set_xlim(0, p)
            ax.set_ylim(0, p)
            ax.set_aspect('equal', adjustable='box')
            ax.axis('off')
            plt.show()


    # 7) Observers & première trace
    for s in geometry_sliders.values():
        s.observe(draw_structure, names='value')
    draw_structure()

    # 8) Assemblage final
    config_controls = widgets.VBox([
        config_name_text, compartment_text,
        widgets.HBox([button_add, button_save]),
        compartment_filter,
        widgets.HBox([config_dropdown, button_load, button_update, button_delete]),
        output_area
    ])
    left_panel  = widgets.VBox(
        slider_widgets + [config_controls],
        layout=widgets.Layout(width='600px')
    )

    # → Remplacement de right_panel pour centrer la figure
    right_panel = widgets.VBox(
        [fig_output],
        layout=widgets.Layout(
            flex='1',            # prend tout l'espace restant
            min_width='0px',     # permet la réduction correcte
            align_items='center',    # centre HORIZONTALEMENT fig_output
            justify_content='center' # (optionnel) centre VERTICALEMENT si besoin
        )
    )

    main_ui = widgets.HBox(
        [left_panel, right_panel],
        layout=widgets.Layout(
            width='100%',
            height='auto'
        )
    )



    # expose la liste des couches homogènes dynamiques
    main_ui.extra_layer_keys = extra_layer_keys
    # expose aussi vos sliders si jamais besoin
    main_ui.geometry_sliders  = geometry_sliders



    return main_ui

geometry_widget = create_geometry_widget()
