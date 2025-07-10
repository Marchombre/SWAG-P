from gap_plasmon_2d import paths
import ipywidgets as widgets
from IPython.display import clear_output
import os
import json

from gap_plasmon_2d.utils.file_watchers import start_watcher


def load_json_config(file_name):
    """
    Charge un fichier JSON situé dans CONFIGURATIONS_dir.
    """
    module_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(module_dir)
    CONFIGURATIONS_dir = os.path.join(str(paths.CONFIGS_DIR))
    path = os.path.join(CONFIGURATIONS_dir, file_name)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_geometry_material_widget():
    """
    Widget pour créer des paires (géométrie, matériau), les combiner,
    les sauvegarder, et supprimer des configurations sauvées.
    """

    # ─── (1) Chargement initial ───────────────────────────────────────
    geom_data = load_json_config("geometry_configurations.json") \
                    .get("ALL_GEOMETRY_CONFIGS", [])
    mat_data  = load_json_config("material_config.json") \
                    .get("ALL_CONFIGS", [])

    module_dir         = os.path.dirname(os.path.abspath(__file__))
    workspace_dir      = os.path.dirname(module_dir)
    CONFIGURATIONS_dir = os.path.join(str(paths.CONFIGS_DIR))
    combos_file        = os.path.join(CONFIGURATIONS_dir, "geom_mat_combinations.json")
    
    row_widgets  = []  # stocke {'geom':…, 'mat':…, 'container':…}
    output_area  = widgets.Output(layout=widgets.Layout(padding="10px"))

    # ─── (2) Fonction de rechargement ─────────────────────────────────
    def _reload_config():
        nonlocal geom_data, mat_data, row_widgets, delete_selector

        # 2.1) reload JSON sources
        geom_data = load_json_config("geometry_configurations.json") \
                        .get("ALL_GEOMETRY_CONFIGS", [])
        mat_data  = load_json_config("material_config.json") \
                        .get("ALL_CONFIGS", [])

        # 2.2) ne ré-affecte que si les options ont réellement changé
        geom_opts = [(c["config_name"], c) for c in geom_data] or [("None", None)]
        mat_opts  = [(c["config_name"], c) for c in mat_data]  or [("None", None)]

        # initialise les caches au premier appel
        if not hasattr(_reload_config, "last_geom_opts"):
            _reload_config.last_geom_opts = None
            _reload_config.last_mat_opts  = None

        # compare avant de toucher les widgets
        if geom_opts != _reload_config.last_geom_opts or mat_opts != _reload_config.last_mat_opts:
            for it in row_widgets:
                it["geom"].options = geom_opts
                if it["geom"].value not in [opt[1] for opt in geom_opts]:
                    it["geom"].value = geom_opts[0][1]
                it["mat"].options = mat_opts
                if it["mat"].value not in [opt[1] for opt in mat_opts]:
                    it["mat"].value = mat_opts[0][1]
            # met à jour les caches
            _reload_config.last_geom_opts = geom_opts
            _reload_config.last_mat_opts  = mat_opts

        # 2.3) reload liste des configs combinées pour suppression
        try:
            saved = json.load(open(combos_file, "r", encoding="utf-8")) \
                        .get("ALL_COMBINED_CONFIGS", [])
        except (FileNotFoundError, json.JSONDecodeError):
            saved = []
        names = [c["config_name"] for c in saved]
        delete_selector.options = names
        # préserver les sélections encore valides
        delete_selector.value = tuple(v for v in delete_selector.value if v in names)

    # ─── (3) Watchdog ───────────────────────────────────────────────────
    _watcher = start_watcher(
        path=CONFIGURATIONS_dir,
        callback=_reload_config,
        extensions=[".json"],
        recursive=False
    )

    # ─── (4) Helpers pour Dropdown géométrie / matériau ───────────────
    def make_geom_dd():
        opts = [(c["config_name"], c) for c in geom_data] or [("None", None)]
        return widgets.Dropdown(
            options=opts, value=opts[0][1],
            description="Geometry:",
            layout=widgets.Layout(width="250px"),
            style={"description_width": "80px"}
        )

    def make_mat_dd():
        opts = [(c["config_name"], c) for c in mat_data] or [("None", None)]
        return widgets.Dropdown(
            options=opts, value=opts[0][1],
            description="Material:",
            layout=widgets.Layout(width="250px"),
            style={"description_width": "80px"}
        )

    # ─── (5) Container de lignes ───────────────────────────────────────
    rows_container = widgets.VBox([], layout=widgets.Layout(margin="10px 0"))

    def update_rows():
        rows_container.children = [it["container"] for it in row_widgets]

    def add_row(_=None):
        geom_dd = make_geom_dd()
        mat_dd  = make_mat_dd()
        del_btn = widgets.Button(
            description="Delete", button_style="danger",
            layout=widgets.Layout(width="80px")
        )
        container = widgets.HBox(
            [geom_dd, mat_dd, del_btn],
            layout=widgets.Layout(margin="5px 0", align_items="center")
        )
        row_widgets.append({"geom": geom_dd, "mat": mat_dd, "container": container})
        update_rows()

        def on_delete(_):
            row_widgets.remove(next(it for it in row_widgets if it["container"] is container))
            update_rows()
        del_btn.on_click(on_delete)

    # création de la première ligne
    add_row()

    # ─── (6) Boutons Add / Combine ──────────────────────────────────────
    add_btn     = widgets.Button(
        description="Add Row", button_style="info",
        layout=widgets.Layout(width="120px")
    )
    combine_btn = widgets.Button(
        description="Combine & Save", button_style="success",
        layout=widgets.Layout(width="150px")
    )
    add_btn.on_click(add_row)

    def on_combine(btn):
        with output_area:
            clear_output()
            print("🔄 Combine clicked")  # debug
            try:
                combined = []
                for it in row_widgets:
                    g, m = it["geom"].value, it["mat"].value
                    if g is None or m is None:
                        continue
                    combined.append({
                        "config_name": f"{g['config_name']} - {m['config_name']}",
                        "geometry":    g,
                        "material":    m
                    })
                if not combined:
                    print("❗ No valid combination selected.")
                    return

                # on s'assure que le dossier existe
                os.makedirs(CONFIGURATIONS_dir, exist_ok=True)

                # on lit l'existant
                try:
                    with open(combos_file, "r", encoding="utf-8") as f:
                        existing = json.load(f).get("ALL_COMBINED_CONFIGS", [])
                except (FileNotFoundError, json.JSONDecodeError):
                    existing = []

                # on fusionne sans doublons
                merged = {c["config_name"]: c for c in existing}
                for c in combined:
                    merged[c["config_name"]] = c

                # on écrit
                with open(combos_file, "w", encoding="utf-8") as f:
                    json.dump(
                        {"ALL_COMBINED_CONFIGS": list(merged.values())},
                        f, indent=2, ensure_ascii=False
                    )
                print(f"✅ Saved {len(merged)} combination(s) to\n   {combos_file}")

                # on rafraîchit l'affichage
                _reload_config()

            except Exception as e:
                # on capture TOUTE erreur et on l'affiche
                print("🔥 Exception in on_combine:", e)

    combine_btn.on_click(on_combine)

    control_buttons = widgets.HBox(
        [add_btn, combine_btn],
        layout=widgets.Layout(justify_content="space-around", margin="10px 0")
    )


    # ─── (7) SelectMultiple + bouton pour suppression ───────────────────
    delete_selector = widgets.SelectMultiple(
        options=[], description="Delete config(s):",
        layout=widgets.Layout(width="400px", height="100px")
    )
    delete_btn = widgets.Button(
        description="Delete selected", button_style="danger",
        layout=widgets.Layout(width="150px")
    )

    def on_delete_selected(_):
        with output_area:
            clear_output()
            to_delete = delete_selector.value
            if not to_delete:
                print("No configuration selected to delete.")
                return
            try:
                saved = json.load(open(combos_file, "r", encoding="utf-8")) \
                                .get("ALL_COMBINED_CONFIGS", [])
            except:
                saved = []
            remaining = [c for c in saved if c["config_name"] not in to_delete]
            json.dump(
                {"ALL_COMBINED_CONFIGS": remaining},
                open(combos_file, "w", encoding="utf-8"),
                indent=2, ensure_ascii=False
            )
            print(f"Deleted {len(to_delete)} configuration(s).")
            delete_selector.value = ()
            _reload_config()

    delete_btn.on_click(on_delete_selected)

    delete_box = widgets.HBox(
        [delete_selector, delete_btn],
        layout=widgets.Layout(justify_content="space-between", margin="10px 0")
    )

    # ─── (8) Assemblage final ────────────────────────────────────────────
    main_widget = widgets.VBox(
        [rows_container, control_buttons, delete_box, output_area],
        layout=widgets.Layout(padding="10px", width="650px", background_color="#f9f9f9")
    )
    # Empêche le watcher d'être garbage-collected
    main_widget._config_watcher = _watcher
    
    # ─── (9) On force un premier chargement des configs existantes ────────
    _reload_config()
    
    return main_widget
