from gap_plasmon_2d import paths
import ipywidgets as widgets
from IPython.display import clear_output
import os
import json
import threading

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


# -----------------------------------------------------------------------------#
#  Geometry × Material — modern UI                                             #
# -----------------------------------------------------------------------------#
def create_geometry_material_widget():
    """
    Sélecteur visuel (géométrie, matériau) modernisé :
      • lignes compactes façon tableau ;
      • boutons icônes (➕  ✅  🗑️) avec tooltips ;
      • panneau « Saved combos » pliable, suppression par clic direct ;
      • rechargement auto quand un JSON change (watchdog conservé).
    """

    # ╭─ 0 |  Data & chemins  ──────────────────────────────────────────────────╮
    CONFIG_DIR  = str(paths.CONFIGS_DIR)
    combos_file = os.path.join(CONFIG_DIR, "geom_mat_combinations.json")

    def _load(fname, key):
        try:
            return (load_json_config(fname) or {}).get(key, [])
        except FileNotFoundError:
            return []

    # chargement initial
    geom_data = _load("geometry_configurations.json", "ALL_GEOMETRY_CONFIGS")
    mat_data  = _load("material_config.json",          "ALL_CONFIGS")


    # ╭─ 1 |  Helpers de rechargement  ────────────────────────────────────────╮
    def _reload_options():
        nonlocal geom_data, mat_data
        geom_data = _load("geometry_configurations.json", "ALL_GEOMETRY_CONFIGS")
        mat_data  = _load("material_config.json",          "ALL_CONFIGS")

        # met à jour chaque dropdown existant
        for row in row_pool:
            for dd, src in ((row.children[0], geom_data),
                            (row.children[1], mat_data)):
                opts = [(c["config_name"], c) for c in src] or [("—", None)]
                prev = dd.value
                dd.options = opts
                if prev not in [o[1] for o in opts]:
                    dd.value = opts[0][1]

        # recharge la liste saved combos
        try:
            saved = json.load(open(combos_file, encoding="utf-8")) \
                        .get("ALL_COMBINED_CONFIGS", [])
        except (FileNotFoundError, json.JSONDecodeError):
            saved = []
        saved_sel.options = [c["config_name"] for c in saved]


    # ╭─ 2 |  Watcher automatique  ────────────────────────────────────────────╮
    # On délègue au DebouncedEventHandler (start_watcher) le filtrage/.json + debounce
    # Le callback _reload_options sera invoqué après tout create/modify/delete .json
    _watcher = start_watcher(
        path=CONFIG_DIR,
        callback=_reload_options,
        extensions=[".json"],
        recursive=False,
    )


    # ╭─ 3 |  Fabrique des Dropdowns  ────────────────────────────────────────╮
    def _geom_dd():
        opts = [(c["config_name"], c) for c in geom_data] or [("—", None)]
        return widgets.Dropdown(
            options=opts, value=opts[0][1],
            layout=widgets.Layout(width="240px")
        )

    def _mat_dd():
        opts = [(c["config_name"], c) for c in mat_data] or [("—", None)]
        return widgets.Dropdown(
            options=opts, value=opts[0][1],
            layout=widgets.Layout(width="240px")
        )


    # ╭─ 4 |  Pool de lignes & conteneur  ────────────────────────────────────╮
    row_pool, rows_box = [], widgets.VBox()


    # ╭─ 5 |  Ajout / suppression d'une ligne  ───────────────────────────────╮
    def _add_row(_=None):
        geom, mat = _geom_dd(), _mat_dd()
        trash = widgets.Button(icon="trash", tooltip="Delete row",
                               layout=widgets.Layout(width="38px"),
                               button_style="danger")
        row = widgets.HBox([geom, mat, trash],
                           layout=widgets.Layout(gap="6px", align_items="center"))
        row_pool.append(row)
        rows_box.children = tuple(row_pool)

        def _on_delete(_btn, row=row):
            row_pool.remove(row)
            rows_box.children = tuple(row_pool)

        trash.on_click(_on_delete)

    _add_row()  # première ligne par défaut


    # ╭─ 6 |  Actions globales (Add / Save)  ─────────────────────────────────╮
    btn_add  = widgets.Button(icon="plus",  tooltip="Add a new row",
                              button_style="info",
                              layout=widgets.Layout(width="38px"))
    btn_save = widgets.Button(icon="check", tooltip="Combine & save selections",
                              button_style="success",
                              layout=widgets.Layout(width="38px"))

    btn_add.on_click(_add_row)


    # ╭─ 7 |  Saved combos (SelectMultiple + Delete)  ────────────────────────╮
    saved_sel = widgets.SelectMultiple(layout=widgets.Layout(height="140px"))
    btn_del_saved = widgets.Button(icon="trash", button_style="danger",
                                   tooltip="Delete selected combos",
                                   layout=widgets.Layout(width="38px"))
    saved_box = widgets.HBox([saved_sel, btn_del_saved],
                             layout=widgets.Layout(gap="6px"))
    acc_saved = widgets.Accordion(children=[saved_box])
    acc_saved.set_title(0, "📁  Saved combinations")


    # ╭─ 8 |  Sauvegarde & suppression dans le fichier JSON  ────────────────╮
    def _save(_):
        combos = []
        for row in row_pool:
            g, m = row.children[0].value, row.children[1].value
            if g and m:
                combos.append({
                    "config_name": f"{g['config_name']} + {m['config_name']}",
                    "geometry": g, "material": m
                })

        if not combos:
            status.value = "⚠️ Nothing to save."
            return

        try:
            existing = json.load(open(combos_file, encoding="utf-8")) \
                          .get("ALL_COMBINED_CONFIGS", [])
        except (FileNotFoundError, json.JSONDecodeError):
            existing = []

        merged = {c["config_name"]: c for c in existing}
        for c in combos:
            merged[c["config_name"]] = c

        os.makedirs(CONFIG_DIR, exist_ok=True)
        json.dump({"ALL_COMBINED_CONFIGS": list(merged.values())},
                  open(combos_file, "w", encoding="utf-8"),
                  indent=2, ensure_ascii=False)
        status.value = f"✅ Saved {len(combos)} structure(s)."
        _reload_options()

    btn_save.on_click(_save)

    def _del_saved(_):
        todel = set(saved_sel.value)
        if not todel:
            status.value = "⚠️ Select combos to delete."
            return
        try:
            data = json.load(open(combos_file, encoding="utf-8"))
            keep = [c for c in data.get("ALL_COMBINED_CONFIGS", [])
                    if c["config_name"] not in todel]
            json.dump({"ALL_COMBINED_CONFIGS": keep},
                      open(combos_file, "w", encoding="utf-8"),
                      indent=2, ensure_ascii=False)
            status.value = f"🗑️ Deleted {len(todel)} combo(s)."
            _reload_options()
        except FileNotFoundError:
            status.value = "⚠️ Nothing to delete."

    btn_del_saved.on_click(_del_saved)


    # ╭─ 9 |  Barre de statut  ───────────────────────────────────────────────╮
    status = widgets.HTML("")
    status.layout.margin = "4px 0 0 4px"


    # ╭─ 10 |  Mise en page globale  ────────────────────────────────────────╮
    header = widgets.HTML(
        "<h3 style='margin:0;font-family:Segoe UI,Arial;'>"
        "Geometry × Material composer</h3>"
    )
    buttons_bar = widgets.HBox(
        [btn_add, btn_save],
        layout=widgets.Layout(gap="8px", align_items="center")
    )
    style = widgets.HTML("""
    <style>
    .combo-panel{
        border:1px solid #ddd;border-radius:8px;padding:12px;
        background:#fafafa;box-shadow:0 2px 4px rgba(0,0,0,.05);
    }
    </style>
    """)

    panel = widgets.VBox(
        [style, header, rows_box, buttons_bar, acc_saved, status],
        layout=widgets.Layout(width="680px", gap="6px"),
    )
    # on garde la référence à l'observer+handler pour qu'ils ne soient pas garbage-collectés
    panel._watcher = _watcher

    # initialisation immédiate
    _reload_options()

    return panel
