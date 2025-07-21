# -*- coding: utf-8 -*-
"""
accuracy_checking.py
--------------------
Onglet « accuracy_checking » (ex-« Difference ») de l’application interactive :

▪ Colonne gauche  : comparaison de deux spectres (Δ Reflectance)  
▪ Colonne droite  : 1 – n sous-onglets, chacun pilotant un notebook PyMoosh
                    converti en .py et placé dans utils/PyMoosh/.

Les notebooks PyMoosh doivent simplement exposer une fonction
create_…_tab() renvoyant un Widget ; ils sont chargés automatiquement.
"""

from __future__ import annotations

# ------------------------------------------------------------------ #
#                                imports                             #
# ------------------------------------------------------------------ #
import io, base64, os, importlib.util, pathlib
from datetime import datetime
from types import ModuleType

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import HTML, display

from gap_plasmon_2d import paths
from gap_plasmon_2d.utils.data_readers          import get_all_spectra_and_summaries
from gap_plasmon_2d.simulation.simulate_and_plot import ordered_params


# ------------------------------------------------------------------ #
#                           chemins communs                          #
# ------------------------------------------------------------------ #
module_dir    = pathlib.Path(__file__).resolve().parent
workspace_dir = module_dir.parent                      # …/gap_plasmon_2d
notebooks_dir = pathlib.Path(paths.RESULTS_DIR)
summary_dir   = notebooks_dir / "summary_simulation"
exp_data_dir  = notebooks_dir / "Experimental_Data"

# Dossier contenant vos notebooks PyMoosh convertis en .py
pymoosh_root  = workspace_dir / "utils" / "PyMoosh"


# ------------------------------------------------------------------ #
#                    utilitaire : chargement dynamique               #
# ------------------------------------------------------------------ #
def _load_py_module(py_path: pathlib.Path) -> ModuleType:
    """Charge dynamiquement un fichier .py arbitraire et renvoie le module."""
    spec = importlib.util.spec_from_file_location(py_path.stem, str(py_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Impossible de charger {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # type: ignore[arg-type]
    return mod


# ------------------------------------------------------------------ #
#                     helper : lien de téléchargement                #
# ------------------------------------------------------------------ #
def _download_link(fig, fname: str) -> HTML:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return HTML(
        f'<a download="{fname}" href="data:image/png;base64,{b64}" '
        'target="_blank" style="font-weight:600">Télécharger l’image</a>'
    )


# ------------------------------------------------------------------ #
#                     sous-onglet : comparateur de spectres          #
# ------------------------------------------------------------------ #
def _create_ratio_panel() -> widgets.VBox:
    ref_dd = widgets.Dropdown(description="Base :",
                              layout=widgets.Layout(width="350px"),
                              style={'description_width': 'initial'})
    tgt_dd = widgets.Dropdown(description="Comparé à :",
                              layout=widgets.Layout(width="350px"),
                              style={'description_width': 'initial'})
    draw_b = widgets.Button(description="Tracer le ratio",
                            button_style="warning")
    out    = widgets.Output(layout=widgets.Layout(
        border="1px solid #AAA", padding="10px", min_height="420px"))

    # --------- chargement initial des options ---------
    def _refresh_options(*_):
        Rup_dict, *_ = get_all_spectra_and_summaries(
            summary_dir, exp_data_dir, ordered_params)
        opts = list(Rup_dict.keys())
        ref_dd.options = opts
        tgt_dd.options = opts
    _refresh_options()

    # --------- callback tracé ---------
    def _draw(_btn):
        with out:
            out.clear_output()
            if not (ref_dd.value and tgt_dd.value):
                print("Merci de sélectionner les deux spectres.")
                return

            Rup_dict, *_ = get_all_spectra_and_summaries(
                summary_dir, exp_data_dir, ordered_params)
            ref = Rup_dict.get(ref_dd.value)
            tgt = Rup_dict.get(tgt_dd.value)
            if ref is None or tgt is None:
                print("Spectres introuvables.")
                return

            wl1, R1 = ref
            wl2, R2 = tgt
            R2_interp = np.interp(wl1, wl2, R2) if not np.array_equal(wl1, wl2) else R2
            diff_R = np.asarray(R2_interp) - np.asarray(R1)

            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(wl1, diff_R, label=f"{tgt_dd.value} – {ref_dd.value}")
            ax.axhline(0, color="k", ls="--", lw=1)
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Δ Reflectance")
            ax.set_title("Difference spectrum")
            ax.grid(True); ax.legend()
            display(fig, _download_link(fig, f"ratio_{datetime.now():%Y%m%d_%H%M%S}.png"))
            plt.close(fig)

    draw_b.on_click(_draw)

    panel = widgets.VBox([widgets.HTML("<b>Comparer deux spectres</b>"),
                          ref_dd, tgt_dd, draw_b, out])
    # expose pour interactive_simulation.py
    panel.update_diff_options = _refresh_options
    return panel


# ------------------------------------------------------------------ #
#                sous-onglets PyMoosh chargés dynamiquement          #
# ------------------------------------------------------------------ #
def _create_pymoosh_tabs() -> widgets.Tab:
    tabs = widgets.Tab(layout=widgets.Layout(min_width="600px"))
    children: list[widgets.Widget] = []
    titles:   list[str]            = []

    if not pymoosh_root.exists():
        err = widgets.HTML(f"<b style='color:red'>Dossier {pymoosh_root} introuvable.</b>")
        tabs.children = [err]
        tabs.set_title(0, "Erreur")
        return tabs

    for py_file in sorted(pymoosh_root.glob("*.py")):
        try:
            mod = _load_py_module(py_file)
        except Exception as exc:
            children.append(widgets.HTML(
                f"<pre style='color:red'>Erreur de chargement {py_file.name} :\n{exc}</pre>"))
            titles.append(py_file.stem[:12])
            continue

        # détecte les fonctions create_*_tab()
        fcts = [getattr(mod, f) for f in dir(mod)
                if f.startswith("create_") and f.endswith("_tab")
                and callable(getattr(mod, f))]
        if not fcts:
            children.append(widgets.HTML(
                f"<pre style='color:red'>Aucune create_*_tab() dans {py_file.name}</pre>"))
            titles.append(py_file.stem[:12])
            continue

        for fct in fcts:
            try:
                widget = fct()
            except Exception as exc:
                widget = widgets.HTML(
                    f"<pre style='color:red'>Exception dans {fct.__name__} :\n{exc}</pre>")
            children.append(widget)
            nice_title = (fct.__name__
                          .replace("create_", "")
                          .replace("_tab", "")
                          .replace("_", " ")
                          .title())
            titles.append(nice_title[:20])

    if not children:
        children = [widgets.HTML("<i>Pas de module PyMoosh détecté.</i>")]
        titles   = ["PyMoosh"]

    tabs.children = children
    for i, t in enumerate(titles):
        tabs.set_title(i, t)
    return tabs


# ------------------------------------------------------------------ #
#                 fonction exportée : create_difference_tab          #
# ------------------------------------------------------------------ #
def create_difference_tab() -> widgets.HBox:
    """
    Construit le widget « accuracy_checking » attendu par interactive_simulation.py :
        ┌───────────────┬────────────────────┐
        │ Comparateur   │  Onglets PyMoosh   │
        └───────────────┴────────────────────┘
    Le widget principal expose .update_diff_options pour rafraîchir la liste
    des spectres comme le fait déjà l’application.
    """
    ratio_panel   = _create_ratio_panel()
    pymoosh_tabs  = _create_pymoosh_tabs()

    # --- même flex pour les deux colonnes ---
    col_layout = widgets.Layout(flex='1 1 0%', width='50%')
    ratio_panel.layout  = col_layout
    pymoosh_tabs.layout = col_layout

    ui = widgets.HBox(
        [ratio_panel, pymoosh_tabs],
        layout=widgets.Layout(width='100%',  # occupe tout le conteneur parent
                              gap='20px',
                              align_items='flex-start')
    )
    
    # interactive_simulation.py appelle diff_tab.update_diff_options()
    ui.update_diff_options = ratio_panel.update_diff_options  # type: ignore[attr-defined]
    return ui
