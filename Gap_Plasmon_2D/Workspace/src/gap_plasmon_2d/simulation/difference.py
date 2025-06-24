from gap_plasmon_2d import paths
#!/usr/bin/env python3
"""
Module: difference.py

Gère l'onglet « Ratio ». Version mise à jour pour la nouvelle API
de data_readers.get_all_spectra_and_summaries, qui renvoie maintenant :
    (Rup_dict, Rup_dn_dict, summaries, metrics_dict)
"""

# ------------------------------------------------------------------ #
#                                imports                             #
# ------------------------------------------------------------------ #
import os
import io, base64
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import HTML, display

from gap_plasmon_2d.utils.data_readers     import get_all_spectra_and_summaries
from gap_plasmon_2d.simulation.simulate_and_plot import ordered_params

# ------------------------------------------------------------------ #
#                           chemins communs                          #
# ------------------------------------------------------------------ #
module_dir    = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(module_dir)
notebooks_dir = os.path.join(str(paths.RESULTS_DIR))
summary_dir   = os.path.join(notebooks_dir, "summary_simulation")
exp_data_dir  = os.path.join(notebooks_dir, "Experimental_Data")

# ------------------------------------------------------------------ #
#                     helper: lien de téléchargement                 #
# ------------------------------------------------------------------ #
def _download_link(fig, fname):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return HTML(
        f'<a download="{fname}" href="data:image/png;base64,{b64}" '
        'target="_blank">Télécharger l’image</a>'
    )

# ------------------------------------------------------------------ #
#                           Onglet « Ratio »                         #
# ------------------------------------------------------------------ #
def create_difference_tab():
    # --------------- widgets principaux ---------------------------- #
    ref_dd = widgets.Dropdown(description="Base:",
                              layout=widgets.Layout(width="500px"),
                              style={'description_width': 'initial'})
    tgt_dd = widgets.Dropdown(description="Comparing to:",
                              layout=widgets.Layout(width="500px"),
                              style={'description_width': 'initial'})
    draw_b = widgets.Button(description="Draw ratio",
                            button_style="warning",
                            tooltip="Draw the ratio between two spectra")
    out    = widgets.Output(layout=widgets.Layout(
                border="2px solid #ccc", padding="10px", min_height="400px"))

    # --------------- chargement / rafraîchissement ---------------- #
    def _refresh_options():
        # ↓↓↓ 4 valeurs depuis la nouvelle API
        Rup_dict, *_ = get_all_spectra_and_summaries(
            summary_dir, exp_data_dir, ordered_params)

        opts = list(Rup_dict.keys())
        ref_dd.options = opts
        tgt_dd.options = opts
    _refresh_options()

    # --------------- callback « Draw » ----------------------------- #
    def _draw(_btn):
        ref_lbl, tgt_lbl = ref_dd.value, tgt_dd.value
        if not ref_lbl or not tgt_lbl:
            with out:
                out.clear_output()
                print("Veuillez sélectionner les deux spectres.")
            return

        # récupération des données
        Rup_dict, *_ = get_all_spectra_and_summaries(
            summary_dir, exp_data_dir, ordered_params)
        ref = Rup_dict.get(ref_lbl); tgt = Rup_dict.get(tgt_lbl)
        if ref is None or tgt is None:
            with out:
                out.clear_output()
                print("Données introuvables pour l'un des spectres.")
            return

        wl1, Rup_dn = ref
        wl2, R2 = tgt
        if np.array_equal(wl1, wl2):
            wl_common = wl1
            diff_R   = np.asarray(R2) - np.asarray(Rup_dn)
        else:
            wl_common = wl1
            diff_R = np.asarray(np.interp(wl1, wl2, R2)) - np.asarray(Rup_dn)

        # tracé
        fig = plt.figure(figsize=(10, 6))
        ax  = fig.add_axes([0.1, 0.15, 0.8, 0.75])
        ax.plot(wl_common, diff_R,
                label=f"{tgt_lbl} – {ref_lbl}", color="blue")
        ax.axhline(0, color="black", ls="--", lw=1)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Δ Reflectance")
        ax.set_title(f"Difference: {tgt_lbl} – {ref_lbl}")
        ax.legend(); ax.grid(True)

        with out:
            out.clear_output()
            display(fig)
            display(_download_link(fig,
                    f"ratio_{datetime.now():%Y%m%d_%H%M%S}.png"))
            plt.close(fig)

    draw_b.on_click(_draw)

    # --------------- assemblage ------------------------------------ #
    ctrls = widgets.VBox([
        widgets.HTML("<h3>Ratio</h3>"),
        ref_dd, tgt_dd, draw_b
    ])
    tab = widgets.VBox([ctrls, out])
    tab.update_diff_options = _refresh_options
    return tab
