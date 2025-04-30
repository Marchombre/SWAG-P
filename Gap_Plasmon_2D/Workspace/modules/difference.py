#!/usr/bin/env python3
"""
Module: difference.py

Cette partie gère l'onglet "Ratio" de l'application interactive.
"""

import os
import io, base64
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import HTML, display, Javascript

from data_readers import get_all_spectra_and_summaries
from simulate_and_plot import ordered_params

# Construction des chemins
module_dir    = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(module_dir)
notebooks_dir = os.path.join(workspace_dir, "notebooks")
summary_dir   = os.path.join(notebooks_dir, "Summary_Simulation")
exp_data_dir  = os.path.join(notebooks_dir, "Experimental_Data")

# --- Téléchargement de la figure ---
def create_download_link(fig, filename="figure.png"):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    href = (
        f'<a download="{filename}" href="data:image/png;base64,{b64}" '
        f'target="_blank">Télécharger l\'image</a>'
    )
    return HTML(href)

# --- Onglet Difference ---
def create_difference_tab():
    # Widgets
    diff_ref_dropdown = widgets.Dropdown(
        options=[],
        description="Base:",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='500px')
    )
    diff_target_dropdown = widgets.Dropdown(
        options=[],
        description="Comparing to:",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='500px')
    )
    diff_button = widgets.Button(
        description="Draw ratio", button_style="warning",
        tooltip="Drawing the ratio between two spectra"
    )
    diff_output = widgets.Output(
        layout=widgets.Layout(border="2px solid #ccc", padding="10px", min_height="400px")
    )

    # Mise à jour des options des dropdowns
    def update_diff_options():
        spectra_all, _, _ = get_all_spectra_and_summaries(
            summary_dir, exp_data_dir, ordered_params
        )
        options = list(spectra_all.keys())
        diff_ref_dropdown.options    = options
        diff_target_dropdown.options = options

    update_diff_options()

    # Callback du bouton
    def on_diff_button_clicked(b):
        ref_label    = diff_ref_dropdown.value
        target_label = diff_target_dropdown.value
        if not ref_label or not target_label:
            with diff_output:
                diff_output.clear_output()
                print("Veuillez sélectionner les deux spectres.")
            return
        spectra_all, _ = get_all_spectra_and_summaries(
            summary_dir, exp_data_dir, ordered_params
        )
        ref_data    = spectra_all.get(ref_label)
        target_data = spectra_all.get(target_label)
        if ref_data is None or target_data is None:
            with diff_output:
                diff_output.clear_output()
                print("Données introuvables pour l'un des spectres.")
            return
        wl1, R1 = ref_data
        wl2, R2 = target_data
        if np.array_equal(wl1, wl2):
            common_wl = wl1
            diff_R     = np.array(R2) - np.array(R1)
        else:
            common_wl = wl1
            diff_R     = np.array(np.interp(wl1, wl2, R2)) - np.array(R1)

        fig = plt.figure(figsize=(10, 6))
        ax  = fig.add_axes([0.1, 0.15, 0.8, 0.75])
        ax.plot(common_wl, diff_R,
                label=f"Diff: {target_label} - {ref_label}",
                color="blue")
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Différence de reflectance")
        ax.set_title(f"Différence: {target_label} - {ref_label}")
        ax.legend()
        ax.grid(True)

        with diff_output:
            diff_output.clear_output()
            display(fig)
            link = create_download_link(
                fig,
                filename=f"ratio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            display(link)
            plt.close(fig)

    diff_button.on_click(on_diff_button_clicked)

    diff_controls = widgets.VBox([
        widgets.HTML("<h3>Ratio</h3>"),
        diff_ref_dropdown,
        diff_target_dropdown,
        diff_button
    ])

    diff_tab = widgets.VBox([diff_controls, diff_output])
    diff_tab.update_diff_options = update_diff_options
    return diff_tab
