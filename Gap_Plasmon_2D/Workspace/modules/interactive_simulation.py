#!/usr/bin/env python3
"""
Module: interactive_simulation.py

Assemble les onglets Simulation, Plot et Difference pour l'application interactive.
"""

import os
import ipywidgets as widgets

from simulation  import create_simulation_tab
from plotting    import create_plot_tab
from difference  import create_difference_tab

# Construction des chemins
module_dir         = os.path.dirname(os.path.abspath(__file__))
workspace_dir      = os.path.dirname(module_dir)
notebooks_dir      = os.path.join(workspace_dir, "notebooks")
summary_dir        = os.path.join(notebooks_dir, "Summary_Simulation")
exp_data_dir       = os.path.join(notebooks_dir, "Experimental_Data")
data_dir           = os.path.join(workspace_dir, "data")
json_combined_path = os.path.join(data_dir, "combined_materials.json")


def create_advanced_app():
    """
    Crée et retourne l'interface interactive complète, avec trois onglets :
      - Simulation
      - Plot
      - Difference
    """
    sim_tab  = create_simulation_tab(json_combined_path, summary_dir, exp_data_dir)
    plot_tab = create_plot_tab()
    diff_tab = create_difference_tab()

    tabs = widgets.Tab(children=[sim_tab, plot_tab, diff_tab])
    tabs.set_title(0, "Simulation")
    tabs.set_title(1, "Plot")
    tabs.set_title(2, "Double checking")
    
    def on_tab_change(change):
        # si on passe à l’onglet Plot (index 1), on rafraîchit la liste des spectres
        if change['new'] == 1:
            plot_tab.update_spectra()
        elif change['new'] == 2:
            diff_tab.update_diff_options()    

    tabs.observe(on_tab_change, names='selected_index')
    

    return widgets.VBox([tabs])
