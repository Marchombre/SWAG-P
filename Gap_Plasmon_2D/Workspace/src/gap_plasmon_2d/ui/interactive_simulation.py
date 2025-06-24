from gap_plasmon_2d import paths
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
interactive_simulation.py

Assemble les onglets Simulation, Plot, Difference et Optimisation
pour l'application interactive.
"""

import os
import inspect
import ipywidgets as widgets

from gap_plasmon_2d.simulation.simulation import SimulationTab
from gap_plasmon_2d.analysis.plotting import PlotTab
from gap_plasmon_2d.analysis.difference import create_difference_tab
from gap_plasmon_2d.optimisation.optimisation import OptimizationTab  


# --------------------------------------------------------------------- #
#                          Construction des chemins                    #
# --------------------------------------------------------------------- #
module_dir         = os.path.dirname(os.path.abspath(__file__))
workspace_dir      = os.path.dirname(module_dir)
notebooks_dir      = os.path.join(str(paths.RESULTS_DIR))
summary_dir        = os.path.join(notebooks_dir, "summary_simulation")
exp_data_dir       = os.path.join(notebooks_dir, "Experimental_Data")
data_dir           = os.path.join(str(paths.DATA_DIR))
json_combined_path = os.path.join(data_dir, "combined_materials.json")


# --------------------------------------------------------------------- #
def _extract_widget(obj):
    """
    Si obj est déjà un Widget, on le renvoie.
    Sinon, on cherche dans ses attributs un premier widget.
    """
    if isinstance(obj, widgets.Widget):
        return obj

    for name in ('widget', 'tab', 'container', 'layout', 'ui', 'root', 'view'):
        val = getattr(obj, name, None)
        if isinstance(val, widgets.Widget):
            return val

    for attr in dir(obj):
        try:
            val = getattr(obj, attr)
            if isinstance(val, widgets.Widget):
                return val
        except Exception:
            continue

    raise RuntimeError(
        f"Impossible de trouver un ipywidgets.Widget dans l'instance {obj!r}."
    )


# --------------------------------------------------------------------- #
def create_advanced_app():
    """
    Construit l'application interactive à onglets :
      0) Simulation,
      1) Plot,
      2) Validation (Difference),
      3) Optimisation
    """
    # 1) Instanciation de SimulationTab
    sig    = inspect.signature(SimulationTab.__init__)
    params = [p for p in sig.parameters if p != 'self']
    mapping = {
        'json_path':          json_combined_path,
        'json_combined_path': json_combined_path,
        'summary_dir':        summary_dir,
        'exp_data_dir':       exp_data_dir,
    }
    args = []
    for name in params:
        if name in mapping:
            args.append(mapping[name])
        else:
            raise RuntimeError(f"SimulationTab.__init__ attend '{name}' …")
    sim_obj = SimulationTab(*args)
    sim_tab = _extract_widget(sim_obj)

    # 2) Onglet Plot (class-based)
    plot_obj = PlotTab()                 # plus de fonction create_plot_tab
    plot_tab = _extract_widget(plot_obj)

    # 3) Onglet Difference (reste fonctionnel via la fonction)
    diff_obj = create_difference_tab()
    diff_tab = _extract_widget(diff_obj)

    # 4) Onglet Optimisation (class-based)
    opt_obj = OptimizationTab(sim_obj)
    opt_tab = _extract_widget(opt_obj)

    # 5) Assemblage des onglets
    tabs = widgets.Tab(children=[sim_tab, plot_tab, diff_tab, opt_tab])
    tabs.set_title(0, "Simulation")
    tabs.set_title(1, "Plot: Multi-spectra")
    tabs.set_title(2, "Validation")
    tabs.set_title(3, "Optimisation")

    # 6) Rafraîchissement à la sélection d'un onglet
    def on_tab_change(change):
        idx = change.get('new')
        if idx == 1 and hasattr(plot_obj, 'update_spectra'):
            plot_obj.update_spectra()
        elif idx == 2 and hasattr(diff_obj, 'update_diff_options'):
            diff_obj.update_diff_options()
        elif idx == 3 and hasattr(opt_obj, 'update_optimization'):
            opt_obj.update_optimization()

    tabs.observe(on_tab_change, names='selected_index')
    return widgets.VBox([tabs])
