# utils/PyMoosh/01_simple_Bragg_optimization_with_pymoosh.py
import numpy as np, ipywidgets as w, matplotlib.pyplot as plt, PyMoosh as pm
from IPython.display import display
import io, base64


def setup_structure(thick_list, mat_env, mat1, mat2):
    """helper to create pymoosh structure object, alternating 2 materials

    Args:
        thick_list (list): list of thicknesses, top layer first
        mat_env (float): environment ref. index
        mat1 (float): material 1 ref. index
        mat2 (float): material 2 ref. index

    Returns:
        PyMoosh.structure: multi-layer structure object
    """
    thick_list = list(
        thick_list)  # convert to list for convenience when stacking layers
    n = len(thick_list)

    materials = [mat_env**2, mat1**2, mat2**2]  # permittivities!
    # periodic stack. first layer: environment, last layer: substrate
    stack = [0] + [2, 1] * (n//2) + [2]
    thicknesses = [0.] + thick_list + [0.]

    structure = pm.Structure(
        materials, stack, np.array(thicknesses), verbose=False)

    return structure




# ------- the optimization target function -------
def cost_minibragg(x, mat_env, mat1, mat2, eval_wl):
    """ cost function: maximize reflectance of a layer-stack

    Args:
        x (list): thicknesses of all the layers, starting with the upper one.

    Returns:
        float: 1 - Reflectivity at target wavelength
    """
    structure = setup_structure(x, mat_env, mat1, mat2)

    # the actual PyMoosh reflectivity simulation
    #_, R = pm.coefficient_I(structure, eval_wl, 0., 0)
    _, _, R, _ = pm.coefficient(structure, eval_wl, 0., 0)
    cost = 1 - R

    return cost



# ------------------------------------------------------------------
# helper : lien de téléchargement + balise <img>                     #
# ------------------------------------------------------------------
def _fig_to_html(fig, fname: str) -> str:
    """<div> contenant l’image PNG + lien « Télécharger »"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode()
    img = f'<img src="data:image/png;base64,{b64}" style="max-width:100%;">'
    link = (f'<a download="{fname}" href="data:image/png;base64,{b64}" '
            'target="_blank">Télécharger</a>')
    return f'<div style="flex:1 1 0%; text-align:center;">{img}<br>{link}</div>'


# ------------------------------------------------------------------
#  create_bragg_tab()                                                #
# ------------------------------------------------------------------
def create_bragg_tab():
    # ───────── widgets de paramétrage ─────────
    n_layers  = w.IntSlider(value=10,  min=2,   max=30,
                            description="nb_layers")
    target_wl = w.FloatSlider(value=600., min=200., max=2000., step=1,
                              description="target_wl [nm]")
    mat_env   = w.FloatText(value=1.0,  description="mat_env")
    mat1      = w.FloatText(value=1.4,  description="mat1")
    mat2      = w.FloatText(value=1.8,  description="mat2")
    min_thick = w.FloatText(value=0.0,  description="min_thick [nm]")
    max_thick = w.Label(description="max_thick [nm]")
    budget_w   = w.IntText(value=10_000, description="budget")
    pop_w      = w.IntText(value=30,     description="population")
    wls_start = w.FloatText(value=400., description="λ start [nm]")
    wls_stop  = w.FloatText(value=1000., description="λ stop [nm]")
    run_btn   = w.Button(description="Run optimisation", button_style="success")
    out       = w.Output(layout=w.Layout(border="1px solid #bbb",
                                         min_height="440px")) 

    # ─── maj auto de max_thick ───
    def _update_max(*_):
        v = target_wl.value / (2 * mat1.value)
        max_thick.value = f"{v:0.1f}"
    target_wl.observe(_update_max, "value"); mat1.observe(_update_max, "value")
    _update_max()

    # ─── callback optimisation ───
    # --------------------------------------------------------
    def _run(_):
        out.clear_output()
        with out:
            # ── 0. paramètres utilisateur ────────────────────
            nb_layers = n_layers.value
            target    = target_wl.value
            env, n1, n2 = mat_env.value, mat1.value, mat2.value
            min_t     = min_thick.value
            max_t     = target / (2 * n1)
            lam_grid  = np.linspace(wls_start.value, wls_stop.value, 121)

            # ── 1. optimisation DE ───────────────────────────
            X_min = np.full(nb_layers, min_t)
            X_max = np.full(nb_layers, max_t)
            budget   = budget_w.value
            popsize  = pop_w.value

            def cost(x): return cost_minibragg(x, env, n1, n2, target)
            best, conv = pm.differential_evolution(
                cost, budget, X_min, X_max, population=popsize)
            

            struct = setup_structure(best, env, n1, n2)

            # ── 2. création des figures (affichage désactivé) ───────
            was_inter = plt.isinteractive()
            plt.ioff()


            # --- Figure A  : pile -------------------------------------------------
            fig_stack = plt.figure()          # nouvelle figure courante
            struct.plot_stack()               # dessine la pile dans CETTE figure

            # --- Figure B  : convergence -----------------------------------------
            fig_conv, ax_conv = plt.subplots()
            ax_conv.plot(range(len(conv)), conv)
            ax_conv.set_xlabel("iteration"); ax_conv.set_ylabel("1 – R")
            ax_conv.set_title("Convergence")

            # --- Figure C  : réflectance -----------------------------------------
            R = [pm.coefficient(struct, wl, 0, 0)[2] for wl in lam_grid]
            fig_ref, ax_ref = plt.subplots()
            ax_ref.plot(lam_grid, R); ax_ref.axvline(target, ls="--")
            ax_ref.set_xlabel("λ (nm)"); ax_ref.set_ylabel("R")
            ax_ref.set_title("Reflectance")

            if was_inter:
                plt.ion()

            # --- Conversion PNG + affichage inline ---------------------------
            html_block = (
                '<div style="display:flex; gap:10px; width:100%;">'
                + _fig_to_html(fig_stack, "stack.png")
                + _fig_to_html(fig_conv,  "convergence.png")
                + _fig_to_html(fig_ref,   "reflectance.png")
                + '</div>'
            )

            out.clear_output()               # supprime toute image parasite
            display(w.HTML(html_block))

    run_btn.on_click(_run)

    ctrls = w.VBox([n_layers, target_wl, mat_env, mat1, mat2,
                    min_thick, max_thick])
    ctrls_2 = w.VBox([budget_w, pop_w, wls_start, wls_stop, run_btn])

    Controleur = w.HBox([ctrls, ctrls_2])
    return w.VBox([Controleur, out])
