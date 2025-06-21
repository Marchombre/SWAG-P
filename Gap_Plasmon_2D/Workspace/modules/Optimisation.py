#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimisation.py

Ce module encapsule l’onglet d’optimisation par Differential Evolution
dans une classe `OptimizationTab`, remplaçant les anciennes fonctions DE_general
et create_optimization_tab. 

Usage dans l’application interactive :
    from Optimisation import create_optimization_tab
    opt_tab = create_optimization_tab(sim_obj)
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
import ipywidgets as widgets
from IPython.display import clear_output
import geometry_settings


from simulation import SimulationTab
from Saving_Functions import save_optimization_hdf5
from data_readers import read_optimization_hdf5, list_optimization_files
from simulate_and_plot import run_simulation_one_combo
from file_watchers import start_watcher


from pathlib import Path
from tqdm.notebook import trange


# --------------------------------------------------------------------- #
#                               chemins                                 #
# --------------------------------------------------------------------- #
# S’assure que le dossier courant est sur le PYTHONPATH
module_dir         = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, module_dir)

BASE_NOTEBOOKS     = Path(__file__).resolve().parent.parent / "notebooks"
summary_opt_dir   = BASE_NOTEBOOKS / "Summary_Optimization"
summary_opt_dir.mkdir(parents=True, exist_ok=True)

workspace_dir      = os.path.dirname(module_dir)
configurations_dir = os.path.join(workspace_dir, "CONFIGURATIONS")
data_dir           = os.path.join(workspace_dir, "data")
json_combined_path = os.path.join(data_dir, "combined_materials.json")



class OptimizationTab:
    def __init__(self, sim_obj: SimulationTab):
        self.sim = sim_obj

        # 1) Conteneur des paramètres à optimiser (sera rempli par update_optimization)
        self.bounds_box = widgets.VBox(
            layout=widgets.Layout(
                border='1px solid #ccc',
                padding='8px',
                gap='5px'            # empile les HBox avec un espace
            )
        )
        
        # on calcule une seule fois le bon dossier
        self.summary_opt_dir = (
            Path(__file__).resolve().parent.parent
            / "notebooks" / "Summary_Optimization"
        )
        self.summary_opt_dir.mkdir(parents=True, exist_ok=True)
        
        # dropdown des fichiers d’optimisation
        self.opt_file_dd = widgets.Dropdown(
            options=list_optimization_files(str(self.summary_opt_dir)),
            description="Fichier Opt:",
            layout=widgets.Layout(width='400px')
        )
        
        
        # 3) watcher filesystem → utilise start_watcher()
        self._observer = start_watcher(
            path=str(self.summary_opt_dir),
            callback=self._refresh_file_list,
            extensions=['.h5'],
            recursive=False
        )
        
        
        # Bouton de tracé
        self.plot_btn = widgets.Button(
            description="Tracer Résultats",
            button_style='info'
        )
        self.plot_btn.on_click(self.plot_optimization_results)


        # 2) Contrôles DE
        self.budget_w = widgets.IntText(value=100, description="Budget")
        self.pop_w    = widgets.IntText(value=30,  description="Population")
        self.run_btn  = widgets.Button(description="Run DE", button_style='primary')
        self.out      = widgets.Output()

        controls = widgets.HBox(
            [
                self.budget_w,
                self.pop_w,
                self.run_btn,
                self.opt_file_dd,
                self.plot_btn
            ],
            layout=widgets.Layout(margin='10px', flex_wrap='wrap', align_items='center')
        )

        # 3) Assemblage de l'UI
        self.ui = widgets.VBox(
            [ self.bounds_box,
              controls,
              self.out ],
            layout=widgets.Layout(padding='10px')
        )

        # 4) Observer la sélection de config 
        for cb in self.sim.config_checkboxes.values():
            cb.observe(self.update_optimization, names='value')
        # initialisation
        self.update_optimization()

        # 5) Lancer l'optimisation
        self.run_btn.on_click(self._on_run)    
        
        
        
        self.json_combined_path = json_combined_path
  

    def _refresh_file_list(self):
        files = list_optimization_files(str(self.summary_opt_dir))
        # n’écrase que si c’est changé
        if set(files) != set(self.opt_file_dd.options):
            self.opt_file_dd.options = files


    def __del__(self):
        self._observer.stop()
        self._observer.join()



    def _on_run(self, _):
        self.out.clear_output()
        # 1) Récupérer les clés cochées
        keys = [k for k,w in self.param_widgets.items() if w['opt'].value]
        if not keys:
            with self.out:
                print("⚠️ Parameters to optimized: None.")
            return

        # 2) Construire lowers/uppers
        lowers = np.array([self.param_widgets[k]['low'].value for k in keys])
        uppers = np.array([self.param_widgets[k]['up'].value for k in keys])

        with self.out:
            print("🚀 Optimization is running, please wait.")
            conv, best = self.DE_general(
                budget=self.budget_w.value,
                Npop=self.pop_w.value,
                lowers=lowers,
                uppers=uppers,
                keys=keys,
                mode="dip"
            )
            
            files = list_optimization_files(str(self.summary_opt_dir))
            self.opt_file_dd.options = files
            
            print("✅ Optimization ended.")
            print(f"Best value : {conv[-1]:.6g}")
            print("Optimized vector :", best)

    def update_optimization(self, change=None):
        """
        Quand l'utilisateur change la config cochée, on :
        1) récupère la config unique,
        2) filtre les clés et bornes sur ses geometry keys,
        3) reconstruit les widgets de bornes,
        4) vide la sortie si nécessaire.

        Reconstruit la liste de paramètres avec, pour chacun :
        [Checkbox 'Opti?', Label clé, FloatText min, FloatText max]
        seuls les cochés seront passés à DE.
        
        Reconstruit dynamiquement, à partir de la config sélectionnée,
        une ligne [Opti?, nom, lower, upper] pour chaque paramètre
        dont la config a une valeur non nulle.
        """
        
        # 1) Trouver la config unique
        sels = [c for c in self.sim.all_configs
                if self.sim.config_checkboxes[c["config_name"]].value]
        if len(sels) != 1:
            self.bounds_box.children = []
            return

        geom = sels[0]["geometry"]["geometry"]
        rows = []
        self.param_widgets = {}

        # 2) Pour chaque couche non nulle
        for k, val in geom.items():
            if val == 0.0:
                continue
            low, high = geometry_settings.geometry_limits.get(k, (0.0, 0.0))
            # 3) Construire la ligne
            chk = widgets.Checkbox(
                value=True, indent=False, layout=widgets.Layout(width='30px'))
            lbl = widgets.Label(
                value=k, layout=widgets.Layout(width='150px'))
            lo = widgets.FloatText(
                value=low, description='min:',
                layout=widgets.Layout(width='120px'),
                style={'description_width':'40px'})
            hi = widgets.FloatText(
                value=high, description='max:',
                layout=widgets.Layout(width='120px'),
                style={'description_width':'40px'})

            self.param_widgets[k] = {'opt': chk, 'low': lo, 'up': hi}

            row = widgets.HBox(
                [chk, lbl, lo, hi],
                layout=widgets.Layout(align_items='center', gap='10px')
            )
            rows.append(row)

        # 4) Injecter verticalement
        self.bounds_box.children = rows
        self.out.clear_output()



    def DE_general(self, *, budget, Npop, lowers, uppers, keys, mode="dip"):
        """
        Differential Evolution “current-to-best/1/bin” sur les dimensions `keys`.
        Arguments :
        - budget : nombre total d’évaluations de cost() autorisées
        - Npop   : taille de la population
        - lowers : array de bornes inférieures, même longueur que keys
        - uppers : array de bornes supérieures, même longueur que keys
        - keys   : liste des noms de paramètres (dimensions) à optimiser
        - mode   : mode de calcul du coût, soit 'dip' soit 'half'
        Retour :
        - conv : array de longueur Ngen contenant à chaque génération
                la meilleure valeur de coût rencontrée
        - best : vecteur (len(keys),) représentant l’individu optimal
        """

        # 1) Vérifier que le budget permet au moins une évaluation par individu
        if budget < Npop:
            raise ValueError("Le budget doit être ≥ à la taille de la population")

        # 2) Déterminer le nombre de générations de DE
        #    Chaque génération réalise Npop évaluations, donc Ngen = budget // Npop
        Ngen = budget // Npop

        # 3) Nombre de paramètres optimisés (Nlayers)
        n_params = len(keys)

        # 4) Buffers pour stocker :
        #    - cf[i]    : coût de l’individu i dans la population
        #    - conv[g]  : meilleur coût trouvé à la génération g
        cf   = np.zeros(Npop)
        conv = np.zeros(Ngen)

        # ──────────────────────────────────────────────────────────
        # 5) INITIALISATION DE LA POPULATION
        # ──────────────────────────────────────────────────────────
        #    On génère uniformément dans [0,1) une matrice de taille
        #    (Npop, n_params)
        pop = np.random.rand(Npop, n_params)

        #    On applique la transformation linéaire pour passer
        #    de [0,1) à [low, high] pour chaque dimension :
        #      pop[:, j] = lowers[j] + (uppers[j] - lowers[j]) * pop[:, j]
        pop = lowers + (uppers - lowers) * pop

        # ──────────────────────────────────────────────────────────
        # 6) ÉVALUATION INITIALE
        # ──────────────────────────────────────────────────────────
        #    On calcule cost() une fois pour chaque individu de la population
        for i in range(Npop):
            # self.sim.cost(x, keys, mode) renvoie un scalaire de coût
            cf[i] = self.sim.cost(pop[i], keys, mode=mode)

        # ──────────────────────────────────────────────────────────
        # 7) BOUCLE PRINCIPALE DE DE (“current-to-best/1/bin”)
        # ──────────────────────────────────────────────────────────
        #    Définition des hyper-paramètres du DE :
        F1, F2, cr = 0.9, 0.8, 0.8  # mutation weights et taux de crossover
        
        self.out.clear_output()
        with self.out:
            for g in trange(Ngen, desc="Differential Evolution in progress"):
                # pour chaque génération g
                for p in range(Npop):
                    # 7a) Sélection aléatoire de 3 individus distincts
                    idxs = np.random.choice(Npop, 3, replace=False)
                    a, b, c = pop[idxs[0]], pop[idxs[1]], pop[idxs[2]]

                    # 7b) Recherche de l’individu “best” (plus petit coût)
                    best = pop[np.argmin(cf)]

                    # 7c) Mutation (current-to-best/1) :
                    y = c + F1 * (a - b) + F2 * (best - c)

                    # 7d) Crossover binomial :
                    #     mask[j] = True si on prend y[j], False si on garde pop[p,j]
                    mask = np.random.rand(n_params) < cr
                    #    S’assurer qu’au moins un paramètre est issu de y
                    if not mask.any():
                        mask[np.random.randint(n_params)] = True

                    #    Construction de l’individu candidat z
                    z = np.where(mask, y, pop[p])

                    # 7e) Remise aux bornes :
                    #     clip fait pop[:, j] = min(max(val, low[j]), high[j])
                    z = np.clip(z, lowers, uppers)

                    # 7f) Évaluation de z
                    cfz = self.sim.cost(z, keys, mode=mode)

                    # 7g) Sélection : si z est meilleur, on remplace l’individu p
                    if cfz < cf[p]:
                        pop[p], cf[p] = z, cfz

                # 7h) Convergence : on stocke la valeur minimale de cf pour la génération g
                conv[g] = cf.min()

        # ──────────────────────────────────────────────────────────
        # 8) RÉÉVALUATION FINALE
        # ──────────────────────────────────────────────────────────
        #    On recalcule cost() sur toute la population pour stabiliser best_final
        cf_final = np.array([self.sim.cost(pop[i], keys, mode=mode)
                            for i in range(Npop)])
        #    Choix de l’individu optimal après réévaluation
        best_final = pop[np.argmin(cf_final)]

        # 9) Générer le spectre pour le meilleur individu
        lam    = np.linspace(
            self.sim.sim_lambda_min.value,
            self.sim.sim_lambda_max.value,
            self.sim.sim_n_points.value
        )
        # injecter best_final dans la config cochée
        cfg = next(
            c for c in self.sim.all_configs
            if self.sim.config_checkboxes[c["config_name"]].value
        )
        for xi, k in zip(best_final, keys):
            cfg["geometry"]["geometry"][k] = float(xi)

        Rup, Rdown, _ = run_simulation_one_combo(
            lam,
            {"angle":0, "polarization":1},
            self.sim.sim_n_mod.value,
            cfg,
            self.json_combined_path
        )
        Rup   = np.asarray(Rup, float)
        Rdown = np.asarray(Rdown, float)


        # ──────────────────────────────────────────────────────────
        # 10) SAUVEGARDE HDF5 COMPLÈTE
        # ──────────────────────────────────────────────────────────
        # calcul de run_id
        run_id = f"budget{budget}_pop{Npop}"

        # sauvegarde HDF5
        save_optimization_hdf5(
            notebook_dir = str(BASE_NOTEBOOKS),
            run_id       = run_id,
            budget       = budget,
            Npop         = Npop,
            keys         = keys,
            lowers       = lowers,
            uppers       = uppers,
            conv         = conv,
            cf_final     = cf_final,
            best         = pop[np.argmin(cf)],
            best_final   = best_final,
            mode         = mode,
            lam          = lam,
            Rup          = Rup,
            Rdown        = Rdown
        )

        return conv, best_final




    def plot_optimization_results(self, _=None):
        """
        Lit le dernier fichier d’optimisation et trace :
        - le spectre RCWA de la meilleure structure,
        - la barre des paramètres optimisés,
        - la courbe de convergence,
        - la distribution finale des coûts (courbe de confiance).
        Trace en 2×2 :
        [ Convergence DE         | Distribution des coûts ]
        [ Paramètres optimisés   | Best config spectrum + tableau ]
        """
        
        files = list_optimization_files(str(self.summary_opt_dir))
        self.opt_file_dd.options = files
        
        
        h5file = self.opt_file_dd.value
        if h5file is None:
            raise RuntimeError("Aucun fichier d’optimisation trouvé.")
        data = read_optimization_hdf5(str(h5file))
        
        # 3) extraire les spectres si présents
        if 'spectra' in data:
            lam   = data['spectra']['wavelength']
            Rup   = data['spectra']['Rup']
            Rdown = data['spectra']['Rdown']
        else:
            lam = Rup = Rdown = None

        
        keys       = data['keys']
        best_vec   = data['best_final']
        conv       = data['conv']
        cf_final   = data['cf_final']

        # 3) Créer la grille 2x2
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        ax0, ax1, ax2, ax3 = axs.flat



        # --- Convergence ---
        ax0.plot(conv, marker='o')
        ax0.set_title("DE convergence curve")
        ax0.set_xlabel("Generation")
        ax0.set_ylabel("Cost function value")
        ax0.grid(True)

        # --- Distribution des coûts finaux ---
        ax1.plot(np.sort(cf_final), marker='.')
        ax1.set_title("Confidence curve: final costs sorted")
        ax1.set_xlabel("Sorted index")
        ax1.set_ylabel("Cost function value")
        ax1.grid(True)

        # --- Paramètres optimisés (bar chart) ---
        ax2.bar(keys, best_vec)
        ax2.set_title("Optimized parameters")
        ax2.set_xticklabels(keys, rotation=45, ha='right')
        ax2.set_ylabel("Optimized value")
        ax2.grid(True)



        Rup = np.asarray(Rup, float)
        Rdown = np.asarray(Rdown, float)

        # --- Best config spectrum + tableau ---
        if lam is not None:
            ax3.plot(lam, Rup, label="Rup")
            if Rdown is not None:
                ax3.plot(lam, Rdown, label="Rdown")
        ax3.set_title("Best config spectrum")
        ax3.set_xlabel("λ (nm)")
        ax3.set_ylabel("Reflectance")
        ax3.legend()
        ax3.grid(True)


        # Insérer un tableau sous le spectre
        # On convertit keys/best_vec en liste de lignes [[clé, valeur], ...]
        table_data = [[k, f"{v:.3g}"] for k, v in zip(keys, best_vec)]
        # Création du tableau, on utilise bbox pour le placer
        table = ax3.table(
            cellText=table_data,
            colLabels=["Parameter", "Value"],
            cellLoc='center',
            colLoc='center',
            bbox=[0.0, -0.6, 1.0, 0.4]  # [x, y, width, height] relatif à ax3
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        ax3.set_ylim(bottom=ax3.get_ylim()[0], top=ax3.get_ylim()[1])  # pour que le spectre ne soit pas rogné

        plt.tight_layout()
        plt.show()





def create_optimization_tab(sim_obj):
    """
    Wrapper de compatibilité : renvoie le widget de l’onglet.
    """
    tab = OptimizationTab(sim_obj)
    return tab
