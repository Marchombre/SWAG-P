import numpy as np
import matplotlib.pyplot as plt

# Import de la fonction de dispersion Brendel-Bormann
from Brendel_Bormann_Faddeeva import BrendelBormann

# Import des fonctions pour charger les matériaux
from MaterialsLoader import load_materials, get_material_params

# Import de la fonction de calcul de réflectance
from Function_reflectance_SWAG import reflectance

# 1) Chargement des données des matériaux depuis un fichier JSON ou autre source
materials_data = load_materials()

# 2) Récupération des paramètres pour l'argent (Ag) et l'or (Au)
f0_Ag, omega_p_Ag, Gamma0_Ag, f_Ag, omega_Ag, gamma_Ag, sigma_Ag = get_material_params("Ag", materials_data)
f0_Au, omega_p_Au, Gamma0_Au, f_Au, omega_Au, gamma_Au, sigma_Au = get_material_params("Au", materials_data)

# 3) Choix de la longueur d'onde à tester (en nm)
lambda_test = 800  # nm

# 4) Liste des valeurs de n_mod à tester pour étudier la convergence
n_mod_values = [5, 10, 20, 30, 50, 75, 100, 150, 200]

# 5) Définition du seuil de convergence pour la dérivée discrète de la réflectance
epsilon = 1e-4

# 6) Listes pour stocker les valeurs de réflectance et leurs dérivées
Rup_values = []
Rdown_values = []
dRup_dn_values = []
dRdown_dn_values = []

# 7) Variables de contrôle pour savoir si la convergence est atteinte
converged = False
optimal_n_mod = None

# 8) Boucle sur les valeurs de n_mod pour calculer la réflectance et vérifier la convergence
for i, n_mod in enumerate(n_mod_values):
    # Calcul de la permittivité de Ag et Au à la longueur d'onde lambda_test
    perm_Ag = BrendelBormann(lambda_test, f0_Ag, omega_p_Ag, Gamma0_Ag,
                             f_Ag, omega_Ag, gamma_Ag, sigma_Ag)
    perm_Au = BrendelBormann(lambda_test, f0_Au, omega_p_Au, Gamma0_Au,
                             f_Au, omega_Au, gamma_Au, sigma_Au)

    # Définition du dictionnaire de matériaux
    materials = {
        "perm_env": 1.0,                   # Milieu incident (air)
        "perm_dielec": 1.45 ** 2,          # Exemple d'un diélectrique (n = 1.45)
        "perm_sub": (1.5 + 1j * 0.1) ** 2,   # Substrat (n = 1.5, k = 0.1)
        "perm_reso": perm_Ag,              # Résonateur (Ag)
        "perm_metalliclayer": perm_Au,     # Couche métallique (Au)
        "perm_accroche": perm_Au           # Couche d'accroche (Au)
    }

    # Définition de la géométrie (en nm)
    geometry = {
        "thick_super": 200,
        "width_reso": 30,
        "thick_reso": 30,
        "thick_gap": 3,
        "thick_func": 1,
        "thick_mol": 2,
        "thick_metalliclayer": 10,
        "thick_sub": 200,
        "thick_accroche": 1,
        "period": 100.2153
    }

    # Paramètres d'onde
    wave = {
        "wavelength": lambda_test,  # nm
        "angle": 0,                 # Incidence normale
        "polarization": 1           # 1 pour TM, 0 pour TE
    }

    # 9) Calcul de la réflectance (Rup : vers le haut, Rdown : vers le bas)
    Rup, Rdown = reflectance(geometry, wave, materials, n_mod)

    # Stockage des résultats
    Rup_values.append(Rup)
    Rdown_values.append(Rdown)

    # 10) Vérification de la convergence via la dérivée discrète (variation relative par incrément de n_mod)
    if i > 0:
        dRup_dn = abs((Rup_values[i] - Rup_values[i-1]) /
                      (n_mod_values[i] - n_mod_values[i-1]))
        dRdown_dn = abs((Rdown_values[i] - Rdown_values[i-1]) /
                        (n_mod_values[i] - n_mod_values[i-1]))

        dRup_dn_values.append(dRup_dn)
        dRdown_dn_values.append(dRdown_dn)

        # Si les deux dérivées sont en dessous du seuil, la convergence est considérée comme atteinte
        if dRup_dn < epsilon and dRdown_dn < epsilon:
            converged = True
            optimal_n_mod = n_mod
            break  # Arrêt dès que la convergence est atteinte

# 11) Affichage du résultat de la convergence
if converged:
    print(f"Convergence atteinte pour n_mod = {optimal_n_mod}")
else:
    print("Attention : la convergence n'a pas été atteinte pour les n_mod testés.")

# # 12) Tracé de la réflectance Rup et Rdown en fonction de n_mod
# plt.figure(figsize=(8, 5))
# plt.plot(n_mod_values[:len(Rup_values)], Rup_values, marker="o", label="Rup")
# plt.plot(n_mod_values[:len(Rdown_values)], Rdown_values, marker="s", label="Rdown")
# plt.xlabel("Nombre de modes (n_mod)")
# plt.ylabel("Réflectance")
# plt.legend()
# plt.grid(True)
# plt.title(f"Convergence de Rup et Rdown à λ = {lambda_test} nm")
# plt.show()

# # 13) Tracé des dérivées discrètes (dRup/dn et dRdown/dn)
# plt.figure(figsize=(8, 5))
# plt.plot(n_mod_values[1:1+len(dRup_dn_values)], dRup_dn_values, marker="o", label="dRup/dn")
# plt.plot(n_mod_values[1:1+len(dRdown_dn_values)], dRdown_dn_values, marker="s", label="dRdown/dn")
# plt.axhline(epsilon, color="r", linestyle="--", label="Seuil epsilon")
# plt.xlabel("Nombre de modes (n_mod)")
# plt.ylabel("Dérivée discrète de la réflectance")
# plt.legend()
# plt.grid(True)
# plt.title(f"Analyse de convergence via la dérivée discrète à λ = {lambda_test} nm")
# plt.show()
