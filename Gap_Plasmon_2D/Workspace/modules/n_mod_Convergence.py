import numpy as np
import matplotlib.pyplot as plt

# Import the Brendel-Bormann dispersion function
from Brendel_Bormann_Faddeeva import BrendelBormann

# Import functions to load materials data
from MaterialsLoader import load_materials, get_material_params

# Import the reflectance calculation function
from Function_reflectance_SWAG import reflectance

# 1) Load materials data from a JSON file or other source
materials_data = load_materials()

# 2) Retrieve parameters for silver (Ag) and gold (Au)
f0_Ag, omega_p_Ag, Gamma0_Ag, f_Ag, omega_Ag, gamma_Ag, sigma_Ag = get_material_params("Ag", materials_data)
f0_Au, omega_p_Au, Gamma0_Au, f_Au, omega_Au, gamma_Au, sigma_Au = get_material_params("Au", materials_data)

# 3) Choose the wavelength to test (in nm)
lambda_test = 800  # nm

# 4) List of n_mod values to test for studying convergence
n_mod_values = [5, 10, 20, 30, 50, 75, 100, 150, 200]

# 5) Define the convergence threshold for the discrete derivative of reflectance
epsilon = 1e-4

# 6) Lists to store the reflectance values and their derivatives
Rup_values = []
Rdown_values = []
dRup_dn_values = []
dRdown_dn_values = []

# 7) Control variables to check if convergence is reached
converged = False
optimal_n_mod = None

# 8) Loop over the n_mod values to calculate reflectance and check for convergence
for i, n_mod in enumerate(n_mod_values):
    # Calculate the permittivity for Ag and Au at the test wavelength
    perm_Ag = BrendelBormann(lambda_test, f0_Ag, omega_p_Ag, Gamma0_Ag,
                             f_Ag, omega_Ag, gamma_Ag, sigma_Ag)
    perm_Au = BrendelBormann(lambda_test, f0_Au, omega_p_Au, Gamma0_Au,
                             f_Au, omega_Au, gamma_Au, sigma_Au)

    # Define the materials dictionary
    materials = {
        "perm_env": 1.0,                   # Incident medium (air)
        "perm_dielec": 1.45 ** 2,          # Example dielectric (n = 1.45)
        "perm_sub": (1.5 + 1j * 0.1) ** 2,   # Substrate (n = 1.5, k = 0.1)
        "perm_reso": perm_Ag,              # Resonator (Ag)
        "perm_metalliclayer": perm_Au,     # Metallic layer (Au)
        "perm_accroche": perm_Au           # Adhesion layer (Au)
    }

    # Define the geometry (in nm)
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

    # Wave parameters
    wave = {
        "wavelength": lambda_test,  # nm
        "angle": 0,                 # Normal incidence
        "polarization": 1           # 1 for TM, 0 for TE
    }

    # 9) Calculate reflectance (Rup: upward, Rdown: downward)
    Rup, Rdown = reflectance(geometry, wave, materials, n_mod)

    # Store the results
    Rup_values.append(Rup)
    Rdown_values.append(Rdown)

    # 10) Check convergence using the discrete derivative (relative variation per n_mod increment)
    if i > 0:
        dRup_dn = abs((Rup_values[i] - Rup_values[i-1]) /
                      (n_mod_values[i] - n_mod_values[i-1]))
        dRdown_dn = abs((Rdown_values[i] - Rdown_values[i-1]) /
                        (n_mod_values[i] - n_mod_values[i-1]))

        dRup_dn_values.append(dRup_dn)
        dRdown_dn_values.append(dRdown_dn)

        # If both derivatives are below the threshold, convergence is considered achieved
        if dRup_dn < epsilon and dRdown_dn < epsilon:
            converged = True
            optimal_n_mod = n_mod
            break  # Stop as soon as convergence is reached

# 11) Display the convergence result
if converged:
    print(f"Convergence reached for n_mod = {optimal_n_mod}")
else:
    print("Warning: Convergence was not reached for the tested n_mod values.")

# # 12) Plot Rup and Rdown as a function of n_mod
# plt.figure(figsize=(8, 5))
# plt.plot(n_mod_values[:len(Rup_values)], Rup_values, marker="o", label="Rup")
# plt.plot(n_mod_values[:len(Rdown_values)], Rdown_values, marker="s", label="Rdown")
# plt.xlabel("Number of modes (n_mod)")
# plt.ylabel("Reflectance")
# plt.legend()
# plt.grid(True)
# plt.title(f"Convergence of Rup and Rdown at λ = {lambda_test} nm")
# plt.show()

# # 13) Plot the discrete derivatives (dRup/dn and dRdown/dn)
# plt.figure(figsize=(8, 5))
# plt.plot(n_mod_values[1:1+len(dRup_dn_values)], dRup_dn_values, marker="o", label="dRup/dn")
# plt.plot(n_mod_values[1:1+len(dRdown_dn_values)], dRdown_dn_values, marker="s", label="dRdown/dn")
# plt.axhline(epsilon, color="r", linestyle="--", label="Epsilon threshold")
# plt.xlabel("Number of modes (n_mod)")
# plt.ylabel("Discrete derivative of reflectance")
# plt.legend()
# plt.grid(True)
# plt.title(f"Convergence analysis via discrete derivative at λ = {lambda_test} nm")
# plt.show()
