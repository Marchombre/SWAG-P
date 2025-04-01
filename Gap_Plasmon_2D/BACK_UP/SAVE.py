
def BrendelBormann_Faddeeva(lambda_test, f0, omega_p, Gamma0, f, omega, gamma, sigma, N):
    """
    Modèle Brendel & Bormann utilisant la fonction de Voigt pour modéliser des résonances lorentziennes
    élargies par une distribution gaussienne, avec l'approximation FFT pour la fonction de Faddeeva.
    
    Paramètres:
      - lambda_test : longueur d'onde en nm
      - f0, omega_p, Gamma0 : paramètres pour chi_f (en eV)
      - f, gamma, omega, sigma : listes ou numpy arrays pour chi_b (en eV)
      - N : paramètre numérique pour le calcul FFT dans la fonction faddeeva
      
    Retourne:
      - epsilon : permittivité complexe calculée
    """
    # Conversion de la longueur d'onde en énergie (eV) : E ≈ 1240 / λ (avec λ en nm)
    E = 1240.0 / lambda_test  # énergie en eV
    w = 6.62606957e-25 * 299792458 / 1.602176565e-19 / lambda_test
    #w = E  # on utilise w comme énergie en eV
    
    chi_b = 0.0 + 0.0j  # Initialisation
    
    f = np.array(f, dtype=float)
    omega = np.array(omega, dtype=float)
    gamma = np.array(gamma, dtype=float)
    sigma = np.array(sigma, dtype=float)
    
    for i in range(len(f)):
        a = (omega[i] - 1j * gamma[i]) / (np.sqrt(2) * sigma[i])
        x = (w - omega[i]) / (np.sqrt(2) * sigma[i])
        y = (w + omega[i]) / (np.sqrt(2) * sigma[i])
        prefactor = 1j * np.sqrt(np.pi) * f[i] * omega_p**2 / (2 * np.sqrt(2) * a * sigma[i])
        chi_b += prefactor * (faddeeva(x, N) + faddeeva(y, N))
    
    chi_f = - (omega_p**2) * f0 / (w * (w + 1j * Gamma0))
    epsilon = 1 + chi_f + chi_b
    return epsilon






### Correction

def BrendelBormann_Faddeeva(lambda_test, f0, omega_p, Gamma0, f, omega, gamma, sigma, N):
    """
    Modèle Brendel & Bormann utilisant la fonction de Voigt pour modéliser des résonances
    lorentziennes élargies par une distribution gaussienne, avec l'approximation FFT pour
    la fonction de Faddeeva. La formule a été corrigée pour correspondre exactement au code de ton amie.
    
    Paramètres:
      - lambda_test : longueur d'onde en nm
      - f0, omega_p, Gamma0 : paramètres pour chi_f (en eV)
      - f, omega, gamma, sigma : listes ou numpy arrays pour chi_b (en eV), où 'gamma' correspond à Gamma.
      - N : paramètre numérique pour le calcul FFT dans la fonction faddeeva (défaut = 64).
      
    Retourne:
      - epsilon : permittivité complexe calculée
    """
    # Conversion de la longueur d'onde en énergie (selon le même facteur que dans le code de ton amie)
    w = 6.62606957e-25 * 299792458 / 1.602176565e-19 / lambda_test
    chi_b = 0.0 + 0.0j  # Initialisation
    
    f = np.array(f, dtype=float)
    omega = np.array(omega, dtype=float)
    gamma = np.array(gamma, dtype=float)
    sigma = np.array(sigma, dtype=float)
    
    for i in range(len(f)):
        # Calcul de a tel que : a = sqrt(w*(w + 1j*gamma[i])) avec correction du signe
        a = np.sqrt(w * (w + 1j * gamma[i]))
        a = a * np.sign(np.real(a))
        # Calcul de x et y avec a (au lieu de w) pour correspondre au code de ton amie
        x = (a - omega[i]) / (np.sqrt(2) * sigma[i])
        y = (a + omega[i]) / (np.sqrt(2) * sigma[i])
        prefactor = 1j * np.sqrt(np.pi) * f[i] * omega_p**2 / (2 * np.sqrt(2) * a * sigma[i])
        chi_b += prefactor * (faddeeva(x, N) + faddeeva(y, N))
    
    # Contribution de Drude (électrons libres)
    chi_f = - (omega_p**2) * f0 / (w * (w + 1j * Gamma0))
    epsilon = 1 + chi_f + chi_b
    return epsilon



# CORRECTION v2

def BrendelBormann_Faddeeva(lambda_test, f0, omega_p, Gamma0, f, omega, gamma, sigma, N):
    """
    Modèle Brendel & Bormann utilisant la fonction de Voigt pour modéliser des résonances lorentziennes
    élargies par une distribution gaussienne, avec l'approximation FFT pour la fonction de Faddeeva.

    Paramètres:
      - lambda_test : longueur d'onde en nm
      - f0, omega_p, Gamma0 : paramètres pour chi_f (en eV)
      - f, omega, gamma, sigma : listes ou numpy arrays pour chi_b (en eV)
      - N : paramètre numérique pour le calcul FFT dans la fonction faddeeva

    Retourne:
      - epsilon : permittivité complexe calculée
    """
    # Convertir les paramètres en tableaux NumPy pour éviter les problèmes de type
    f = np.array(f, dtype=float)
    omega = np.array(omega, dtype=float)
    gamma = np.array(gamma, dtype=float)
    sigma = np.array(sigma, dtype=float)
    
    w_val = 6.62606957e-25 * 299792458 / (1.602176565e-19 * lambda_test)
    a = np.sqrt(w_val * (w_val + 1j * gamma))
    a = a * np.sign(np.real(a))

    x = (a - omega) / (np.sqrt(2) * sigma)
    y = (a + omega) / (np.sqrt(2) * sigma)

    chi_b = np.sum(1j * np.sqrt(np.pi) * f * omega_p**2 / (2 * np.sqrt(2) * a * sigma) *
                   (faddeeva(x, N) + faddeeva(y, N)))
    chi_f = - (omega_p**2) * f0 / (w_val * (w_val + 1j * Gamma0))

    epsilon = 1 + chi_f + chi_b
    
    return epsilon



def faddeeva(z, N):
    """
    Approximation de la fonction de Faddeeva en utilisant une méthode basée sur la FFT.
    
    Paramètres:
      - z : argument complexe (scalaire ou tableau)
      - N : nombre de termes utilisés dans le calcul (définit la précision)
      
    Retourne:
      - w(z) ≈ exp(-z^2) erfc(-i z)
    """
    # On s'assure que z est un tableau
    z_arr = np.atleast_1d(z)
    w_val = np.zeros(z_arr.shape, dtype=complex)
    
    idx = (np.real(z_arr) == 0)
    w_val[idx] = np.exp(-np.abs(z_arr[idx])**2) * (1 - erf(np.imag(z_arr[idx])))
    
    idx_non = ~idx
    idx1 = np.where(np.imag(z_arr[idx_non]) < 0)[0]
    if idx1.size > 0:
        z_arr[idx_non][idx1] = np.conj(z_arr[idx_non][idx1])
    
    M = 2 * N
    M2 = 2 * M
    k = np.arange(-M + 1, M)
    L = np.sqrt(N / np.sqrt(2))
    theta = k * np.pi / M
    t = L * np.tan(theta / 2)
    f_val = np.exp(-t**2) * (L**2 + t**2)
    f_val = np.append(0, f_val)
    a_coeff = np.real(np.fft.fft(np.fft.fftshift(f_val))) / M2
    a_coeff = np.flipud(a_coeff[1:N+1])
    
    Z = (L + 1.0j * z_arr[idx_non]) / (L - 1.0j * z_arr[idx_non])
    p = np.polyval(a_coeff, Z)
    w_val[idx_non] = 2 * p / (L - 1.0j * z_arr[idx_non])**2 + (1 / np.sqrt(np.pi)) / (L - 1.0j * z_arr[idx_non])
    
    if idx1.size > 0:
        overall_idx = np.where(idx_non)[0][idx1]
        w_val[overall_idx] = np.conj(2 * np.exp(-z_arr[overall_idx]**2) - w_val[overall_idx])
    
    if np.ndim(z) == 0:
        return w_val[0]
    else:
        return w_val
    




# ancienne version
# 
# 
def faddeeva(z, N):
    """
    Approximation de la fonction de Faddeeva utilisant une méthode basée sur la FFT.
    """
    z = np.array(z, copy=True, ndmin=1)
    w = np.zeros(z.size, dtype=complex)

    idx = (np.real(z) == 0)
    w[idx] = np.exp(np.abs(-z[idx]**2)) * (1 - erf(np.imag(z[idx])))
    idx = np.invert(idx)
    idx1 = idx + (np.imag(z) < 0)

    z[idx1] = np.conj(z[idx1])

    M = 2 * N
    M2 = 2 * M
    k = np.arange(-M + 1, M)
    L = np.sqrt(N / np.sqrt(2))

    theta = k * np.pi / M
    t = L * np.tan(theta / 2)
    f_val = np.exp(-t**2) * (L**2 + t**2)
    f_val = np.append(0, f_val)
    a_coeff = np.real(np.fft.fft(np.fft.fftshift(f_val))) / M2
    a_coeff = np.flipud(a_coeff[1:N+1])

    Z = (L + 1.0j * z[idx]) / (L - 1.0j * z[idx])
    p = np.polyval(a_coeff, Z)
    w[idx] = 2 * p / (L - 1.0j * z[idx])**2 + (1 / np.sqrt(np.pi)) / (L - 1.0j * z[idx])
    w[idx1] = np.conj(2 * np.exp(-z[idx1]**2) - w[idx1])
    if np.ndim(z) == 0:
        return w[0]
    else:
        return w    
      
      
      
      
      
      
      
      
      
      
# Function_reflectance_SWAG.py

import numpy as np
from Functions_RCWA import cascade, c_bas, interface, homogene, grating

### SWAG Reflectance Function

def reflectance(geometry, wave, materials, n_mod):  
    # Normalize geometric parameters by period
    period = geometry["period"]
    width_reso = geometry["width_reso"] / period
    thick_reso = geometry["thick_reso"] / period
    thick_gap  = geometry["thick_gap"] / period
    thick_func = geometry["thick_func"] / period
    thick_mol  = geometry["thick_mol"] / period
    # Remplacer thick_poly_lat par thick_diel
    thick_diel = geometry["thick_diel"] / period
    thick_metalliclayer = geometry["thick_metalliclayer"] / period
    thick_accroche = geometry["thick_accroche"] / period
    thick_sub  = geometry["thick_sub"] / period

    # Normalize wave parameters
    wavelength = wave["wavelength"] / period
    angle = wave["angle"]
    polarization = wave["polarization"]

    # Material permittivities (nouvelles clés issues de material_selector)
    perm_env = materials["perm_env"]
    perm_mol = materials["perm_mol"]
    perm_func = materials["perm_func"]
    # Pour les transitions verticales, on utilise perm_polymer pour le gap polymer
    perm_polymer = materials["perm_polymer"]
    # La nouvelle variable thick_diel s'associe à perm_diel pour la branche latérale
    perm_diel = materials["perm_diel"]
    perm_reso = materials["perm_reso"]
    perm_metalliclayer = materials["perm_metalliclayer"]
    perm_accroche = materials["perm_accroche"]
    perm_sub = materials["perm_sub"]

    # Position for grating calculations (utilisé pour les transitions)
    pos_reso = np.array([[width_reso, (1 - width_reso) / 2]])
    
    n = 2 * n_mod + 1
    k0 = 2 * np.pi / wavelength
    a0 = k0 * np.sin(angle * np.pi / 180)

    # ----------------- Vertical (Central) Branch -----------------
    Pup, Vup = homogene(k0, a0, polarization, perm_env, n)
    S_central = np.block([
        [np.zeros((n, n), dtype=complex), np.eye(n, dtype=complex)],
        [np.eye(n, dtype=complex), np.zeros((n, n), dtype=complex)]
    ])

    # Si thick_mol+thick_func est nul, on revient au comportement d'origine
    if (thick_mol + thick_func) == 0:
        P1, V1 = grating(k0, a0, polarization, perm_env, perm_reso, n, pos_reso)
        S_central = cascade(S_central, interface(Pup, P1))
        S_central = c_bas(S_central, V1, thick_reso)
    
        P2, V2 = grating(k0, a0, polarization, perm_env, perm_diel, n, pos_reso)
        S_central = cascade(S_central, interface(P1, P2))
        S_central = c_bas(S_central, V2, thick_gap)
    
        P3, V3 = homogene(k0, a0, polarization, perm_diel, n)
        S_central = cascade(S_central, interface(P2, P3))
        S_central = c_bas(S_central, V3, 0)
    else:
        if (thick_mol + thick_func) < thick_gap:
            # ----- CASE A (vertical) -----
            # Transition: Molecule -> gap polymer: grating from perm_mol to perm_polymer
            P1, V1 = grating(k0, a0, polarization, perm_mol, perm_polymer, n, pos_reso)
            S_central = cascade(S_central, interface(Pup, P1))
            S_central = c_bas(S_central, V1, thick_mol)
            
            # Transition: Functionalisation -> gap polymer: grating from perm_func to perm_polymer
            P2, V2 = grating(k0, a0, polarization, perm_func, perm_polymer, n, pos_reso)
            S_central = cascade(S_central, interface(P1, P2))
            S_central = c_bas(S_central, V2, thick_func)
            
            # Propagate through remaining gap polymer
            gap_remain = thick_gap - (thick_mol + thick_func)
            S_central = c_bas(S_central, V2, gap_remain)
            
            # Transition into Nanocube: homogeneous with perm_reso
            P3, V3 = homogene(k0, a0, polarization, perm_reso, n)
            S_central = cascade(S_central, interface(P2, P3))
            S_central = c_bas(S_central, V3, thick_reso)
        else:
            # ----- CASE B (vertical) -----
            # Transition: Molecule -> nanocube: grating from perm_mol to perm_reso
            P1, V1 = grating(k0, a0, polarization, perm_mol, perm_reso, n, pos_reso)
            S_central = cascade(S_central, interface(Pup, P1))
            S_central = c_bas(S_central, V1, thick_reso - (thick_mol - (thick_gap - thick_func)))
            
            # Transition: Functionalisation -> nanocube: grating from perm_func to perm_reso
            P2, V2 = grating(k0, a0, polarization, perm_func, perm_reso, n, pos_reso)
            S_central = cascade(S_central, interface(P1, P2))
            S_central = c_bas(S_central, V2, thick_mol - (thick_gap - thick_func))
            
            # Propagate through gap layer
            S_central = c_bas(S_central, V2, thick_gap)
            
            # Transition into Nanocube
            P3, V3 = homogene(k0, a0, polarization, perm_reso, n)
            S_central = cascade(S_central, interface(P2, P3))
            S_central = c_bas(S_central, V3, thick_reso)

    # ----------------- Lateral Branch -----------------
    # Lateral branch starts from a lateral environment with permittivity perm_diel.
    P_lat_env, V_lat_env = homogene(k0, a0, polarization, perm_diel, n)
    S_lateral = np.block([
        [np.zeros((n, n), dtype=complex), np.eye(n, dtype=complex)],
        [np.eye(n, dtype=complex), np.zeros((n, n), dtype=complex)]
    ])
    S_lateral = cascade(S_lateral, interface(P_lat_env, P_lat_env))  # Identity

    if (thick_mol + thick_func) == 0:
        # For consistency with the vertical branch when no molecule/func layers exist,
        # use the original lateral transitions from perm_env to perm_diel.
        P_lat1, V_lat1 = grating(k0, a0, polarization, perm_env, perm_reso, n, pos_reso)
        S_lateral = cascade(S_lateral, interface(P_lat_env, P_lat1))
        S_lateral = c_bas(S_lateral, V_lat1, 0)  # no propagation since thick_mol=0
        P_lat2, V_lat2 = grating(k0, a0, polarization, perm_env, perm_diel, n, pos_reso)
        S_lateral = cascade(S_lateral, interface(P_lat1, P_lat2))
        S_lateral = c_bas(S_lateral, V_lat2, thick_gap)
        P_dummy, V_dummy = homogene(k0, a0, polarization, perm_diel, n)
        S_lateral = cascade(S_lateral, interface(P_lat2, P_dummy))
        S_lateral = c_bas(S_lateral, V_dummy, 0)
    else:
        if (thick_mol + thick_func) < thick_gap:
            # ----- CASE A (lateral) -----
            # Transition: lateral Molecule -> lateral polymer: grating from perm_mol to perm_polymer
            P_lat1, V_lat1 = grating(k0, a0, polarization, perm_mol, perm_polymer, n, pos_reso)
            S_lateral = cascade(S_lateral, interface(P_lat_env, P_lat1))
            S_lateral = c_bas(S_lateral, V_lat1, thick_mol)
            
            # Transition: lateral Functionalisation -> lateral polymer: grating from perm_func to perm_polymer
            P_lat2, V_lat2 = grating(k0, a0, polarization, perm_func, perm_polymer, n, pos_reso)
            S_lateral = cascade(S_lateral, interface(P_lat1, P_lat2))
            S_lateral = c_bas(S_lateral, V_lat2, thick_func)
            
            # Propagate through lateral polymer layer; limited to not exceed the gap:
            lateral_layer_thickness = min(thick_diel, thick_gap - (thick_mol + thick_func))
            S_lateral = c_bas(S_lateral, V_lat2, lateral_layer_thickness)
        else:
            # ----- CASE B (lateral) -----
            # Transition: lateral Molecule -> lateral nanocube: grating from perm_mol to perm_reso
            P_lat1, V_lat1 = grating(k0, a0, polarization, perm_mol, perm_reso, n, pos_reso)
            S_lateral = cascade(S_lateral, interface(P_lat_env, P_lat1))
            S_lateral = c_bas(S_lateral, V_lat1, thick_mol)
            
            # Transition: lateral Functionalisation -> lateral nanocube: grating from perm_func to perm_reso
            P_lat2, V_lat2 = grating(k0, a0, polarization, perm_func, perm_reso, n, pos_reso)
            S_lateral = cascade(S_lateral, interface(P_lat1, P_lat2))
            S_lateral = c_bas(S_lateral, V_lat2, thick_func)
            
            # Propagate through lateral polymer layer (limited):
            lateral_layer_thickness = min(thick_diel, thick_gap - (thick_mol + thick_func))
            S_lateral = c_bas(S_lateral, V_lat2, lateral_layer_thickness)
    
    # ----------------- Combine Vertical and Lateral Responses -----------------
    Rup_central = abs(S_central[n_mod, n_mod])**2
    Rdown_central = abs(S_central[n + n_mod, n + n_mod])**2
    Rup_lateral = abs(S_lateral[n_mod, n_mod])**2
    Rdown_lateral = abs(S_lateral[n + n_mod, n + n_mod])**2

    weight_central = width_reso      # central zone fraction
    weight_lateral = 1 - width_reso  # lateral fraction

    Rup = Rup_central * weight_central + Rup_lateral * weight_lateral
    Rdown = Rdown_central * weight_central + Rdown_lateral * weight_lateral

    # ----------------- Common Vertical Layers -----------------
    # Metallic layer
    P_metal, V_metal = homogene(k0, a0, polarization, perm_metalliclayer, n)
    S_common = cascade(S_central, interface(P3, P_metal))
    S_common = c_bas(S_common, V_metal, thick_metalliclayer)
    
    # Accroche layer
    P_acc, V_acc = homogene(k0, a0, polarization, perm_accroche, n)
    S_common = cascade(S_common, interface(P_metal, P_acc))
    S_common = c_bas(S_common, V_acc, thick_accroche)
    
    # Substrate
    P_sub, V_sub = homogene(k0, a0, polarization, perm_sub, n)
    S_common = cascade(S_common, interface(P_acc, P_sub))
    S_common = c_bas(S_common, V_sub, thick_sub)
    
    # Apply common layers uniformly
    Rup *= abs(S_common[n_mod, n_mod])**2
    Rdown *= abs(S_common[n + n_mod, n + n_mod])**2

    return Rup, Rdown








# Plotter n_mine VS n_Pauline for double checking

import pandas as pd
import matplotlib.pyplot as plt

# Définir la plage de longueurs d'onde (en nm)
lambda_range = np.linspace(450, 1000, 200)


# Charger le fichier JSON
with open(json_combined_path, "r", encoding="utf-8") as f:
    materials_data = json.load(f)

# --- Méthode Pauline ---
from swag_ITO_thickAu import nk_ITO, epsAgbb, epsAubb, epsAlbb

# --- Méthode via build_material_configuration_dynamic ---
from Material_Configuration import build_material_configuration_dynamic

# Pour créer un DataFrame de configuration pour un matériau standard ou custom,
# on définit une fonction utilitaire qui retourne un DataFrame d'une seule ligne.
def create_df_for_material(role, material_type, material_value):
    """
    role: chaîne utilisée comme clé dans la configuration dynamique
    material_type: "Standard" ou "Custom"
    material_value: pour Standard, le nom du matériau (ex: "Au", "Ag", "ITO");
                      pour Custom, l'expression (ex: "1.45**2")
    """
    if material_type.lower() == "custom":
        mat_dict = {"type": "Custom", "expression": material_value}
    else:
        mat_dict = {"type": "Standard", "material": material_value}
    # Construire un DataFrame d'une seule ligne
    return pd.DataFrame({"key": [role], "material": [mat_dict]})

# --- Calculs et comparaison pour chaque matériau ---

results = {}  # Dictionnaire pour stocker les 2 méthodes pour chaque matériau

# 1) ITO
# Pauline: on utilise nk_ITO et on calcule ε = (nk_ITO)²
eps_Pauline_ITO = np.array([nk_ITO(lam)[2]**2 for lam in lambda_range])
# Modules build_material_configuration_dynamic : pour ITO, avec un rôle par exemple "perm_sub"
df_ITO = create_df_for_material("perm_sub", "Standard", "ITO")
eps_mine_ITO = np.array([build_material_configuration_dynamic(df_ITO, lam, json_combined_path)["perm_sub"]
                         for lam in lambda_range])
results["ITO"] = (eps_Pauline_ITO, eps_mine_ITO)

# 2) Ag
eps_Pauline_Ag = np.array([epsAgbb(lam) for lam in lambda_range])
# Our: on crée une configuration avec rôle "perm_reso" et matériau "Ag"
df_Ag = create_df_for_material("perm_reso", "Standard", "Silver")
eps_mine_Ag = np.array([build_material_configuration_dynamic(df_Ag, lam, json_combined_path)["perm_reso"]
                        for lam in lambda_range])
results["Silver"] = (eps_Pauline_Ag, eps_mine_Ag)

# 3) Au
eps_Pauline_Au = np.array([epsAubb(lam) for lam in lambda_range])
# Our: on crée une configuration avec rôle "perm_metalliclayer" et matériau "Au"
df_Au = create_df_for_material("perm_metalliclayer", "Standard", "Gold")
eps_mine_Au = np.array([build_material_configuration_dynamic(df_Au, lam, json_combined_path)["perm_metalliclayer"]
                        for lam in lambda_range])
results["Gold"] = (eps_Pauline_Au, eps_mine_Au)

# 4) Al
eps_Pauline_Al = np.array([epsAlbb(lam) for lam in lambda_range])
# Our: on crée une configuration avec rôle "perm_metalliclayer" et matériau "Al"
df_Al = create_df_for_material("perm_accroche", "Standard", "Aluminium")
eps_mine_Al = np.array([build_material_configuration_dynamic(df_Al, lam, json_combined_path)["perm_accroche"]
                        for lam in lambda_range])
results["Aluminium"] = (eps_Pauline_Al, eps_mine_Al)

# --- Tracé des comparaisons ---
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
material_names = ["ITO", "Silver", "Gold", "Aluminium"]

for ax, mat in zip(axs.flatten(), material_names):
    eps_f, eps_o = results[mat]
    # Tracer la partie réelle
    ax.plot(lambda_range, np.real(np.sqrt(eps_f)), 'b--', label="Re(ε) - Pauline")
    ax.plot(lambda_range, np.real(np.sqrt(eps_o)), 'r-', label="Re(ε) - mine")
    # Tracer la partie imaginaire
    ax.plot(lambda_range, np.imag(np.sqrt(eps_f)), 'g--', label="Im(ε) - Pauline")
    ax.plot(lambda_range, np.imag(np.sqrt(eps_o)), 'k-', label="Im(ε) - mine")
    ax.set_title(f"{mat}")
    ax.set_xlabel("Longueur d'onde (nm)")
    ax.set_ylabel("ε")
    ax.legend(fontsize=8)
    ax.grid(True)

plt.suptitle("ε(λ) via build_material_configuration_dynamic double checking with Pauline's works", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
