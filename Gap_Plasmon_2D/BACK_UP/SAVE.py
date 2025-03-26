
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
