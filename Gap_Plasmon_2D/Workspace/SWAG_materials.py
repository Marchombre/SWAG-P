import PyMoosh as pm
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.linalg import toeplitz, inv
import json

### Chargement des matériaux depuis le fichier JSON
with open('/home/chardon-grossard/Bureau/SWAG-P/Gap_Plasmon_2D/Workspace/data/material_data.json') as file:
    data = json.load(file)

def get_n_k(material_name, lam):
    """
    Récupère l'indice de réfraction complexe (n + i*k) pour un matériau donné à une longueur d'onde donnée.
    
    Paramètres :
      - material_name : Nom du matériau dans le JSON (ex: "BK7", "Water", "SiA").
      - lam : Longueur d'onde en nm.
      
    Retourne :
      - n (partie réelle de l'indice de réfraction)
      - k (partie imaginaire, absorption)
    """
    if material_name not in data:
        raise ValueError(f"Le matériau '{material_name}' n'est pas dans la base de données.")

    material = data[material_name]

    if material["model"] == "ExpData":
        wl = np.array(material["wavelength_list"])
        epsilon_real = np.array(material["permittivities"])
        epsilon_imag = np.array(material.get("permittivities_imag", np.zeros_like(epsilon_real)))

        if lam < wl[0] or lam > wl[-1]:
            raise ValueError(f"La longueur d'onde {lam} nm est hors de l'intervalle [{wl[0]}, {wl[-1]}] nm pour {material_name}.")

        eps_r = np.interp(lam, wl, epsilon_real)
        eps_i = np.interp(lam, wl, epsilon_imag)
        eps_complex = eps_r + 1.0j * eps_i
        n_complex = np.sqrt(eps_complex)
        n = np.real(n_complex)
        k = np.imag(n_complex)
        return n, k
    else:
        raise ValueError(f"Le modèle '{material['model']}' pour {material_name} n'est pas supporté.")

# Test de get_n_k
material_test = "Water"
lambda_test = 500  # nm
try:
    n_value, k_value = get_n_k(material_test, lambda_test)
    print(f"Pour {material_test} à {lambda_test} nm :")
    print(f"  n = {n_value:.6f}")
    print(f"  k = {k_value:.6f}")
except ValueError as e:
    print(e)

### Fonctions pour le calcul de la permittivité par les modèles de dispersion
def faddeeva(z, N):
    """Calcul approximatif de la fonction de Faddeeva."""
    w = np.zeros(z.size, dtype=complex)
    idx = np.real(z) == 0
    w[idx] = np.exp(-np.abs(z[idx]**2)) * (1 - erf(np.imag(z[idx])))
    idx = np.invert(idx)
    idx1 = idx + (np.imag(z) < 0)
    z[idx1] = np.conj(z[idx1])
    M = 2 * N
    M2 = 2 * M
    k = np.arange(-M + 1, M)
    L = np.sqrt(N / np.sqrt(2))
    theta = k * np.pi / M
    t = L * np.tan(theta / 2)
    f = np.exp(-t**2) * (L**2 + t**2)
    f = np.append(0, f)
    a = np.real(np.fft.fft(np.fft.fftshift(f))) / M2
    a = np.flipud(a[1:N+1])
    Z = (L + 1.0j * z[idx]) / (L - 1.0j * z[idx])
    p = np.polyval(a, Z)
    w[idx] = 2 * p / (L - 1.0j * z[idx])**2 + (1 / np.sqrt(np.pi)) / (L - 1.0j * z[idx])
    w[idx1] = np.conj(2 * np.exp(-z[idx1]**2) - w[idx1])
    return w

def epsAubb(lam):
    """Permet de calculer la permittivité de l'or (Au) via le modèle Brendel-Bormann."""
    w = 6.62606957e-25 * 299792458 / 1.602176565e-19 / lam
    f0 = 0.770; Gamma0 = 0.050; omega_p = 9.03
    f = np.array([0.054, 0.050, 0.312, 0.719, 1.648])
    Gamma = np.array([0.074, 0.035, 0.083, 0.125, 0.179])
    omega = np.array([0.218, 2.885, 4.069, 6.137, 27.97])
    sigma = np.array([0.742, 0.349, 0.830, 1.246, 1.795])
    a = np.sqrt(w * (w + 1.0j * Gamma))
    a = a * np.sign(np.real(a))
    x = (a - omega) / (np.sqrt(2) * sigma)
    y = (a + omega) / (np.sqrt(2) * sigma)
    epsilon = 1 - omega_p**2 * f0 / (w * (w + 1.0j * Gamma0)) \
              + np.sum(1.0j * np.sqrt(np.pi) * f * omega_p**2 / (2 * np.sqrt(2) * a * sigma) * (faddeeva(x, 64) + faddeeva(y, 64)))
    return epsilon

def epsAgbb(lam):
    """Permet de calculer la permittivité de l'argent (Ag) via le modèle Brendel-Bormann."""
    w = 6.62606957e-25 * 299792458 / 1.602176565e-19 / lam
    f0 = 0.821; Gamma0 = 0.049; omega_p = 9.01
    f = np.array([0.050, 0.133, 0.051, 0.467, 4.000])
    Gamma = np.array([0.189, 0.067, 0.019, 0.117, 0.052])
    omega = np.array([2.025, 5.185, 4.343, 9.809, 18.56])
    sigma = np.array([1.894, 0.665, 0.189, 1.170, 0.516])
    a = np.sqrt(w * (w + 1.0j * Gamma))
    a = a * np.sign(np.real(a))
    x = (a - omega) / (np.sqrt(2) * sigma)
    y = (a + omega) / (np.sqrt(2) * sigma)
    aha = 1.0j * np.sqrt(np.pi) * f * omega_p**2 / (2 * np.sqrt(2) * a * sigma) * (faddeeva(x, 64) + faddeeva(y, 64))
    epsilon = 1 - omega_p**2 * f0 / (w * (w + 1.0j * Gamma0)) + np.sum(aha)
    return epsilon

def epsCrbb(lam):
    """Permet de calculer la permittivité du chrome (Cr) via le modèle Brendel-Bormann."""
    w = 6.62606957e-25 * 299792458 / 1.602176565e-19 / lam
    f0 = 0.154; Gamma0 = 0.048; omega_p = 10.75
    f = np.array([0.338, 0.261, 0.817, 0.105])
    Gamma = np.array([4.256, 3.957, 2.218, 6.983])
    omega = np.array([0.281, 0.584, 1.919, 6.997])
    sigma = np.array([0.115, 0.252, 0.225, 4.903])
    a = np.sqrt(w * (w + 1.0j * Gamma))
    a = a * np.sign(np.real(a))
    x = (a - omega) / (np.sqrt(2) * sigma)
    y = (a + omega) / (np.sqrt(2) * sigma)
    aha = 1.0j * np.sqrt(np.pi) * f * omega_p**2 / (2 * np.sqrt(2) * a * sigma) * (faddeeva(x, 64) + faddeeva(y, 64))
    epsilon = 1 - omega_p**2 * f0 / (w * (w + 1.0j * Gamma0)) + np.sum(aha)
    return epsilon

def epsAlbb(lam):
    """Permet de calculer la permittivité de l'aluminium (Al) via le modèle Brendel-Bormann."""
    w = 6.62606957e-25 * 299792458 / 1.602176565e-19 / lam
    f0 = 0.526; Gamma0 = 0.047; omega_p = 14.98
    f = np.array([0.213, 0.060, 0.182, 0.014])
    Gamma = np.array([0.312, 0.315, 1.587, 2.145])
    omega = np.array([0.163, 1.561, 1.827, 4.495])
    sigma = np.array([0.013, 0.042, 0.256, 1.735])
    a = np.sqrt(w * (w + 1.0j * Gamma))
    a = a * np.sign(np.real(a))
    x = (a - omega) / (np.sqrt(2) * sigma)
    y = (a + omega) / (np.sqrt(2) * sigma)
    aha = 1.0j * np.sqrt(np.pi) * f * omega_p**2 / (2 * np.sqrt(2) * a * sigma) * (faddeeva(x, 64) + faddeeva(y, 64))
    epsilon = 1 - omega_p**2 * f0 / (w * (w + 1.0j * Gamma0)) + np.sum(aha)
    return epsilon

def epsNibb(lam):
    """Permet de calculer la permittivité du nickel (Ni) via le modèle Brendel-Bormann."""
    w = 6.62606957e-25 * 299792458 / 1.602176565e-19 / lam
    f0 = 0.083; Gamma0 = 0.022; omega_p = 15.92
    f = np.array([0.357, 0.039, 0.127, 0.654])
    Gamma = np.array([2.820, 0.120, 1.822, 6.637])
    omega = np.array([0.317, 1.059, 4.583, 8.825])
    sigma = np.array([0.606, 1.454, 0.379, 0.510])
    a = np.sqrt(w * (w + 1.0j * Gamma))
    a = a * np.sign(np.real(a))
    x = (a - omega) / (np.sqrt(2) * sigma)
    y = (a + omega) / (np.sqrt(2) * sigma)
    aha = 1.0j * np.sqrt(np.pi) * f * omega_p**2 / (2 * np.sqrt(2) * a * sigma) * (faddeeva(x, 64) + faddeeva(y, 64))
    epsilon = 1 - omega_p**2 * f0 / (w * (w + 1.0j * Gamma0)) + np.sum(aha)
    return epsilon

### Fonctions RCWA

def cascade(T, U):
    '''Cascade deux matrices de diffusion T et U (dimensions 2n x 2n).'''
    n = int(T.shape[1] / 2)
    J = np.linalg.inv(np.eye(n) - np.matmul(U[0:n, 0:n], T[n:2*n, n:2*n]))
    K = np.linalg.inv(np.eye(n) - np.matmul(T[n:2*n, n:2*n], U[0:n, 0:n]))
    S = np.block([
        [T[0:n, 0:n] + np.matmul(np.matmul(np.matmul(T[0:n, n:2*n], J), U[0:n, 0:n]), T[n:2*n, 0:n]),
         np.matmul(np.matmul(T[0:n, n:2*n], J), U[0:n, n:2*n])],
        [np.matmul(np.matmul(U[n:2*n, 0:n], K), T[n:2*n, 0:n]),
         U[n:2*n, n:2*n] + np.matmul(np.matmul(np.matmul(U[n:2*n, 0:n], K), T[n:2*n, n:2*n]), U[0:n, n:2*n])]
    ])
    return S

def c_bas(A, V, h):
    '''Cascade la matrice de diffusion A avec la propagation d'une couche de hauteur h.'''
    n = int(A.shape[1] / 2)
    D = np.diag(np.exp(1j * V * h))
    S = np.block([
        [A[0:n, 0:n], np.matmul(A[0:n, n:2*n], D)],
        [np.matmul(D, A[n:2*n, 0:n]), np.matmul(np.matmul(D, A[n:2*n, n:2*n]), D)]
    ])
    return S

def c_haut(A, valp, h):
    n = int(A[0].size / 2)
    D = np.diag(np.exp(1j * valp * h))
    S11 = np.dot(D, np.dot(A[0:n, 0:n], D))
    S12 = np.dot(D, A[0:n, n:2*n])
    S21 = np.dot(A[n:2*n, 0:n], D)
    S22 = A[n:2*n, n:2*n]
    S1 = np.append(S11, S12, axis=1)
    S2 = np.append(S21, S22, axis=1)
    S = np.append(S1, S2, axis=0)
    return S    

def intermediaire(T, U):
    n = int(T.shape[0] / 2)
    H = np.linalg.inv(np.eye(n) - np.matmul(U[0:n, 0:n], T[n:2*n, n:2*n]))
    K = np.linalg.inv(np.eye(n) - np.matmul(T[n:2*n, n:2*n], U[0:n, 0:n]))
    a = np.matmul(K, T[n:2*n, 0:n])
    b = np.matmul(K, np.matmul(T[n:2*n, n:2*n], U[0:n, n:2*n]))
    c = np.matmul(H, np.matmul(U[0:n, 0:n], T[n:2*n, 0:n]))
    d = np.matmul(H, U[0:n, n:2*n])
    S = np.block([[a, b], [c, d]])
    return S

def couche(valp, h):
    n = len(valp)
    AA = np.diag(np.exp(1j * valp * h))
    C = np.block([[np.zeros((n, n)), AA], [AA, np.zeros((n, n))]])
    return C

def step(a, b, w, x0, n):
    '''Calcule la matrice de Toeplitz générée par la série de Fourier d'une fonction en escalier.'''
    tmp = np.exp(-2 * 1j * np.pi * (x0 + w / 2) * np.arange(0, n)) * np.sinc(w * np.arange(0, n)) * w
    l = np.conj(tmp) * (b - a)
    m = tmp * (b - a)
    l[0] = l[0] + a
    m[0] = l[0]
    T = toeplitz(l, m)
    return T

def grating(k0, a0, pol, e1, e2, n, blocs):
    '''Génère la matrice de Fourier pour un réseau constitué de blocs de matériau e2 dans un milieu e1.'''
    n_blocs = blocs.shape[0]
    nmod = int(n / 2)
    M1 = e1 * np.eye(n, dtype=complex)
    M2 = 1 / e1 * np.eye(n, dtype=complex)
    for k in range(n_blocs):
        M1 = M1 + step(0, e2 - e1, blocs[k, 0], blocs[k, 1], n)
        M2 = M2 + step(0, 1 / e2 - 1 / e1, blocs[k, 0], blocs[k, 1], n)
    alpha = np.diag(a0 + 2 * np.pi * np.arange(-nmod, nmod + 1)) + 0j
    if pol == 0:
        M = alpha * alpha - k0**2 * M1
        L, E = np.linalg.eig(M)
        L = np.sqrt(-L + 0j)
        L = (1 - 2 * (np.imag(L) < -1e-15)) * L
        P = np.block([[E], [np.matmul(E, np.diag(L))]])
    else:
        T_inv = np.linalg.inv(M2)
        M = np.matmul(np.matmul(np.matmul(T_inv, alpha), np.linalg.inv(M1)), alpha) - k0**2 * T_inv
        L, E = np.linalg.eig(M)
        L = np.sqrt(-L + 0j)
        L = (1 - 2 * (np.imag(L) < -1e-15)) * L
        P = np.block([[E], [np.matmul(np.matmul(M2, E), np.diag(L))]])
    return P, L

def homogene(k0, a0, pol, epsilon, n):
    nmod = int(n / 2)
    valp = np.sqrt(epsilon * k0**2 - (a0 + 2 * np.pi * np.arange(-nmod, nmod + 1))**2 + 0j)
    valp = valp * (1 - 2 * (valp < 0))
    P = np.block([[np.eye(n, dtype=complex)], [np.diag(valp * (pol / epsilon + (1 - pol)))]])
    return P, valp

def interface(P, Q):
    '''Calcule la matrice de diffusion d'une interface à partir de P et Q.'''
    n = int(P.shape[1])
    A = np.block([[P[0:n, 0:n], -Q[0:n, 0:n]],
                  [P[n:2*n, 0:n],  Q[n:2*n, 0:n]]])
    B = np.block([[-P[0:n, 0:n], Q[0:n, 0:n]],
                  [ P[n:2*n, 0:n], Q[n:2*n, 0:n]]])
    S = np.matmul(np.linalg.inv(A), B)
    return S

def HErmes(T, U, V, P, Amp, ny, h, a0):
    n = int(np.shape(T)[0] / 2)
    nmod = int((n - 1) / 2)
    nx = n
    X = np.matmul(intermediaire(T, cascade(couche(V, h), U)), Amp.reshape(Amp.size, 1))
    D = X[0:n]
    X = np.matmul(intermediaire(cascade(T, couche(V, h)), U), Amp.reshape(Amp.size, 1))
    E = X[n:2*n]
    M = np.zeros((ny, nx - 1), dtype=complex)
    for k in range(ny):
        y = h / ny * (k + 1)
        Fourier = np.matmul(P, np.matmul(np.diag(np.exp(1j * V * y)), D) + np.matmul(np.diag(np.exp(1j * V * (h - y))), E))
        MM = np.fft.ifftshift(Fourier[0:len(Fourier) - 1])
        M[k, :] = MM.reshape(len(MM))
    M = np.conj(np.fft.ifft(np.conj(M).T, axis=0)).T * n
    x, y = np.meshgrid(np.linspace(0, 1, nx - 1), np.linspace(0, 1, ny))
    M = M * np.exp(1j * a0 * x)
    return M

### Fonction reflectance SWAG

def reflectance(geometry, wave, materials, n_mod):
    period = geometry["period"]
    width_reso = geometry["width_reso"] / period
    thick_reso = geometry["thick_reso"] / period
    thick_gap = geometry["thick_gap"] / period
    thick_func = geometry["thick_func"] / period
    thick_mol = geometry["thick_mol"] / period
    thick_metalliclayer = geometry["thick_metalliclayer"] / period
    thick_sub = geometry["thick_sub"] / period
    thick_accroche = geometry["thick_accroche"] / period 

    wavelength = wave["wavelength"] / period
    angle = wave["angle"]
    polarization = wave["polarization"]

    perm_env = materials["perm_env"]
    perm_dielec = materials["perm_dielec"]
    perm_sub = materials["perm_sub"]
    perm_reso = materials["perm_reso"]
    perm_metalliclayer = materials["perm_metalliclayer"]
    perm_accroche = materials["perm_accroche"]

    pos_reso = np.array([[width_reso, (1 - width_reso) / 2]])
    n = 2 * n_mod + 1
    k0 = 2 * np.pi / wavelength
    a0 = k0 * np.sin(angle * np.pi / 180)
    Pup, Vup = homogene(k0, a0, polarization, perm_env, n)
    S = np.block([[np.zeros([n, n], dtype=complex), np.eye(n, dtype=complex)],
                  [np.eye(n, dtype=complex), np.zeros([n, n], dtype=complex)]])

    if thick_mol < (thick_gap - thick_func):
        P1, V1 = grating(k0, a0, polarization, perm_env, perm_reso, n, pos_reso)
        S = cascade(S, interface(Pup, P1))
        S = c_bas(S, V1, thick_reso)
    
        P2, V2 = grating(k0, a0, polarization, perm_env, perm_dielec, n, pos_reso)
        S = cascade(S, interface(P1, P2))
        S = c_bas(S, V2, thick_gap - (thick_mol + thick_func))
    
        P3, V3 = homogene(k0, a0, polarization, perm_dielec, n)
        S = cascade(S, interface(P2, P3))
        S = c_bas(S, V3, thick_mol + thick_func)
    
    else:
        P1, V1 = grating(k0, a0, polarization, perm_env, perm_reso, n, pos_reso)
        S = cascade(S, interface(Pup, P1))
        S = c_bas(S, V1, thick_reso - (thick_mol - (thick_gap - thick_func)))
    
        P2, V2 = grating(k0, a0, polarization, perm_dielec, perm_reso, n, pos_reso)
        S = cascade(S, interface(P1, P2))
        S = c_bas(S, V2, thick_mol - (thick_gap - thick_func))
    
        P3, V3 = homogene(k0, a0, polarization, perm_dielec, n)
        S = cascade(S, interface(P2, P3))
        S = c_bas(S, V3, thick_gap)
    
    Pmetalliclayer, Vmetalliclayer = homogene(k0, a0, polarization, perm_metalliclayer, n)
    S = cascade(S, interface(P3, Pmetalliclayer))
    S = c_bas(S, Vmetalliclayer, thick_metalliclayer)
    
    Pacc, Vacc = homogene(k0, a0, polarization, perm_accroche, n)
    S = cascade(S, interface(Pmetalliclayer, Pacc))
    S = c_bas(S, Vacc, thick_accroche)
    
    Pdown, Vdown = homogene(k0, a0, polarization, perm_sub, n)
    S = cascade(S, interface(Pacc, Pdown))
    S = c_bas(S, Vdown, thick_sub)
    
    Rup = abs(S[n_mod, n_mod]) ** 2 
    Rdown = abs(S[n + n_mod, n + n_mod]) ** 2 
    return Rup, Rdown

### Définition des paramètres de la structure SWAG
thick_super = 200
width_reso = 30      # largeur du cube
thick_reso = width_reso  # hauteur du cube
thick_gap = 3        # hauteur du diélectrique en dessous du cube
thick_func = 1       # présent partout
thick_mol = 2        # épaisseur si molécules détectées
thick_metal = 10     # hauteur de la couche métallique (or)
thick_accroche = 1   # couche d'accroche 
period = 100.2153
thick_sub = 200

angle = 0
polarization = 1     # 1 pour TM, 0 pour TE

## Paramètres des matériaux de base
perm_env = 1.0
perm_dielec = 1.45 ** 2  # matériau spacer
# Pour l'indice du substrat, on récupère n et k pour "SiA" et on calcule ε = (n + i*k)**2
n_value, k_value = get_n_k(material_test, lambda_test)
perm_sub_base = (n_value + 1j * k_value) ** 2

perm_Ag = epsAgbb(lambda_test)  # argent
perm_Au = epsAubb(lambda_test)  # or
perm_Cr = epsCrbb(lambda_test)   # chrome (si besoin)

materials = {
    "perm_env": perm_env,
    "perm_dielec": perm_dielec,
    "perm_sub": perm_sub_base,
    "perm_reso": perm_Ag,
    "perm_metalliclayer": perm_Au,
    "perm_accroche": perm_Au
}

wave = {"wavelength": lambda_test, "angle": angle, "polarization": polarization}

n_mod = 100 
geometry = {
    "thick_super": thick_super,
    "width_reso": width_reso,
    "thick_reso": thick_reso,
    "thick_gap": thick_gap,
    "thick_func": thick_func,
    "thick_mol": thick_mol,
    "thick_metalliclayer": thick_metal,
    "thick_sub": thick_sub,
    "thick_accroche": thick_accroche,
    "period": period
}

# Définir une plage de longueurs d'onde (par exemple de 400 nm à 1000 nm)
lambda_range = np.linspace(450, 1000, 100)  # 100 points
Ru_material_test = np.empty(lambda_range.size)
Rd_material_test = np.empty(lambda_range.size)

# Boucle sur la plage de longueurs d'onde
for idx_wav, wavelength in enumerate(lambda_range):
    # Mise à jour des permittivités pour chaque longueur d'onde
    perm_Ag = epsAgbb(wavelength)
    perm_Au = epsAubb(wavelength)
    
    # Calcul de la permittivité du substrat via SiA : ε = (n + i*k)**2
    n_val, k_val = get_n_k(material_test, wavelength)
    perm_material_test = (n_val + 1j * k_val)**2

    materials = {
        "perm_env": perm_env,
        "perm_dielec": perm_dielec,
        "perm_sub": perm_material_test,
        "perm_reso": perm_Ag,
        "perm_metalliclayer": perm_Au,
        "perm_accroche": perm_Au
    }
    wave = {"wavelength": wavelength, "angle": angle, "polarization": polarization}
    Ru_material_test[idx_wav], Rd_material_test[idx_wav] = reflectance(geometry, wave, materials, n_mod)

# Tracé du graphe de réflectance Rup en fonction de la longueur d'onde
plt.figure(figsize=(8, 5))
plt.plot(lambda_range, Ru_material_test, label=f"Au thick : {int(thick_metal)} nm")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance (Rup)")
plt.title("Réflectance R_up en fonction de la longueur d'onde")
plt.legend()
plt.grid(True)
plt.savefig("material_test_Rup.jpg")
plt.show(block=False)

# Sauvegarde des résultats
R = [Ru_material_test, Rd_material_test]
np.savez("data_accroches_all_Rdown-Rup.npz", list_wavelength=lambda_range, R=R)
