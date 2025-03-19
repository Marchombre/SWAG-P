# Functions_Models_Data.py
import os
import json
import numpy as np
from scipy.special import erf, wofz



def faddeeva(z,N):
    "Bidouille les signes et les parties réelles et imaginaires d'un nombre complexe --> à creuser"
    w=np.zeros(z.size,dtype=complex)

    idx=np.real(z)==0
    w[idx]=np.exp(np.abs(-z[idx]**2))*(1-erf(np.imag(z[idx])))
    idx=np.invert(idx)
    idx1=idx + np.imag(z)<0

    z[idx1]=np.conj(z[idx1])

    M=2*N
    M2=2*M
    k = np.arange(-M+1,M)
    L=np.sqrt(N/np.sqrt(2))

    theta=k*np.pi/M
    t=L*np.tan(theta/2)
    f=np.exp(-t**2)*(L**2+t**2)
    f=np.append(0,f)
    a=np.real(np.fft.fft(np.fft.fftshift(f)))/M2
    a=np.flipud(a[1:N+1])

    Z=(L+1.0j*z[idx])/(L-1.0j*z[idx])
    p=np.polyval(a,Z)
    w[idx]=2*p/(L-1.0j*z[idx])**2+(1/np.sqrt(np.pi))/(L-1.0j*z[idx])
    w[idx1]=np.conj(2*np.exp(-z[idx1]**2) - w[idx1])
    return(w)




def BrendelBormann_Faddeeva(lambda_test, f0, omega_p, Gamma0, f, omega, gamma, sigma, N):
    """
    Modèle Brendel & Bormann utilisant la fonction de Voigt pour modéliser des résonances lorentziennes
    élargies par une distribution gaussienne, avec l'approximation FFT pour la fonction de Faddeeva.

    Paramètres:
      - lambda_test : longueur d'onde en nm
      - f0, omega_p, Gamma0 : paramètres pour la contribution Drude (en eV)
      - f, omega, gamma, sigma : listes ou numpy arrays pour les résonances (en eV)
      - N : paramètre numérique pour le calcul FFT dans la fonction faddeeva

    Retourne:
      - epsilon : permittivité complexe calculée
    """

    # Convertir les paramètres en tableaux NumPy pour éviter les problèmes de type
    f = np.array(f, dtype=float)
    omega = np.array(omega, dtype=float)
    gamma = np.array(gamma, dtype=float)
    sigma = np.array(sigma, dtype=float)

    # Calcul de w, qui correspond à l'énergie en eV (w = 6.62606957e-25 * c / (e * lambda_test))
    w_val = 6.62606957e-25 * 299792458 / (1.602176565e-19 * lambda_test)
    
    # Calcul de a comme la racine de w*(w + 1j*gamma)
    a = np.sqrt(w_val * (w_val + 1j * gamma))
    # On force le signe de la partie réelle de a pour assurer une cohérence dans le choix de la branche
    a = a * np.sign(np.real(a))
    
    # Calcul de x et y pour les fonctions de Voigt, en utilisant np.sqrt(2)*sigma dans le dénominateur
    x = (a - omega) / (np.sqrt(2) * sigma)
    y = (a + omega) / (np.sqrt(2) * sigma)
    
    # Calcul de la contribution chi_b via la somme vectorisée
    chi_b = np.sum(
        1j * np.sqrt(np.pi) * f * omega_p**2 / (2 * np.sqrt(2) * a * sigma) *
        (faddeeva(x, N) + faddeeva(y, N))
    )
    
    # Calcul de la contribution Drude chi_f
    chi_f = - (omega_p**2) * f0 / (w_val * (w_val + 1j * Gamma0))
    
    # La permittivité complexe finale
    epsilon = 1 + chi_f + chi_b
    return epsilon



def compute_permittivity(lam, f0, omega_p, Gamma0, f, omega, gamma, sigma, N=64):
    """
    Wrapper pour calculer la permittivité à l'aide du modèle Brendel & Bormann.
    """
    return BrendelBormann_Faddeeva(lam, f0, omega_p, Gamma0, f, omega, gamma, sigma, N)

def get_n_k(material_name, lam, json_combined_path):
    """
    Extrait n et k à partir d'un fichier JSON contenant des données expérimentales.
    lam est en nm.
    """
    with open(json_combined_path) as file:
        data = json.load(file)

    if material_name not in data:
        raise ValueError(f"Material '{material_name}' is not in the database.")
    material = data[material_name]

    if material["model"] == "expdata":
        wl = np.array(material["wavelength_list"])
        epsilon_real = np.array(material["permittivities"])
        epsilon_imag = np.array(material.get("permittivities_imag", np.zeros_like(epsilon_real)))

        if lam < wl[0] or lam > wl[-1]:
            raise ValueError(f"Wavelength {lam} nm is out of the range [{wl[0]}, {wl[-1]}] nm.")
        
        eps_r = np.interp(lam, wl, epsilon_real)
        eps_i = np.interp(lam, wl, epsilon_imag)
        n_complex = np.sqrt(eps_r + 1.0j * eps_i)

        return np.real(n_complex), np.imag(n_complex)
    else:
        raise ValueError(f"Model '{material['model']}' for '{material_name}' is not supported.")

def get_permittivity_from_txt(material_name, lambda_val_nm, data_dir):
    """
    Lit un fichier texte dans data_dir pour un matériau donné.
    Le fichier doit contenir trois colonnes : longueur d'onde (en µm), n, k.
    Les longueurs d'onde du fichier (en µm) sont converties en nanomètres pour que 
    l'interpolation se fasse avec lambda_val_nm (qui est en nm).
    Retourne ε = (n + i*k)^2 interpolé à lambda_val_nm.
    """
    import glob
    import numpy as np
    import os

    # Cherche le fichier [material_name].txt ou contenant material_name dans data_dir
    pattern = os.path.join(data_dir, f"{material_name}.txt")
    txt_files = glob.glob(pattern)
    if not txt_files:
        pattern = os.path.join(data_dir, f"*{material_name}*.txt")
        txt_files = glob.glob(pattern)
        if not txt_files:
            raise ValueError(f"Le fichier texte pour '{material_name}' n'a pas été trouvé dans {data_dir}.")
    txt_file = txt_files[0]

    # Lire toutes les lignes du fichier
    with open(txt_file, "r") as f:
        lines = f.readlines()
    nb_lines = len(lines)
    wl_data, n_data, k_data = [], [], []

    # On suppose que les données se trouvent entre la 3ème ligne et l'avant-dernière ligne
    for idx in range(2, nb_lines - 2):
        line = lines[idx].strip()
        if not line:
            continue
        try:
            # Convertit la ligne en une liste de float
            vals = [float(v) for v in line.split()]
            if len(vals) < 3:
                continue
            wl_data.append(vals[0])
            n_data.append(vals[1])
            k_data.append(vals[2])
        except ValueError:
            continue

    # Conversion des longueurs d'onde du fichier :
    # Le fichier fournit les wl en micromètres. Pour que l'interpolation se fasse 
    # avec lambda_val_nm (en nanomètres), on convertit wl_data en nm en multipliant par 1000.
    wl_data = np.array(wl_data)# * 1000
    n_data = np.array(n_data)
    k_data = np.array(k_data)

    # Interpolation linéaire pour obtenir n et k à la longueur d'onde lambda_val_nm
    n_val = np.interp(lambda_val_nm, wl_data, n_data)
    k_val = np.interp(lambda_val_nm, wl_data, k_data)
    return (n_val + 1j * k_val)**2

def get_material_permittivity(material_name, lambda_val_nm, json_combined_path, data_dir):
    """
    Retourne la permittivité ε pour un matériau donné à la longueur d'onde lambda_val_nm.
    
    Recherche dans le fichier JSON combiné :
      - Si le modèle est "expdata", on utilise get_n_k.
      - Sinon, on utilise compute_permittivity.
    
    Si le matériau n'est pas trouvé dans le JSON, on lit le fichier texte correspondant dans data_dir.
    """
    with open(json_combined_path, "r", encoding="utf-8") as f:
        materials_data = json.load(f)
    material_lower = material_name.lower()
    found_in_json = False
    for key in materials_data:
        if key.lower() == material_lower:
            found_in_json = True
            actual_mat = key
            break
    if found_in_json:
        material = materials_data[actual_mat]
        model = material.get("model", "").lower()
        if model == "expdata":
            n_val, k_val = get_n_k(actual_mat, lambda_val_nm, json_combined_path)
            return (n_val + 1j * k_val)**2
        elif model == "brendelbormann":
            try:
                f0 = material["f0"]
                omega_p = material["omega_p"]
                Gamma0 = material["Gamma0"]
                f = material["f"]
                omega = material["omega"]
                gamma = material["Gamma"]
                sigma = material["sigma"]
                return compute_permittivity(lambda_val_nm, f0, omega_p, Gamma0, f, omega, gamma, sigma, N=64)
            except KeyError as e:
                raise ValueError(f"Les paramètres pour '{actual_mat}' sont incomplets dans le JSON : {e}")
            
        else:
            raise ValueError(f"Modèle '{material.get('model')}' non supporté pour le matériau '{actual_mat}'.")
        
    else:
        # Matériau non présent dans le JSON, lecture depuis un fichier texte dans data_dir.
        return get_permittivity_from_txt(material_name, lambda_val_nm, data_dir)


