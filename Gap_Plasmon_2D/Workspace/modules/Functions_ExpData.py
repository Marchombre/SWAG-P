import json
import numpy as np

def get_n_k(material_name, lam, json_path):
    with open(json_path) as file:
        data = json.load(file)
    if material_name not in data:
        raise ValueError(f"Le matériau '{material_name}' est pas dans la base de données.")
    material = data[material_name]
    if material["model"] == "ExpData":
        wl = np.array(material["wavelength_list"])
        epsilon_real = np.array(material["permittivities"])
        epsilon_imag = np.array(material.get("permittivities_imag", np.zeros_like(epsilon_real)))
        if lam < wl[0] or lam > wl[-1]:
            raise ValueError(f"La longueur d'onde {lam} nm est hors de l'intervalle [{wl[0]}, {wl[-1]}] nm.")
        eps_r = np.interp(lam, wl, epsilon_real)
        eps_i = np.interp(lam, wl, epsilon_imag)
        eps_complex = eps_r + 1.0j * eps_i
        n_complex = np.sqrt(eps_complex)
        return np.real(n_complex), np.imag(n_complex)
    else:
        raise ValueError(f"Le modèle '{material['model']}' pour '{material_name}' est pas supporté.")