### To get n and k

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
