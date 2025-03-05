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
