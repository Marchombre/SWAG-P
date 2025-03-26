# Function_reflectance_SWAG.py

import numpy as np
from Functions_RCWA import cascade, c_bas, interface, homogene, grating

def reflectance(geometry, wave, materials, n_mod):
    """
    Calcule la réflectance (Rup, Rdown) en tenant compte de :
      - Un nanocube (résonateur) en haut,
      - Un gap (polymère) strictement sous le nanocube,
      - Trois couches latérales superposées (diélectrique "diel", fonctionnalisation "func" et molécules "mol")
        qui modulent horizontalement le gap via la fonction grating,
      - Une couche métallique,
      - Une couche d'accroche,
      - Et un substrat.
      
    La méthode consiste à traiter verticalement la structure en cascade de matrices,
    tout en modélisant horizontalement, via grating, la distribution de permittivité dans le gap.
    Les épaisseurs latérales peuvent être nulles, et leur somme peut être inférieure ou supérieure
    à l'épaisseur du gap. Dans le cas d'un dépassement, seule la portion jusqu'à hauteur du gap
    intervient dans la transition, la partie excédentaire étant prise en compte par une propagation
    supplémentaire dans le même milieu (ce qui modifie la modulation horizontale).
    """

    # ---------------------------------
    # 1. Paramètres géométriques (normalisés par "period")
    # ---------------------------------
    period = geometry["period"]
    width_reso = geometry["width_reso"] / period
    thick_reso = geometry["thick_reso"] / period       # Nanocube (résonateur)
    thick_gap  = geometry["thick_gap"]  / period         # Gap (polymère)

    # Couches latérales
    thick_diel = geometry["thick_diel"] / period         # Couche diélectrique latérale
    thick_func = geometry["thick_func"] / period         # Couche de fonctionnalisation
    thick_mol  = geometry["thick_mol"]  / period          # Couche moléculaire

    # Couches sous le gap
    thick_metalliclayer = geometry["thick_metalliclayer"] / period
    thick_accroche      = geometry["thick_accroche"]      / period
    thick_sub           = geometry["thick_sub"]           / period

    # ---------------------------------
    # 2. Paramètres optiques
    # ---------------------------------
    wavelength   = wave["wavelength"] / period
    angle        = wave["angle"]
    polarization = wave["polarization"]

    # Permittivités (matériaux)
    perm_env   = materials["perm_env"]             # Environnement (au-dessus du nanocube)
    perm_reso  = materials["perm_reso"]            # Nanocube (résonateur)
    perm_gap   = materials["perm_gap"]          # Gap (polymère)
    perm_diel  = materials["perm_diel"]      # Couche latérale : diélectrique
    perm_func  = materials["perm_func"]              # Couche latérale : fonctionnalisation
    perm_mol   = materials["perm_mol"]               # Couche latérale : moléculaire
    perm_metal = materials["perm_metalliclayer"]     # Couche métallique
    perm_acc   = materials["perm_accroche"]          # Couche d'accroche
    perm_sub   = materials["perm_sub"]               # Substrat

    # Position pour le grating (définit la fraction du motif horizontal)
    pos_reso = np.array([[width_reso, (1 - width_reso) / 2]])
    n = 2 * n_mod + 1  # Nombre de modes

    # ---------------------------------
    # 3. Constantes RCWA
    # ---------------------------------
    k0 = 2 * np.pi / wavelength
    a0 = k0 * np.sin(angle * np.pi / 180)

    # ---------------------------------
    # 4. Initialisation de la matrice S (provenant de l'environnement)
    # ---------------------------------
    Pup, Vup = homogene(k0, a0, polarization, perm_env, n)
    S = np.block([
        [np.zeros((n, n), dtype=np.complex128), np.eye(n, dtype=np.complex128)],
        [np.eye(n, dtype=np.complex128),        np.zeros((n, n), dtype=np.complex128)]
    ])

    # ---------------------------------
    # 5. Couche du nanocube (résonateur)
    # Transition via grating entre l'environnement et le matériau du nanocube
    # ---------------------------------
    P_reso, V_reso = grating(k0, a0, polarization, perm_env, perm_reso, n, pos_reso)
    S = cascade(S, interface(Pup, P_reso))
    S = c_bas(S, V_reso, thick_reso)
    # Après propagation du nanocube, la couche immédiatement inférieure est le gap

    # ---------------------------------
    # 6. Gestion du gap et des couches latérales
    # Le gap reste la couche en contact verticalement sous le nanocube.
    # Les couches latérales (diel, func, mol) modulent horizontalement le gap via grating.
    # On considère deux cas :
    #   (a) Si la somme des épaisseurs latérales est <= gap, elles s'insèrent entièrement dans le gap.
    #   (b) Si la somme dépasse l'épaisseur du gap, on ne dépose dans le gap que la portion disponible,
    #       et la portion excédentaire est prise en compte par une propagation supplémentaire dans le même milieu.
    # ---------------------------------
    sum_lat = thick_diel + thick_func + thick_mol
    # P_current désigne le milieu en cours (initialement, c'est la sortie du nanocube)
    P_current = P_reso

    if sum_lat <= thick_gap:
        # (a) Les couches latérales sont entièrement contenues dans le gap.
        leftover_gap = thick_gap - sum_lat
        # D'abord, si le gap n'est pas complètement consommé, propager la portion de gap "libre"
        if leftover_gap > 0:
            P_gap, V_gap = grating(k0, a0, polarization, perm_env, perm_gap, n, pos_reso)
            S = cascade(S, interface(P_current, P_gap))
            S = c_bas(S, V_gap, leftover_gap)
            P_current = P_gap
        # Puis, on ajoute successivement les couches latérales via grating
        if thick_diel > 0:
            P_diel, V_diel = grating(k0, a0, polarization, perm_gap, perm_diel, n, pos_reso)
            S = cascade(S, interface(P_current, P_diel))
            S = c_bas(S, V_diel, thick_diel)
            P_current = P_diel
        if thick_func > 0:
            P_func, V_func = grating(k0, a0, polarization, perm_gap, perm_func, n, pos_reso)
            S = cascade(S, interface(P_current, P_func))
            S = c_bas(S, V_func, thick_func)
            P_current = P_func
        if thick_mol > 0:
            P_mol, V_mol = grating(k0, a0, polarization, perm_gap, perm_mol, n, pos_reso)
            S = cascade(S, interface(P_current, P_mol))
            S = c_bas(S, V_mol, thick_mol)
            P_current = P_mol

    else:
        # (b) La somme des épaisseurs latérales dépasse l'épaisseur du gap.
        # On consomme d'abord le gap disponible (leftover_gap) pour chaque couche,
        # puis, si une couche n'est pas entièrement déposée, on propage la portion excédentaire
        # dans le même milieu (ce qui modifie la modulation horizontale dans le gap).
        leftover_gap = thick_gap
        # Traitement de la couche diélectrique
        if thick_diel > 0:
            if thick_diel <= leftover_gap:
                # Toute la couche diel s'insère dans le gap
                P_diel, V_diel = grating(k0, a0, polarization, perm_gap, perm_diel, n, pos_reso)
                S = cascade(S, interface(P_current, P_diel))
                S = c_bas(S, V_diel, thick_diel)
                P_current = P_diel
                leftover_gap -= thick_diel
                thick_diel = 0
            else:
                # Seule une partie de la couche diel s'insère dans le gap
                P_diel, V_diel = grating(k0, a0, polarization, perm_gap, perm_diel, n, pos_reso)
                S = cascade(S, interface(P_current, P_diel))
                S = c_bas(S, V_diel, leftover_gap)
                P_current = P_diel
                thick_diel -= leftover_gap
                leftover_gap = 0
                # Propagation de la portion excédentaire (en restant dans le même milieu)
                S = c_bas(S, V_diel, thick_diel)
                thick_diel = 0
        # Traitement de la couche de fonctionnalisation
        if thick_func > 0:
            if thick_func <= leftover_gap:
                P_func, V_func = grating(k0, a0, polarization, perm_gap, perm_func, n, pos_reso)
                S = cascade(S, interface(P_current, P_func))
                S = c_bas(S, V_func, thick_func)
                P_current = P_func
                leftover_gap -= thick_func
                thick_func = 0
            else:
                P_func, V_func = grating(k0, a0, polarization, perm_gap, perm_func, n, pos_reso)
                S = cascade(S, interface(P_current, P_func))
                S = c_bas(S, V_func, leftover_gap)
                P_current = P_func
                thick_func -= leftover_gap
                leftover_gap = 0
                S = c_bas(S, V_func, thick_func)
                thick_func = 0
        # Traitement de la couche moléculaire
        if thick_mol > 0:
            if thick_mol <= leftover_gap:
                P_mol, V_mol = grating(k0, a0, polarization, perm_gap, perm_mol, n, pos_reso)
                S = cascade(S, interface(P_current, P_mol))
                S = c_bas(S, V_mol, thick_mol)
                P_current = P_mol
                leftover_gap -= thick_mol
                thick_mol = 0
            else:
                P_mol, V_mol = grating(k0, a0, polarization, perm_gap, perm_mol, n, pos_reso)
                S = cascade(S, interface(P_current, P_mol))
                S = c_bas(S, V_mol, leftover_gap)
                P_current = P_mol
                thick_mol -= leftover_gap
                leftover_gap = 0
                S = c_bas(S, V_mol, thick_mol)
                thick_mol = 0
        # S'il reste un gap non consommé, le propager
        if leftover_gap > 0:
            P_gap, V_gap = grating(k0, a0, polarization, perm_gap, perm_gap, n, pos_reso)
            S = cascade(S, interface(P_current, P_gap))
            S = c_bas(S, V_gap, leftover_gap)
            P_current = P_gap
            leftover_gap = 0

    # ---------------------------------
    # 7. Couche métallique
    # ---------------------------------
    if thick_metalliclayer > 0:
        P_metal, V_metal = homogene(k0, a0, polarization, perm_metal, n)
        S = cascade(S, interface(P_current, P_metal))
        S = c_bas(S, V_metal, thick_metalliclayer)
        P_current = P_metal
    else:
        # Même si l'épaisseur est nulle, on effectue l'interface
        P_metal = homogene(k0, a0, polarization, perm_metal, n)[0]
        S = cascade(S, interface(P_current, P_metal))
        P_current = P_metal

    # ---------------------------------
    # 8. Couche d'accroche
    # ---------------------------------
    if thick_accroche > 0:
        P_acc, V_acc = homogene(k0, a0, polarization, perm_acc, n)
        S = cascade(S, interface(P_current, P_acc))
        S = c_bas(S, V_acc, thick_accroche)
        P_current = P_acc
    else:
        P_acc = homogene(k0, a0, polarization, perm_acc, n)[0]
        S = cascade(S, interface(P_current, P_acc))
        P_current = P_acc

    # ---------------------------------
    # 9. Substrat
    # ---------------------------------
    if thick_sub > 0:
        P_sub, V_sub = homogene(k0, a0, polarization, perm_sub, n)
        S = cascade(S, interface(P_current, P_sub))
        S = c_bas(S, V_sub, thick_sub)
        P_current = P_sub
    else:
        P_sub = homogene(k0, a0, polarization, perm_sub, n)[0]
        S = cascade(S, interface(P_current, P_sub))
        P_current = P_sub

    # ---------------------------------
    # 10. Calcul de la réflectance
    # ---------------------------------
    Rup   = abs(S[n_mod,       n_mod])**2
    Rdown = abs(S[n + n_mod, n + n_mod])**2

    return Rup, Rdown
