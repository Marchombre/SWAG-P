# Function_reflectance_SWAG_V2bis.py

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
      
    La méthode consiste à traiter verticalement  (suivant z) la structure en cascade de matrices,
    tout en modélisant horizontalement, via grating, la distribution de permittivité dans certaines tranche de la cellule.
    Les épaisseurs peuvent être nulles, et la somme des trois zones latérales superposées, attenantes aux faces latérales du gap et du nanocube, 
    peut être inférieure ou supérieure à l'épaisseur du gap, et par extension, à celle du nanocube.
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
    perm_env   = materials["perm_env"]             # Environnement (au-dessus du nanocube et sur les côtés)
    perm_reso  = materials["perm_reso"]            # Nanocube (résonateur)
    perm_gap   = materials["perm_gap"]              # Gap (polymère)
    perm_diel  = materials["perm_diel"]             # Couche latérale : diélectrique
    perm_func  = materials["perm_func"]              # Couche latérale : fonctionnalisation
    perm_metal = materials["perm_metalliclayer"]     # Couche métallique
    perm_acc   = materials["perm_accroche"]          # Couche d'accroche
    perm_sub   = materials["perm_sub"]               # Substrat

    # Position pour le grating (définit la fraction du motif horizontal)
    pos_reso = np.array([[width_reso, (1 - width_reso) / 2]])
    n = 2 * n_mod + 1  # Nombre de modes, avec n_mod modes de chaque côté de l’ordre nul, on a au total 2 * n_mod + 1 modes.

    # ---------------------------------
    # 3. Constantes RCWA
    # ---------------------------------
    k0 = 2 * np.pi / wavelength
    a0 = k0 * np.sin(angle * np.pi / 180)

    # ---------------------------------
    # 4. Initialisation de la matrice S (provenant de l'environnement)
    # ---------------------------------
    # Cette matrice S représente l’état initial de la propagation 
    # (on l’utilise pour accumuler les effets de chaque couche via les cascades).
    Pup, Vup = homogene(k0, a0, polarization, perm_env, n)
    S = np.block([
        [np.zeros((n, n), dtype=np.complex128), np.eye(n, dtype=np.complex128)],
        [np.eye(n, dtype=np.complex128),        np.zeros((n, n), dtype=np.complex128)]
    ])


    # ---------------------------------
    # 5. Gestion du gap et des couches latérales
    # Le gap est strictement sous le nanocube et sa largeur est égale à celle du nanocube.
    # Les couches latérales (diel, func, mol) modulent horizontalement le gap et le nanocube via grating.
    # On considère deux cas :
    
    #   (a) Si la somme des épaisseurs latérales est < gap, elles s'insèrent entièrement dans le gap.
            # On commence donc par un grating de l'environnement vers le resonateur, puis on cascade vers le gap.
            # Puis la partie selon z de gap disponible non collé aux couches diel, func et mol, 
            # est alors traité en grating environnement vers gap.
            # Ensuite, on traite simplement en grating les couches latérales mol, func et diel vers le gap avec les cascades, 
            # interfaces et c_bas associés. Attention à bien tenir compte des cas où l'épaisseur 
            # d'une couche diel, func ou mol est nulle.
            
    #   (b) Si la somme est égale ou dépasse l'épaisseur du gap, on commence donc par un grating de l'environnement 
            # vers le resonateur en tenant compte de la longueur de l'interface selon l'axe z environnement/nanocube 
            # disponible. Ensuite on traite en grating toute les couches latérals mol et/ou func et/ou diel, ou tout du moins 
            # leur portions disponnibles, vers le nanocube avec les cascades, interfaces et c_bas associés.
            # Ensuite, on traite simplement en grating toute les couches latérals mol et/ou func et/ou diel, ou tout du moins 
            # leur portions disponnibles vers le gap avec les cascades, interfaces et c_bas associés.
            # Attention à bien tenir compte des cas où l'épaisseur d'une couche diel, func ou mol est nulle.
    #      
    # ---------------------------------
    sum_lat = thick_diel + thick_func


    if sum_lat < thick_gap:
        # (a) Les couches latérales diel + func + mol sont entièrement contenues dans le gap.
        
        P_reso, V_reso = grating(k0, a0, polarization, perm_env, perm_reso, n, pos_reso)
        # P_current désigne le milieu en cours (initialement, c'est la sortie du nanocube)
        P_current = P_reso
        
        S = cascade(S, interface(Pup, P_reso))
        S = c_bas(S, V_reso, thick_reso)
            
        leftover_gap = thick_gap - sum_lat
        # D'abord, si le gap n'est pas complètement "consommé", propager la portion de gap "libre"
        
        if leftover_gap > 0:
            P_gap, V_gap = grating(k0, a0, polarization, perm_env, perm_gap, n, pos_reso)
            S = cascade(S, interface(P_current, P_gap))
            S = c_bas(S, V_gap, leftover_gap)
            P_current = P_gap
        # Puis, on ajoute successivement les couches latérales via grating

        if thick_func > 0:
            P_func, V_func = grating(k0, a0, polarization, perm_func, perm_gap, n, pos_reso)
            S = cascade(S, interface(P_current, P_func))
            S = c_bas(S, V_func, thick_func)
            P_current = P_func
                          
        if thick_diel > 0:
            P_diel, V_diel = grating(k0, a0, polarization, perm_diel, perm_gap, n, pos_reso)
            S = cascade(S, interface(P_current, P_diel))
            S = c_bas(S, V_diel, thick_diel)
            P_current = P_diel

    else:
        # (b) La somme des épaisseurs latérales est égale ou dépasse l'épaisseur du gap.
        # On traite la partie diponible du reso en grating env vers reso avant de traiter les couches latérales.
        P_reso, V_reso = grating(k0, a0, polarization, perm_env, perm_reso, n, pos_reso)
        # P_current désigne le milieu en cours (initialement, c'est la sortie du nanocube)
        P_current = P_reso
        
        S = cascade(S, interface(Pup, P_reso))
        S = c_bas(S, V_reso, thick_reso - (sum_lat - thick_gap))
                
        leftover_gap = thick_gap
                    
                                    
        # Traitement de la couche de fonctionnalisation
        if thick_func > 0:
            if thick_func < leftover_gap:
                P_func, V_func = grating(k0, a0, polarization, perm_func, perm_gap, n, pos_reso)
                S = cascade(S, interface(P_current, P_func))
                S = c_bas(S, V_func, thick_func)
                P_current = P_func
                leftover_gap -= thick_func
                thick_func = 0
            else:
                # 1) Insérer dans le gap la portion disponible                
                P_func, V_func = grating(k0, a0, polarization, perm_func, perm_gap, n, pos_reso)
                S = cascade(S, interface(P_current, P_func))
                S = c_bas(S, V_func, leftover_gap)
                P_current = P_func  # <-- Pour mettre à jour pour refléter 
                # l’état après cette propagation avant de traiter la portion excédentaire
                                
                # 2) Pour la portion excédentaire, calculer un nouveau grating avec le nanocube
                exc_func = thick_func - leftover_gap
                # On suppose que la partie latérale du nanocube est décrite par perm_reso
                P_func_new, V_func_new = grating(k0, a0, polarization, perm_func, perm_reso, n, pos_reso)
                S = cascade(S, interface(P_current, P_func_new))
                S = c_bas(S, V_func_new, exc_func)
                # Mise à jour : la couche est entièrement traitée et le gap est épuisé
                P_current = P_func_new
                thick_func = 0
                leftover_gap = 0
                                
                
                    
        # Traitement de la couche diélectrique
        if thick_diel > 0:
            if thick_diel < leftover_gap:
                # La couche entière s'insère dans le gap
                P_diel, V_diel = grating(k0, a0, polarization, perm_diel, perm_gap, n, pos_reso)
                S = cascade(S, interface(P_current, P_diel))
                S = c_bas(S, V_diel, thick_diel)
                P_current = P_diel
                leftover_gap -= thick_diel
                thick_diel = 0
            else:
                # 1) Insérer dans le gap la portion disponible
                P_diel, V_diel = grating(k0, a0, polarization, perm_diel, perm_gap, n, pos_reso)
                S = cascade(S, interface(P_current, P_diel))
                S = c_bas(S, V_diel, leftover_gap)
                P_current = P_diel  # <-- Pour mettre à jour pour refléter 
                # l’état après cette propagation avant de traiter la portion excédentaire
                
                # 2) Pour la portion excédentaire, calculer un nouveau grating avec le nanocube
                exc_diel = thick_diel - leftover_gap
                # On suppose que la partie latérale du nanocube est décrite par perm_reso
                P_diel_new, V_diel_new = grating(k0, a0, polarization, perm_diel, perm_reso, n, pos_reso)
                S = cascade(S, interface(P_current, P_diel_new))
                S = c_bas(S, V_diel_new, exc_diel)
                # Mise à jour : la couche est entièrement traitée et le gap est épuisé
                P_current = P_diel_new
                thick_diel = 0
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
