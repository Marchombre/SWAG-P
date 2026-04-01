# Function_reflectance_SWAG.py

import numpy as np
from gap_plasmon_2d.models.functions_rcwa import cascade, c_bas, interface, homogene, grating


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

    La méthode consiste à traiter verticalement (suivant z) la structure en cascade de matrices,
    tout en modélisant horizontalement, via grating, la distribution de permittivité dans certaines
    tranches de la cellule.

    Les épaisseurs peuvent être nulles. Dans ce cas, les couches correspondantes ne sont tout simplement
    pas traitées. Les matériaux associés ne sont donc requis que si la couche est réellement utilisée
    dans la cascade RCWA.
    """

    # -------------------------------------------------------------------------
    # 0. Helpers internes
    # -------------------------------------------------------------------------
    def _get_thickness(key):
        """
        Retourne l'épaisseur brute en nm depuis geometry.
        Si la clé est absente, on considère l'épaisseur comme nulle.
        """
        return float(geometry.get(key, 0.0))

    def _get_perm(key):
        """
        Retourne la permittivité associée à une clé matériau.
        On ne l'appelle QUE lorsque la couche ou l'interface correspondante
        est réellement nécessaire.
        """
        if key not in materials:
            raise KeyError(
                f"Matériau manquant : '{key}'. "
                f"Cette clé est nécessaire car une partie de la structure l'utilise réellement."
            )
        return materials[key]

    # -------------------------------------------------------------------------
    # 1. Paramètres géométriques (normalisés par "period")
    # -------------------------------------------------------------------------
    period = geometry["period"]

    width_reso = _get_thickness("width_reso") / period
    thick_reso = _get_thickness("thick_reso") / period
    thick_gap = _get_thickness("thick_gap") / period

    # Couches latérales
    thick_diel = _get_thickness("thick_diel") / period
    thick_func = _get_thickness("thick_func") / period
    thick_mol = _get_thickness("thick_mol") / period

    # Couches sous le gap
    thick_metalliclayer = _get_thickness("thick_metalliclayer") / period
    thick_XIAOYI = _get_thickness("thick_XIAOYI") / period
    thick_accroche = _get_thickness("thick_accroche") / period
    thick_sub = _get_thickness("thick_sub") / period

    # -------------------------------------------------------------------------
    # 2. Paramètres optiques
    # -------------------------------------------------------------------------
    wavelength = wave["wavelength"] / period
    angle = wave["angle"]
    polarization = wave["polarization"]

    # L'environnement est toujours nécessaire car il sert de milieu incident
    perm_env = _get_perm("perm_env")

    # Position pour le grating (définit la fraction du motif horizontal)
    pos_reso = np.array([[width_reso, (1 - width_reso) / 2]])
    n = 2 * n_mod + 1

    # -------------------------------------------------------------------------
    # 3. Constantes RCWA
    # -------------------------------------------------------------------------
    k0 = 2 * np.pi / wavelength
    a0 = k0 * np.sin(angle * np.pi / 180)

    # -------------------------------------------------------------------------
    # 4. Initialisation de la matrice S depuis l'environnement
    # -------------------------------------------------------------------------
    Pup, Vup = homogene(k0, a0, polarization, perm_env, n)
    S = np.block([
        [np.zeros((n, n), dtype=np.complex128), np.eye(n, dtype=np.complex128)],
        [np.eye(n, dtype=np.complex128), np.zeros((n, n), dtype=np.complex128)]
    ])

    # On part toujours de l’environnement comme milieu courant initial
    P_current = Pup

    # -------------------------------------------------------------------------
    # 5. Gestion du nanocube / gap / couches latérales
    # -------------------------------------------------------------------------
    # Cette partie ne doit être traitée que si une structure "supérieure"
    # existe réellement.
    if thick_reso > 0 or thick_gap > 0:

        sum_lat = thick_diel + thick_func + thick_mol

        # Le résonateur n'est requis ici que si on entre réellement
        # dans la logique du haut de structure.
        perm_reso = _get_perm("perm_reso")

        if sum_lat < thick_gap:
            # -----------------------------------------------------------------
            # (a) Les couches latérales sont entièrement contenues dans le gap
            # -----------------------------------------------------------------
            P_reso, V_reso = grating(
                k0, a0, polarization,
                perm_env, perm_reso,
                n, pos_reso
            )
            P_current = P_reso

            S = cascade(S, interface(Pup, P_reso))
            S = c_bas(S, V_reso, thick_reso)

            leftover_gap = thick_gap - sum_lat

            # Portion libre de gap
            if leftover_gap > 0:
                perm_gap = _get_perm("perm_gap")
                P_gap, V_gap = grating(
                    k0, a0, polarization,
                    perm_env, perm_gap,
                    n, pos_reso
                )
                S = cascade(S, interface(P_current, P_gap))
                S = c_bas(S, V_gap, leftover_gap)
                P_current = P_gap

            # Couche moléculaire
            if thick_mol > 0:
                perm_mol = _get_perm("perm_mol")
                perm_gap = _get_perm("perm_gap")
                P_mol, V_mol = grating(
                    k0, a0, polarization,
                    perm_mol, perm_gap,
                    n, pos_reso
                )
                S = cascade(S, interface(P_current, P_mol))
                S = c_bas(S, V_mol, thick_mol)
                P_current = P_mol

            # Couche de fonctionnalisation
            if thick_func > 0:
                perm_func = _get_perm("perm_func")
                perm_gap = _get_perm("perm_gap")
                P_func, V_func = grating(
                    k0, a0, polarization,
                    perm_func, perm_gap,
                    n, pos_reso
                )
                S = cascade(S, interface(P_current, P_func))
                S = c_bas(S, V_func, thick_func)
                P_current = P_func

            # Couche diélectrique
            if thick_diel > 0:
                perm_diel = _get_perm("perm_diel")
                perm_gap = _get_perm("perm_gap")
                P_diel, V_diel = grating(
                    k0, a0, polarization,
                    perm_diel, perm_gap,
                    n, pos_reso
                )
                S = cascade(S, interface(P_current, P_diel))
                S = c_bas(S, V_diel, thick_diel)
                P_current = P_diel

        else:
            # -----------------------------------------------------------------
            # (b) Les couches latérales égalent ou dépassent l'épaisseur du gap
            # -----------------------------------------------------------------
            P_reso, V_reso = grating(
                k0, a0, polarization,
                perm_env, perm_reso,
                n, pos_reso
            )
            P_current = P_reso

            S = cascade(S, interface(Pup, P_reso))
            S = c_bas(S, V_reso, thick_reso - (sum_lat - thick_gap))

            leftover_gap = thick_gap

            # -------------------------------------------------------------
            # Couche moléculaire
            # -------------------------------------------------------------
            if thick_mol > 0:
                perm_mol = _get_perm("perm_mol")
                perm_gap = _get_perm("perm_gap")

                if thick_mol < leftover_gap:
                    P_mol, V_mol = grating(
                        k0, a0, polarization,
                        perm_mol, perm_gap,
                        n, pos_reso
                    )
                    S = cascade(S, interface(P_current, P_mol))
                    S = c_bas(S, V_mol, thick_mol)
                    P_current = P_mol
                    leftover_gap -= thick_mol
                    thick_mol = 0

                else:
                    P_mol, V_mol = grating(
                        k0, a0, polarization,
                        perm_mol, perm_gap,
                        n, pos_reso
                    )
                    S = cascade(S, interface(P_current, P_mol))
                    S = c_bas(S, V_mol, leftover_gap)
                    P_current = P_mol

                    exc_mol = thick_mol - leftover_gap

                    P_mol_new, V_mol_new = grating(
                        k0, a0, polarization,
                        perm_mol, perm_reso,
                        n, pos_reso
                    )
                    S = cascade(S, interface(P_current, P_mol_new))
                    S = c_bas(S, V_mol_new, exc_mol)

                    P_current = P_mol_new
                    thick_mol = 0
                    leftover_gap = 0

            # -------------------------------------------------------------
            # Couche de fonctionnalisation
            # -------------------------------------------------------------
            if thick_func > 0:
                perm_func = _get_perm("perm_func")
                perm_gap = _get_perm("perm_gap")

                if thick_func < leftover_gap:
                    P_func, V_func = grating(
                        k0, a0, polarization,
                        perm_func, perm_gap,
                        n, pos_reso
                    )
                    S = cascade(S, interface(P_current, P_func))
                    S = c_bas(S, V_func, thick_func)
                    P_current = P_func
                    leftover_gap -= thick_func
                    thick_func = 0

                else:
                    P_func, V_func = grating(
                        k0, a0, polarization,
                        perm_func, perm_gap,
                        n, pos_reso
                    )
                    S = cascade(S, interface(P_current, P_func))
                    S = c_bas(S, V_func, leftover_gap)
                    P_current = P_func

                    exc_func = thick_func - leftover_gap

                    P_func_new, V_func_new = grating(
                        k0, a0, polarization,
                        perm_func, perm_reso,
                        n, pos_reso
                    )
                    S = cascade(S, interface(P_current, P_func_new))
                    S = c_bas(S, V_func_new, exc_func)

                    P_current = P_func_new
                    thick_func = 0
                    leftover_gap = 0

            # -------------------------------------------------------------
            # Couche diélectrique
            # -------------------------------------------------------------
            if thick_diel > 0:
                perm_diel = _get_perm("perm_diel")
                perm_gap = _get_perm("perm_gap")

                if thick_diel < leftover_gap:
                    P_diel, V_diel = grating(
                        k0, a0, polarization,
                        perm_diel, perm_gap,
                        n, pos_reso
                    )
                    S = cascade(S, interface(P_current, P_diel))
                    S = c_bas(S, V_diel, thick_diel)
                    P_current = P_diel
                    leftover_gap -= thick_diel
                    thick_diel = 0

                else:
                    P_diel, V_diel = grating(
                        k0, a0, polarization,
                        perm_diel, perm_gap,
                        n, pos_reso
                    )
                    S = cascade(S, interface(P_current, P_diel))
                    S = c_bas(S, V_diel, leftover_gap)
                    P_current = P_diel

                    exc_diel = thick_diel - leftover_gap

                    P_diel_new, V_diel_new = grating(
                        k0, a0, polarization,
                        perm_diel, perm_reso,
                        n, pos_reso
                    )
                    S = cascade(S, interface(P_current, P_diel_new))
                    S = c_bas(S, V_diel_new, exc_diel)

                    P_current = P_diel_new
                    thick_diel = 0
                    leftover_gap = 0

    # -------------------------------------------------------------------------
    # 6. Toutes les couches restantes
    #    (métal → toutes les homo_* → XIAOYI → accroche → substrat)
    #    dans cet ordre fixe, uniquement si présentes et d'épaisseur > 0
    # -------------------------------------------------------------------------
    ordered_keys = (
        ["thick_metalliclayer"]
        + [k for k in geometry if k.startswith("thick_homo_")]
        + ["thick_XIAOYI", "thick_accroche", "thick_sub"]
    )

    present_keys = [k for k in ordered_keys if geometry.get(k, 0) > 0]

    for key in present_keys:
        thickness = geometry[key] / period

        mat_key = "perm_" + key[len("thick_"):]
        perm_layer = _get_perm(mat_key)

        P_layer, V_layer = homogene(k0, a0, polarization, perm_layer, n)
        S = cascade(S, interface(P_current, P_layer))
        S = c_bas(S, V_layer, thickness)
        P_current = P_layer

    # -------------------------------------------------------------------------
    # 7. Calcul de la réflectance
    # -------------------------------------------------------------------------
    Rup = abs(S[n_mod, n_mod]) ** 2
    Rdown = abs(S[n + n_mod, n + n_mod]) ** 2

    return Rup, Rdown