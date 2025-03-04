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
