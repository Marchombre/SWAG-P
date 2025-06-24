import numpy as np
from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy.interpolate import interp1d
from copy import deepcopy
from gap_plasmon_2d.simulation.simulate_and_plot import run_simulation_one_combo
from gap_plasmon_2d.utils.data_readers import get_baseline_n

def _find_dip_core(
    wavelength, reflectance,
    smooth_win, polyorder,
    dip_prom, dip_dist,
    peak_dist,
    verbose=False, cfg_name=None
):

    lam = np.asarray(wavelength)
    R   = np.asarray(reflectance)

    # 1) lissage optionnel
    if smooth_win > 1:
        R_smooth = savgol_filter(R, smooth_win, polyorder)
    else:
        R_smooth = R.copy()

    # 2) inversion et détection de tous les dips
    inv_R = -R_smooth
    dips, _ = find_peaks(inv_R, prominence=dip_prom, distance=dip_dist)

    # cas "pas de dip"
    if dips.size == 0:
        if verbose:
            print(f"[find_core] Aucun dip détecté pour « {cfg_name} »")
        empty = []
        return (
            empty, empty, empty,  # dip_idx_list, lam_dip_list, R_dip_list
            empty,                # y_level_list
            empty, empty, empty,  # lam_left_list, lam_right_list, fwhm_list
            empty, empty,         # lam_max_l_list, R_max_l_list
            empty, empty,         # lam_max_r_list, R_max_r_list
            empty, empty,         # lam_sym_list, R_sym_list
            empty          # depth_list
        ), cfg_name

    # listes à remplir
    dip_idx_list       = []
    lam_dip_list       = []
    R_dip_list         = []
    y_level_list       = []
    lam_left_list      = []
    lam_right_list     = []
    fwhm_list          = []
    lam_max_l_list     = []
    R_max_l_list       = []
    lam_max_r_list     = []
    R_max_r_list       = []
    lam_sym_list       = []
    R_sym_list         = []
    depth_list         = []

    # pré-calculs pour pente et FWHM manuel
    dR    = np.gradient(R_smooth, lam)
    grad  = np.abs(dR)

    # 3) boucle sur chaque dip
    for j, i0 in enumerate(dips):
        # 3a) affinement du dip sur le spectre brut
        lo, hi = max(0, i0-1), min(len(R), i0+2)
        sub    = lo + np.argmin(R[lo:hi])
        lam_dip = lam[sub]
        R_dip   = R[sub]

        # interpolation parabolique 3 points
        if 0 < sub < len(R)-1:
            y0, y1, y2 = R[sub-1], R[sub], R[sub+1]
            a = (y0 + y2)/2 - y1
            b = (y2 - y0)/2
            if a != 0:
                delta = -b/(2*a)
                lam_dip = lam[sub] + delta*(lam[1]-lam[0])
                R_dip   = y1 - b*delta/2

        # 3b) maxima adjacents + plateau
        peaks, _ = find_peaks(R_smooth, prominence=dip_prom, distance=peak_dist)

        # gauche
        left_peaks = peaks[peaks < sub]
        # détection de plateaux
        dR_left = dR[:sub]
        flat_left = []
        if dR_left.size:
            thr_l = 0.01 * np.nanmax(np.abs(dR_left))
            cond  = np.abs(dR_left) < thr_l
            i_l = 0
            while i_l < len(cond):
                if cond[i_l]:
                    j2 = i_l
                    while j2+1 < len(cond) and cond[j2+1]:
                        j2 += 1
                    if (j2 - i_l + 1) >= 5:
                        flat_left.append(i_l)
                    i_l = j2+1
                else:
                    i_l += 1
        candidates_l = np.concatenate([left_peaks, flat_left]) if (left_peaks.size or flat_left) else np.array([])
        lm = int(candidates_l.max()) if candidates_l.size else 0
        lam_max_l = lam[lm]; R_max_l = R[lm]

        # droite
        right_peaks = peaks[peaks > sub]
        dR_right = dR[sub+1:]
        flat_right = []
        if dR_right.size:
            thr_r = 0.01 * np.nanmax(np.abs(dR_right))
            cond  = np.abs(dR_right) < thr_r
            i_r = 0
            while i_r < len(cond):
                if cond[i_r]:
                    j2 = i_r
                    while j2+1 < len(cond) and cond[j2+1]:
                        j2 += 1
                    if (j2 - i_r + 1) >= 5:
                        flat_right.append(sub+1 + i_r)
                    i_r = j2+1
                else:
                    i_r += 1
        candidates_r = np.concatenate([right_peaks, flat_right]) if (right_peaks.size or flat_right) else np.array([])
        rm = int(candidates_r.min()) if candidates_r.size else len(R_smooth)-1
        lam_max_r = lam[rm]; R_max_r = R[rm]

        # 3c) symétrie du plus petit max
        if R_max_l < R_max_r:
            ref_val = R_max_l
            seg_lam = lam[sub:rm+1]; seg_R = R_smooth[sub:rm+1]
        else:
            ref_val = R_max_r
            seg_lam = lam[lm:sub+1][::-1]; seg_R = R_smooth[lm:sub+1][::-1]
        lam_sym = np.interp(ref_val, seg_R, seg_lam)
        R_sym   = ref_val

        # 3d) profondeur
        depth = ref_val - R_dip

        # 3e) FWHM manuel
        half = R_dip + 0.5*(ref_val - R_dip)
        if R_max_l < R_max_r:
            lam1 = np.interp(half, R_smooth[sub:rm+1], lam[sub:rm+1])
            lam2 = np.interp(half, R_smooth[lm:sub+1][::-1], lam[lm:sub+1][::-1])
        else:
            lam1 = np.interp(half, R_smooth[lm:sub+1][::-1], lam[lm:sub+1][::-1])
            lam2 = np.interp(half, R_smooth[sub:rm+1],      lam[sub:rm+1])
        lam_l, lam_r = sorted((lam1, lam2))
        fwhm = lam_r - lam_l

        # 3f) y_level
        y_level = R_dip + 0.5*(min(R_max_l, R_max_r) - R_dip)

        # 4) stocker dans toutes les listes
        dip_idx_list.append(int(sub))
        lam_dip_list.append(lam_dip)
        R_dip_list.append(R_dip)
        y_level_list.append(y_level)
        lam_left_list.append(lam_l)
        lam_right_list.append(lam_r)
        fwhm_list.append(fwhm)
        lam_max_l_list.append(lam_max_l)
        R_max_l_list.append(R_max_l)
        lam_max_r_list.append(lam_max_r)
        R_max_r_list.append(R_max_r)
        lam_sym_list.append(lam_sym)
        R_sym_list.append(R_sym)
        depth_list.append(depth)

    return (
        dip_idx_list,    lam_dip_list,    R_dip_list,
        y_level_list,
        lam_left_list,   lam_right_list,  fwhm_list,
        lam_max_l_list,  R_max_l_list,
        lam_max_r_list,  R_max_r_list,
        lam_sym_list,    R_sym_list,
        depth_list
    ), cfg_name




def minmax(x):
    a = np.array(x, dtype=float)
    mn, mx = a.min(), a.max()
    return (a - mn)/(mx - mn) if mx > mn else np.zeros_like(a)




def simulate_delta_spectrum(
    cfg,
    lam, wave, n_modes,
    sel_layers, delta_n, 
    lam_dip, R_dip,
    lam_left, lam_right,
    base_spectrum,
    json_combined_path,
    dip_index: int = 0,
    mode: str = "dip"   # "dip" ou "half"
):
    """
    À partir d’un spectre de base R(base_spectrum),
    construit R(n+Δn), identifie son dip principal via _core,
    et calcule S_lambda, S_R, dR_half.
    """
    
    # 1) baseline n0 pour chaque layer
    # on crée un dictionnaire nommé n0s
    n0s = {
        # pour chaque couche 'lay' dans la liste sel_layers,
        # on appelle la fonction get_baseline_n pour calculer la valeur de n₀ à λ = lam_dip
        # et on associe cette valeur à la clé 'lay' dans le dictionnaire
        lay: get_baseline_n(
            cfg,                # la configuration globale contenant les paramètres des matériaux
            lay,                # la clé de la couche pour laquelle on veut n₀
            lam_dip,            # la longueur d’onde du dip (en nm)
            json_combined_path  # chemin vers le fichier JSON décrivant les matériaux
        )
        for lay in sel_layers   # on répète cela pour chaque élément de sel_layers
    }
    # au final, n0s ressemble à :
    # {
    #   'couche1': valeur_n0_pour_couche1,
    #   'couche2': valeur_n0_pour_couche2,
    #   ...
    # }


    # 2) deep copy & custom ε=(n0+Δn)**2
    cfg2 = deepcopy(cfg)  # duplique la configuration pour ne pas modifier l’original
    for e in cfg2["material"]["MATERIALS_CONFIG"]:  # parcourt chaque entrée matériau
        if e["key"] in sel_layers:  # si la clé du matériau est dans les couches ciblées
            e["material"]["type"]       = "Custom"  # passe ce matériau en type personnalisé
            e["material"]["expression"] = f"({n0s[e['key']]+delta_n})**2"  
            # définit ε = (n0 + Δn)² pour ce matériau

    # 3) Simulation RCWA pour R(n+Δn)
    Rup_dn, _, _ = run_simulation_one_combo(lam, wave, n_modes, cfg2, json_combined_path)
    # exécute la simulation RCWA sur le nouveau cfg2 pour obtenir la réflectance R(n+Δn)
    Rup_dn = np.asarray(Rup_dn, dtype=float)  # convertit la liste retournée en array de floats

    # 4) dip sur R+Δn → on récupère maintenant 15 listes
    (dip_idx_list_dn, lam_dip_list_dn, R_dip_list_dn,
    y_level_list_dn,
    lam_left_list_dn, lam_right_list_dn, fwhm_list_dn,
    lam_max_l_list_dn, R_max_l_list_dn,
    lam_max_r_list_dn, R_max_r_list_dn,
    lam_sym_list_dn, R_sym_list_dn,
    depth_list_dn), _ = _find_dip_core(
        wavelength=lam,
        reflectance=Rup_dn,
        smooth_win=0,
        polyorder=0,
        dip_prom=1e-2,
        dip_dist=1,
        peak_dist=1,
        verbose=False,
        cfg_name=cfg.get("config_name")
    )

    # Aucun dip détecté
    if not dip_idx_list_dn:
        return Rup_dn, lam_calc, R_base_calc, lam_calc_dn, R_dn_calc, S_lambda, S_R, dR_half

    # on choisit le même index qu’en base (ou le plus proche si hors bornes)
    if dip_index is None or dip_index < 0 or dip_index >= len(dip_idx_list_dn):
        diffs = [abs(l - lam_dip) for l in lam_dip_list_dn]
        dip_index_dn = int(np.argmin(diffs))
    else:
        dip_index_dn = dip_index

    lam_dip_dn = lam_dip_list_dn[dip_index_dn]
    R_dip_dn   = R_dip_list_dn[dip_index_dn]


    # calcul demi-hauteur
    step = lam[1]-lam[0]
    base_interp = interp1d(lam, base_spectrum, kind='cubic')
    # juste avant de calculer slope_l / slope_r
    # on s’assure que lam_left +/- step sont bien dans la grille
    lam_min, lam_max = lam[0], lam[-1]
    x_l = np.clip(lam_left + step, lam_min, lam_max)
    x_r = np.clip(lam_right + step, lam_min, lam_max)
    x_lm = np.clip(lam_left - step, lam_min, lam_max)
    x_rm = np.clip(lam_right - step, lam_min, lam_max)

    slope_l = (base_interp(x_l) - base_interp(x_lm)) / (2*step)
    slope_r = (base_interp(x_r) - base_interp(x_rm)) / (2*step)

    #slope_l = (base_interp(lam_left+step) - base_interp(lam_left-step))/(2*step)
    #slope_r = (base_interp(lam_right+step)- base_interp(lam_right-step))/(2*step)
    lambda_half_pt = lam_left if abs(slope_l)>abs(slope_r) else lam_right
    R_half_base = float(base_interp(lambda_half_pt))
    
    # 1) clamp du point de demi-hauteur
    x_hp = np.clip(lambda_half_pt, lam_min, lam_max)

    # 2) création d’un interpolateur pour Rup_dn
    delta_interp = interp1d(
        lam, Rup_dn, kind='cubic',
        bounds_error=False,
        fill_value="extrapolate"
    )

    # 3) récupération de R_half_dn
    R_half_dn = float(delta_interp(x_hp))

    #R_half_dn   = float(interp1d(lam, Rup_dn, kind='cubic')(lambda_half_pt))

    # choix du point
    if mode=="half":
        lam_calc = lambda_half_pt
        R_base_calc = R_half_base
        # on interpole Rup_dn pour mesurer ΔR/Δn au même lam_half_pt
        dn_interp    = interp1d(lam, Rup_dn, kind='cubic', bounds_error=False, fill_value='extrapolate')
        R_dn_calc    = float(dn_interp(lambda_half_pt))
        # sur Rup_dn, on veut la FWHM (half_pt) sur la même branche que celle choisie sur le spectre de base
        if lambda_half_pt == lam_left:
            lam_calc_dn = lam_left_list_dn[dip_index_dn]
        else:
            lam_calc_dn = lam_right_list_dn[dip_index_dn]

        #lam_calc, R_base_calc, R_dn_calc, lam_calc_dn = lambda_half_pt, R_half_base, R_half_dn, R_half_dn
        #on cherche λ_dn tel que Rup_dn(λ_dn) = R_half_dn
        #on suppose qu'on peut inverser Rup_dn→lam sur l'intervalle autour de lambda_half_pt
        #on construit l'interpolateur inverse
        #inv_interp = interp1d(Rup_dn, lam, kind='linear', bounds_error=False, fill_value='extrapolate')
        #lam_calc_dn = float(inv_interp(R_half_dn)) # donne la longueur d’onde pour laquelle le spectre Rup_dn retourne exactement la demi-hauteur R_half_dn
                              
    else: # λ_ref = lam_dip
        lam_calc, R_base_calc, lam_calc_dn = lam_dip, R_dip, lam_dip_dn
        # recaler R_dn à la même λ_ref, pas à lam_dip_dn
        R_dn_calc   = float(interp1d(lam, Rup_dn, kind='cubic')(lam_dip))


    # sensibilités
    S_lambda = abs((lam_calc - lam_calc_dn) / delta_n)
    S_R      = abs((R_base_calc  - R_dn_calc) / delta_n)
    dR_half  = abs(R_half_dn - R_half_base)

    return Rup_dn, lam_calc, R_base_calc, lam_calc_dn, R_dn_calc, S_lambda, S_R, dR_half 




def find_best_dip(
    cfg,
    wavelength, reflectance,
    wave, n_modes,
    sel_layers, delta_n,
    json_combined_path,
    smooth_win, polyorder,
    dip_prom, dip_dist,
    peak_dist,
    verbose=False, cfg_name=None,  mode: str = "dip"
):
    """
    Wrapper public :
      1) détecte tous les dips via _find_dip_core (qui ne renvoie que des listes),
      2) pour chaque dip j, simule Δn et calcule S_R,
      3) choisit le dip j qui maximise S_R,
      4) renvoie un tuple de scalaires + la liste des indices + la liste des dR_over_dns.
    """

    # 1) détecte les candidats dips : on récupère 15 listes (toutes même longueur)
    (dip_idx_list,    lam_dip_list,   R_dip_list, y_level_list,
     lam_left_list,   lam_right_list, fwhm_list,
     lam_max_l_list,  R_max_l_list,
     lam_max_r_list,  R_max_r_list,
     lam_sym_list,    R_sym_list,
     depth_list), _ = _find_dip_core(
        wavelength=wavelength,
        reflectance=reflectance,
        smooth_win=smooth_win,
        polyorder=polyorder,
        dip_prom=dip_prom,
        dip_dist=dip_dist,
        peak_dist=peak_dist,
        verbose=verbose,
        cfg_name=cfg_name
    )


    lam_arr    = np.asarray(wavelength)      # vecteur des longueurs d’onde
    R_arr      = np.asarray(reflectance)     # vecteur des réflectances

    # 1) Lissage (identique à _find_dip_core)
    if smooth_win > 1:
        R_smooth   = savgol_filter(R_arr, smooth_win, polyorder)
    else:
        R_smooth   = R_arr.copy()


    interp_R = interp1d(lam_arr, R_smooth, kind='cubic',
                    bounds_error=False, fill_value='extrapolate')

    # on récupère Δλ "basique" = pas de la grille
    delta = lam_arr[1] - lam_arr[0]

    # 2) Calcul du gradient lissé
    dR_smooth   = np.gradient(R_smooth, lam_arr)   # dérivée dR/dλ du spectre lissé
    grad_smooth = np.abs(dR_smooth)                # valeurs absolues des pentes


    if not dip_idx_list:
        # pas de dip
        return None, cfg_name, None
    
    # On instancie deux variantes de score :
    #  - best_idx_dn : comparaison via S_R (ΔR/Δn), initialisé à -inf
    best_idx_dn, best_dR = None, -np.inf
    #  - best_idx_raw : comparaison via raw_score(depth, slope, fwhm), initialisé à -inf
    best_idx_raw, best_raw_score = None, -np.inf

    best_Slam, best_dR_half = None, None
    dR_over_dn_list   = [] # pour ΔR/Δn de chaque dip
    dLam_over_dn_list = [] # pour Δλ/Δn de chaque dip
        
    dR_base = np.gradient(reflectance, wavelength)

    # 2) boucle sur chaque candidat dip
    for j in range(len(dip_idx_list)):
        depth = depth_list[j]   # profondeur du j-ième dip
        fwhm  = fwhm_list[j]    # largeur FWHM du j-ième dip
        
        # → Si on a au moins une couche dans `sel_layers`, on effectue la
        #   simulation Δn pour ce dip j :
        if sel_layers:
            # simulate_delta_spectrum retourne aussi S_lam (Δλ/Δn) et S_R (ΔR/Δn)
            Rup_dn, lam0, R0, lam1, R1, S_lam, S_R, dR_half = simulate_delta_spectrum(
                cfg=cfg,
                lam=wavelength,
                wave=wave,
                n_modes=n_modes,
                sel_layers=sel_layers,
                delta_n=delta_n,
                lam_dip=lam_dip_list[j],
                R_dip=R_dip_list[j],
                lam_left=lam_left_list[j],
                lam_right=lam_right_list[j],
                base_spectrum=reflectance,
                json_combined_path=json_combined_path,
                dip_index=j,
                mode=mode
            )
            dR_over_dn_list.append(S_R)
            dLam_over_dn_list.append(S_lam)

            # MISE À JOUR du meilleur dip en mode Δn (compare S_R)
            if S_R > best_dR:
                best_dR      = S_R
                best_idx_dn  = j
                best_Slam    = S_lam
                best_dR_half = dR_half
        else:
            # Pas de Δn demandé → on calcule raw_score à partir de (depth, slope, fwhm)
            # ─── Calcul de la pente au demi-hauteur ───

            # 1) récupère lam_left_list[j] et lam_right_list[j]
            lam_left  = lam_left_list[j]
            lam_right = lam_right_list[j]

            # 2) trouve l’indice entier le plus proche de lam_left et lam_right dans lam_arr
            idx_left  = np.argmin(np.abs(lam_arr - lam_left))
            idx_right = np.argmin(np.abs(lam_arr - lam_right))
            # (a) pente sur le flanc gauche au demi-hauteur :
            y_plus_L  = interp_R(lam_left + delta)
            y_minus_L = interp_R(lam_left - delta)
            slope_left  = abs((y_plus_L - y_minus_L) / (2 * delta))

            # (b) pente sur le flanc droit au demi-hauteur :
            y_plus_R  = interp_R(lam_right + delta)
            y_minus_R = interp_R(lam_right - delta)
            slope_right = abs((y_plus_R - y_minus_R) / (2 * delta))

            # (c) on retient la pente la plus raide parmi les deux flancs :
            slope = max(slope_left, slope_right)



            # Paramètres de pondération que vous avez donnés :
            alpha = 2.0

            # Calcul du raw_score
            #   Attention : depth**alpha augmente le poids de la profondeur
            raw_score = (depth**alpha) * (slope) / (fwhm)

            # Mettre à jour le meilleur dip au sens de raw_score
            if raw_score > best_raw_score:
                best_raw_score = raw_score
                best_idx_raw   = j

            # On stocke toutefois S_R=0 et S_lam=0 dans les listes pour la cohérence des tableaux
            dR_over_dn_list.append(0.0)
            dLam_over_dn_list.append(0.0)

    # 3) Fin de la boucle : si sel_layers non vide, on retient best_idx_dn ;
    #    sinon, on retient best_idx_raw
    if sel_layers:
        best_idx = best_idx_dn
        best_dR   = best_dR
        # best_Slam et best_dR_half ont déjà été mis à jour dans la boucle
    else:
        best_idx = best_idx_raw
        best_dR   = 0.0       # S_R n’a pas de sens ici
        best_Slam = 0.0       # Δλ/Δn = 0, pas de Δn appliqué
        best_dR_half = 0.0

    if best_idx is None:
        if verbose:
            print(f"[find_best_dip] Aucun dip retenu pour «{cfg_name}»")
        return None, cfg_name, None

    # 4) Extraction des données du creux sélectionné
    lam_left  = lam_left_list[best_idx]
    lam_right = lam_right_list[best_idx]
    fwhm      = fwhm_list[best_idx]
    depth     = depth_list[best_idx]
    lam_dip   = lam_dip_list[best_idx]
    R_dip     = R_dip_list[best_idx]
    ylev      = y_level_list[best_idx]
    lam_max_l = lam_max_l_list[best_idx]; R_max_l = R_max_l_list[best_idx]
    lam_max_r = lam_max_r_list[best_idx]; R_max_r = R_max_r_list[best_idx]
    lam_sym   = lam_sym_list[best_idx];   R_sym   = R_sym_list[best_idx]

    # 5) Montage du tuple de sortie comme avant
    out = (
        lam_left, lam_right,
        fwhm, depth,
        lam_dip, R_dip, ylev,
        lam_max_l, R_max_l,
        lam_max_r, R_max_r,
        lam_sym, R_sym,
        best_dR,    # ΔR/Δn du creux retenu (sera 0 si pas de Δn)
        best_Slam,  # Δλ/Δn du creux retenu (sera 0 si pas de Δn)
        best_dR_half,
        dip_idx_list,
        dR_over_dn_list,
        dLam_over_dn_list
    )
    return out, cfg_name, best_idx



def compute_half_point(lam, R, lam_left, lam_right):
    """
    Retourne le λ du point demi-hauteur, en choisissant 
    entre lam_left et lam_right selon la pente la plus forte.
    
    lam : array de longueurs d'onde
    R   : array de réflectances (même longueur que lam)
    lam_left, lam_right : bornes du demi-hauteur
    """
    step = lam[1] - lam[0]
    interp = interp1d(lam, R, kind='cubic', bounds_error=False, fill_value='extrapolate')
    slope_l = (interp(lam_left + step)  - interp(lam_left - step))  / (2*step)
    slope_r = (interp(lam_right + step) - interp(lam_right - step)) / (2*step)
    return lam_left if abs(slope_l) > abs(slope_r) else lam_right