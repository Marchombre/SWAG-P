import numpy as np
from scipy.signal import savgol_filter, find_peaks, peak_widths

def find_best_dip_fwhm(wavelength, reflectance,
                       smooth_win, polyorder,
                       dip_prom, dip_dist,
                       peak_dist, verbose=False):
    """
    Renvoie :
      lam_left, lam_right, fwhm,        # bords et largeur FWHM du dip retenu
      lam_dip, R_dip, y_level,          # position et profondeur du dip retenu
      lam_max_l, R_max_l,               # pic gauche adjacent
      lam_max_r, R_max_r,               # pic droit adjacent
      lam_sym, R_sym,                   # point symétrique du plus petit max
      slope, raw_score,                 # métriques internes
      dips, scores_list                 # pour debug
    """
    lam = np.asarray(wavelength)
    R   = np.asarray(reflectance)

    # 1) lissage optionnel
    if smooth_win > 1:
        R_s = savgol_filter(R, smooth_win, polyorder)
    else:
        R_s = R.copy()

    # 2) inversion et détection de tous les dips
    inv = -R_s
    dips, props = find_peaks(inv, prominence=dip_prom, distance=dip_dist)
    lam_max_ls = []; R_max_ls = []
    lam_max_rs = []; R_max_rs = []
    lam_syms   = []; R_syms   = []
    scores_list = []  # pour debugger chaque dip

    if dips.size == 0:
        dips = np.array([np.argmin(R_s)])
        props = {"prominences": np.array([0.0])}

    # Pré-calculs partagés
    grad      = np.abs(np.gradient(R_s, lam))
    dR        = np.gradient(R_s, lam)
    zc        = np.where((dR[:-1] > 0) & (dR[1:] < 0))[0]
    widths_px, height_p, left_ips, right_ips = peak_widths(inv, dips, rel_height=0.5)
    lam_lefts  = np.interp(left_ips,  np.arange(len(lam)), lam)
    lam_rights = np.interp(right_ips, np.arange(len(lam)), lam)

    best_score = -np.inf
    best       = {}
    # listes pour debug complet
    depths = []
    slopes = []
    widths = []


    # 3) boucle sur chaque dip pour le scorer
    for j, i0 in enumerate(dips):
        # j = index de la boucle (0, 1, 2, …), i0 = position approximative du dip dans le tableau
        # « enumerate » permet de traiter chaque dip et de se souvenir de son rang j.

        # 3a) affinement du dip sur le signal brut
        lo, hi = max(0, i0-1), min(len(R), i0+2)
        # on définit une petite fenêtre [i0-1, i0+1] autour de l’indice brut i0
        #  → on recadre dans les bornes valides du tableau

        argmin_l = np.argmin(R[lo:hi])
        # on cherche la position du minimum absolu dans cette fenêtre réduite
        #  « argmin_l » est un index relatif à [lo:hi]

        dip_idx = lo + argmin_l
        # on convertit cet index relatif en index absolu dans R

        lam_dip, R_dip = lam[dip_idx], R[dip_idx]
        # lam_dip = valeur de longueur d’onde au creux
        # R_dip   = réflectance brute au creux (dip)
        
        #---------------------------------------------------------
        # 3b) recherche des maxima adjacents
        # combine zero-crossing (dR +→−) et détection d’aplatissement brusque
        #----------------------------------------------------------
        
        # — côté gauche du dip —
        zc_left = zc[zc < dip_idx]
        # zc = indices où la dérivée s’inverse (+→−), donc candidats maxima
        # zc_left = ceux qui sont avant dip_idx

        dR_left = dR[:dip_idx]
        # portion de dérivée avant le dip, pour y chercher un aplatissement

 
        # détection des chutes relatives de pente ≥50 % sur ≥5 points à gauche
        red_factor = 0.5  # on cherche une réduction d’au moins 50 % de la pente
        # diffs_rel_left[i] = (pente[i+1]  – pente[i]) / pente[i]
        diffs_rel_left = (dR_left[1:] - dR_left[:-1]) / np.maximum(np.abs(dR_left[:-1]), 1e-8)
        cond_left = diffs_rel_left < -red_factor
        
        
        flat_left = []
        i = 0
        while i < len(cond_left):
            if cond_left[i]:
                j = i
                while j+1 < len(cond_left) and cond_left[j+1]:
                    j += 1
                # si la séquence dure au moins 5 points, on retient le dernier point
                if (j - i + 1) >= 5:
                    # +1 pour passer de diffs_rel_left→dR_left index
                    flat_left.append(i+1)
                i = j + 1
            else:
                i += 1

        flat_left = np.array(flat_left)


        
        # --- détection des plateaux (pente ≃ 0) à gauche ---
        # on prend ici 10% de la pente max comme seuil de “quasi-plat”
        plat_thr = 0.1 * np.nanmax(np.abs(dR_left)) if dR_left.size else 0
        cond_plateau = np.abs(dR_left) < plat_thr
        plateaux = []
        i = 0
        while i < len(cond_plateau):
            if cond_plateau[i]:
                j = i
                while j+1 < len(cond_plateau) and cond_plateau[j+1]:
                    j += 1
                if (j - i + 1) >= 5:
                    # +1 pour convertir l’indice de diff→indice de dR_left
                    plateaux.append(i+1)
                i = j + 1
            else:
                i += 1
        # fusionner chutes brutales + vrais plateaux
        flat_left = np.unique(np.concatenate([flat_left, plateaux]))


        if zc_left.size and flat_left.size:
            lm = max(zc_left.max(), flat_left.max())
            # si on a à la fois un zero-crossing et un aplatissement,
            # on prend celui le plus proche du creux (indice le plus grand)

        elif zc_left.size:
            lm = zc_left.max()
            # sinon si seulement zero-crossing, on le prend

        elif flat_left.size:
            lm = flat_left.max()
            # sinon si seulement plat brutal, on le prend

        else:
            if dip_idx > 0:
                lm = int(np.argmax(R[:dip_idx]))
            else:
                lm = 0




        # — côté droit du dip —
        zc_right = zc[zc > dip_idx]
        # indices où dR inverse avant le dip_idx (dérivée +→−) après le dip

        dR_right = dR[dip_idx+1:]
        # portion de dérivée après dip pour détection de plat
        
        # (a) plateau dur → pente très faible par rapport au max
        max_slope = np.nanmax(dR_right)
        plat_thr  = 0.2 * max_slope  # ici 20 % du max_slope, ajustez si besoin
        cond_plateau = dR_right < plat_thr
        flat_plateau = []
        i = 0
        while i < len(cond_plateau):
            if cond_plateau[i]:
                j = i
                while j+1 < len(cond_plateau) and cond_plateau[j+1]:
                    j += 1
                if (j - i + 1) >= 5:
                    # dip_idx+1 pour remonter à l’indice global
                    flat_plateau.append(dip_idx + 1 + i)
                i = j + 1
            else:
                i += 1
        flat_plateau = np.array(flat_plateau)

        
        # détection de runs de chute brutale (≥5 points)
        # on veut repérer les endroits où la pente chute de ≥50 % par rapport à sa valeur précédente
        red_factor = 0.5
        # dR_right[i] = pente au point i; diffs_rel[i] = variation relative d’une marche à l’autre
        diffs_rel = (dR_right[1:] - dR_right[:-1]) / np.maximum(dR_right[:-1], 1e-8)
        # cond = True si la pente a diminué de plus de red_factor (i.e. de 50 %)
        cond_right = diffs_rel < -red_factor
        
        flat_right = []
        i = 0
        while i < len(cond_right):
            if cond_right[i]:
                j = i
                while j+1 < len(cond_right) and cond_right[j+1]:
                    j += 1
                if (j - i + 1) >= 5:
                    # dip_idx+1 car diffs_rel[0] correspond au saut dip_idx→dip_idx+1
                    flat_right.append(dip_idx + 1 + j)
                i = j + 1
            else:
                i += 1
        # on concatène
        flat_right = np.unique(
            np.concatenate([ flat_right, flat_plateau ])
        )
        
                
        # --- détection des vrais plateaux (slope ≃ 0) ---
        # on prend ici 10% de la pente max comme seuil de “quasi-plat”
        plat_thr = 0.1 * np.nanmax(np.abs(dR_right)) if dR_right.size else 0
        cond_plateau = np.abs(dR_right) < plat_thr
        plateaux = []
        i = 0
        while i < len(cond_plateau):
            if cond_plateau[i]:
                j = i
                while j+1 < len(cond_plateau) and cond_plateau[j+1]:
                    j += 1
                if (j - i + 1) >= 5:
                    # conversion en indice global
                    plateaux.append(dip_idx + 2 + j)
                i = j + 1
            else:
                i += 1
        flat_right = np.unique(np.concatenate([flat_right, plateaux]))

        

        if zc_right.size and flat_right.size:
            rm = min(zc_right.min(), flat_right.min())
            # si on a les deux, on prend le plus proche du creux (indice le plus petit)

        elif zc_right.size:
            rm = zc_right.min()
            # sinon zéro-crossing uniquement

        elif flat_right.size:
            rm = flat_right.min()
            # sinon plat brutal uniquement

        else:
            if dip_idx < len(R) - 1:
                offset = np.argmax(R[dip_idx+1:])
                rm = int(dip_idx + 1 + offset)
            else:
                rm = dip_idx


        # s’assurer que lm et rm sont bien des entiers scalaires
        lm = int(lm)
        rm = int(rm)

        lam_max_l, R_max_l = lam[lm], R[lm]
        lam_max_r, R_max_r = lam[rm], R[rm]
        # on extrait les vraies positions et hauteurs des maxima adjacents, brut (R)

        
        # stocker pour debug
        lam_max_ls.append(lam_max_l); R_max_ls.append(R_max_l)
        lam_max_rs.append(lam_max_r); R_max_rs.append(R_max_r)

        # 3c) symétrie du plus petit max
        if R_max_l < R_max_r:
            lam_min = lam_max_l
            y_small = R_max_l
            seg_lam = lam[dip_idx:rm+1]
            seg_R   = R[dip_idx:rm+1]
            lam_sym = np.interp(y_small, seg_R, seg_lam)
        else:
            lam_min = lam_max_r
            y_small = R_max_r
            seg_lam = lam[lm:dip_idx+1][::-1]
            seg_R   = R[lm:dip_idx+1][::-1]
            lam_sym = np.interp(y_small, seg_R, seg_lam)
        R_sym = y_small

        # stocker pour debug
        lam_syms.append(lam_sym); R_syms.append(R_sym)
        
        
        # 3d) profondeur et pente
        depth = y_small - R_dip
        slope = grad[lm] + grad[rm]
        depths.append(depth)
        slopes.append(slope)

        # --- CALCUL DE LA FWHM BASÉ SUR LA DEMI-HAUTEUR ENTRE LE PIC MINIMUM ET SON SYMÉTRIQUE ---
        half_level = R_dip + 0.5 * (y_small - R_dip)
        if lam_min < lam_sym:
            seg_R1       = R_s[lm:dip_idx+1]
            seg_lam1     = lam[lm:dip_idx+1]
            lam_min_fwhm = np.interp(half_level, seg_R1[::-1], seg_lam1[::-1])
            seg_R2       = R_s[dip_idx:rm+1]
            seg_lam2     = lam[dip_idx:rm+1]
            lam_sym_fwhm = np.interp(half_level, seg_R2, seg_lam2)
        else:
            seg_R1       = R_s[dip_idx:rm+1]
            seg_lam1     = lam[dip_idx:rm+1]
            lam_min_fwhm = np.interp(half_level, seg_R1, seg_lam1)
            seg_R2       = R_s[lm:dip_idx+1]
            seg_lam2     = lam[lm:dip_idx+1]
            lam_sym_fwhm = np.interp(half_level, seg_R2[::-1], seg_lam2[::-1])

        # s'assurer que lam_left < lam_right
        lam_left, lam_right = sorted((lam_min_fwhm, lam_sym_fwhm))
        fwhm = lam_right - lam_left
        widths.append(fwhm)

        # 3e) score = profondeur (au carré) × pente / largeur  
        # ➔ plus le dip est profond, moins la pente plate est pénalisée
        alpha = 2.0     # accentue le rôle de la profondeur
        beta  = 0.5     # /sqrt(fwhm)
        raw_score = (depth**alpha) * (slope**(1.0 - depth)) / (fwhm**beta)

        scores_list.append(raw_score)

        # 3f) mise à jour si meilleur score
        if raw_score > best_score:
            best_score = raw_score
            best = {
                "lam_left":  lam_left,
                "lam_right": lam_right,
                "fwhm":      fwhm,
                "lam_dip":   lam_dip,
                "R_dip":     R_dip,
                "lam_max_l": lam_max_l,
                "R_max_l":   R_max_l,
                "lam_max_r": lam_max_r,
                "R_max_r":   R_max_r,
                "lam_sym":   lam_sym,
                "R_sym":     R_sym,
                "slope":     slope,
                "depth":     depth,
                "raw_score": raw_score,
            }

    # 4) niveau à mi-hauteur pour tracer la barre
    y_level = best["R_dip"] + 0.5 * (min(best["R_max_l"], best["R_max_r"]) - best["R_dip"])

    return (
        best["lam_left"], best["lam_right"], best["fwhm"],
        best["lam_dip"], best["R_dip"], y_level,
        best["lam_max_l"], best["R_max_l"],
        best["lam_max_r"], best["R_max_r"],
        best["lam_sym"], best["R_sym"],
        best["slope"], best["depth"],
        best["raw_score"],
        dips, scores_list, 
        depths, slopes, widths,
        lam_max_ls, R_max_ls,
        lam_max_rs, R_max_rs,
        lam_syms, R_syms,
    )

# normalisation Min–Max
def minmax(x):
    a = np.array(x, dtype=float)
    mn, mx = a.min(), a.max()
    return (a - mn) / (mx - mn) if mx > mn else np.zeros_like(a)
