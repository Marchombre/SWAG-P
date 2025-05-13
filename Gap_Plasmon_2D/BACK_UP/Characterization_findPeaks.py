def find_best_dip_fwhm(wavelength, reflectance,
                       smooth_win, polyorder,
                       dip_prom, dip_dist,
                       peak_dist, verbose=False, cfg_name=None):
    """
    Renvoie :
      lam_left, lam_right, fwhm,        # bords et largeur FWHM du dip retenu
      lam_dip, R_dip, y_level,          # position et profondeur du dip retenu
      lam_max_l, R_max_l,               # pic gauche adjacent
      lam_max_r, R_max_r,               # pic droit adjacent
      lam_sym, R_sym,                   # point symétrique du plus petit max
      slope, raw_score,                 # métriques internes
      dips, scores_list,                # pour debug
      depths, slopes, widths,           # listes debug
      lam_max_ls, R_max_ls,             # positions brutes des max gauche
      lam_max_rs, R_max_rs,             # positions brutes des max droite
      lam_syms, R_syms                  # symétrie debug
    """
    import numpy as np
    from scipy.signal import savgol_filter, find_peaks, peak_widths

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

    # cas "pas de dip"
    if dips.size == 0:
        if verbose:
            print(f"[find_best_dip_fwhm] Aucun dip détecté pour « {cfg_name} »")
        return None, cfg_name

    # initialisation des listes debug
    lam_max_ls = []; R_max_ls = []
    lam_max_rs = []; R_max_rs = []
    lam_syms   = []; R_syms   = []
    scores_list = []
    depths = []; slopes = []; widths = []

    # pré-calculs pour score & FWHM manuel
    grad      = np.abs(np.gradient(R_s, lam))
    dR        = np.gradient(R_s, lam)
    widths_px, height_p, left_ips, right_ips = peak_widths(inv, dips, rel_height=0.5)
    lam_lefts  = np.interp(left_ips,  np.arange(len(lam)), lam)
    lam_rights = np.interp(right_ips, np.arange(len(lam)), lam)

    best_score = -np.inf
    best       = {}

    # 3) boucle sur chaque dip
    for j, i0 in enumerate(dips):
        # 3a) affinement du dip sur le spectre brut
        lo, hi = max(0, i0-1), min(len(R), i0+2)
        argmin_l = np.argmin(R[lo:hi])
        dip_idx = lo + argmin_l
        lam_dip, R_dip = lam[dip_idx], R[dip_idx]

        # interpolation parabolique 3 points pour affiner lam_dip
        i = dip_idx
        if 0 < i < len(R)-1:
            y0, y1, y2 = R[i-1], R[i], R[i+1]
            a = (y0 + y2)/2 - y1
            b = (y2 - y0)/2
            if a != 0:
                delta = -b/(2*a)
                lam_dip_fine = lam[i] + delta*(lam[1]-lam[0])
                R_dip_fine   = y1 - b*delta/2
                lam_dip, R_dip = lam_dip_fine, R_dip_fine

        # -------------------------------------------------------------
        # 3b) recherche des maxima adjacents avec détection de semi-plateaux
        peaks, _ = find_peaks(R_s, prominence=dip_prom, distance=peak_dist)
        dip_i = dip_idx

        # ----- gauche -----
        left_peaks = peaks[peaks < dip_i]
        dR_left = dR[:dip_i]
        flat_left = []
        if dR_left.size:
            max_slope_l = np.nanmax(np.abs(dR_left))
            thr_l = 0.2 * max_slope_l
            cond_plateau_l = np.abs(dR_left) < thr_l
            i_l = 0
            while i_l < len(cond_plateau_l):
                if cond_plateau_l[i_l]:
                    j2 = i_l
                    while j2+1 < len(cond_plateau_l) and cond_plateau_l[j2+1]:
                        j2 += 1
                    if (j2 - i_l + 1) >= 5:
                        flat_left.append(i_l)
                    i_l = j2 + 1
                else:
                    i_l += 1
        candidates_l = np.concatenate([left_peaks, flat_left]) if (left_peaks.size or flat_left) else np.array([])
        if candidates_l.size:
            lm = int(candidates_l.max())
        else:
            lm = 0
        lam_max_l, R_max_l = lam[lm], R[lm]
        lam_max_ls.append(lam_max_l); R_max_ls.append(R_max_l)

        # ----- droite -----
        right_peaks = peaks[peaks > dip_i]
        dR_right = dR[dip_i+1:]
        flat_right = []
        if dR_right.size:
            max_slope_r = np.nanmax(np.abs(dR_right))
            thr_r = 0.2 * max_slope_r
            cond_plateau_r = np.abs(dR_right) < thr_r
            i_r = 0
            while i_r < len(cond_plateau_r):
                if cond_plateau_r[i_r]:
                    j2 = i_r
                    while j2+1 < len(cond_plateau_r) and cond_plateau_r[j2+1]:
                        j2 += 1
                    if (j2 - i_r + 1) >= 5:
                        flat_right.append(dip_i + 1 + i_r)
                    i_r = j2 + 1
                else:
                    i_r += 1
        candidates_r = np.concatenate([right_peaks, flat_right]) if (right_peaks.size or flat_right) else np.array([])
        if candidates_r.size:
            rm = int(candidates_r.min())
        else:
            rm = len(R_s) - 1
        lam_max_r, R_max_r = lam[rm], R[rm]
        lam_max_rs.append(lam_max_r); R_max_rs.append(R_max_r)

        # 3c) point symétrique du plus petit max
        if R_max_l < R_max_r:
            y_small = R_max_l
            seg_lam = lam[dip_idx:rm+1]
            seg_R   = R_s[dip_idx:rm+1]
        else:
            y_small = R_max_r
            seg_lam = lam[lm:dip_idx+1][::-1]
            seg_R   = R_s[lm:dip_idx+1][::-1]
        lam_sym = np.interp(y_small, seg_R, seg_lam)
        R_sym   = y_small
        lam_syms.append(lam_sym); R_syms.append(R_sym)

        # 3d) profondeur et pente
        depth = y_small - R_dip
        slope = grad[lm] + grad[rm]
        depths.append(depth)
        slopes.append(slope)

        # 3e) calcul du FWHM manuel entre dip et symétrique
        half_level = R_dip + 0.5 * (y_small - R_dip)
        if R_max_l < R_max_r:
            seg_R1   = R_s[dip_idx:rm+1]
            seg_lam1 = lam[dip_idx:rm+1]
            lam_min_fwhm = np.interp(half_level, seg_R1, seg_lam1)
            seg_R2   = R_s[lm:dip_idx+1]
            seg_lam2 = lam[lm:dip_idx+1]
            lam_sym_fwhm = np.interp(half_level, seg_R2[::-1], seg_lam2[::-1])
        else:
            seg_R1   = R_s[lm:dip_idx+1]
            seg_lam1 = lam[lm:dip_idx+1]
            lam_min_fwhm = np.interp(half_level, seg_R1[::-1], seg_lam1[::-1])
            seg_R2   = R_s[dip_idx:rm+1]
            seg_lam2 = lam[dip_idx:rm+1]
            lam_sym_fwhm = np.interp(half_level, seg_R2, seg_lam2)
        lam_left, lam_right = sorted((lam_min_fwhm, lam_sym_fwhm))
        fwhm = lam_right - lam_left
        widths.append(fwhm)

        # 3f) score
        alpha = 2.0
        beta  = 0.5
        raw_score = (depth**alpha) * (slope**(1.0 - depth)) / (fwhm**beta)
        scores_list.append(raw_score)

        # 3g) mise à jour du meilleur
        if raw_score > best_score:
            best_score = raw_score
            best = {
                "lam_left":   lam_left,
                "lam_right":  lam_right,
                "fwhm":       fwhm,
                "lam_dip":    lam_dip,
                "R_dip":      R_dip,
                "lam_max_l":  lam_max_l,
                "R_max_l":    R_max_l,
                "lam_max_r":  lam_max_r,
                "R_max_r":    R_max_r,
                "lam_sym":    lam_sym,
                "R_sym":      R_sym,
                "slope":      slope,
                "depth":      depth,
                "raw_score":  raw_score
            }

    # 4) calcul du niveau à mi-hauteur pour tracer la barre
    if not best:
        if verbose:
            print(f"[find_best_dip_fwhm] Aucun dip retenu pour « {cfg_name} »")
        return None, cfg_name

    y_level = best["R_dip"] + 0.5 * (min(best["R_max_l"], best["R_max_r"]) - best["R_dip"])

    return (
        best["lam_left"], best["lam_right"], best["fwhm"],
        best["lam_dip"],  best["R_dip"],   y_level,
        best["lam_max_l"], best["R_max_l"],
        best["lam_max_r"], best["R_max_r"],
        best["lam_sym"],   best["R_sym"],
        best["slope"],     best["depth"],
        best["raw_score"],
        dips, scores_list,
        depths, slopes, widths,
        lam_max_ls, R_max_ls,
        lam_max_rs, R_max_rs,
        lam_syms,   R_syms
    ), cfg_name
