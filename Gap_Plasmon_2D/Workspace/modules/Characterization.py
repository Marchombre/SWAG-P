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
      dips, scores_list,                # pour debug
      depths, slopes, widths,           # listes debug
      lam_max_ls, R_max_ls,             # positions brutes des max gauche
      lam_max_rs, R_max_rs,             # positions brutes des max droite
      lam_syms, R_syms                  # symétrie debug
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

    # initialisation des listes debug
    lam_max_ls = []; R_max_ls = []
    lam_max_rs = []; R_max_rs = []
    lam_syms   = []; R_syms   = []
    scores_list = []
    depths = []; slopes = []; widths = []

    if dips.size == 0:
        dips = np.array([np.argmin(R_s)])
        props = {"prominences": np.array([0.0])}

    # Pré-calculs pour score & FWHM manuel
    grad      = np.abs(np.gradient(R_s, lam))
    dR        = np.gradient(R_s, lam)
    widths_px, height_p, left_ips, right_ips = peak_widths(inv, dips, rel_height=0.5)
    lam_lefts  = np.interp(left_ips,  np.arange(len(lam)), lam)
    lam_rights = np.interp(right_ips, np.arange(len(lam)), lam)

    best_score = -np.inf
    best       = {}

    # 3) boucle sur chaque dip
    for j, i0 in enumerate(dips):
        # 3a) affinement du dip sur le signal brut
        lo, hi = max(0, i0-1), min(len(R), i0+2)
        argmin_l = np.argmin(R[lo:hi])
        dip_idx = lo + argmin_l
        lam_dip, R_dip = lam[dip_idx], R[dip_idx]

        # --------------------------------------------------------
        # 3b) recherche des maxima adjacents via find_peaks sur R_s
        # --------------------------------------------------------
        peaks, peak_props = find_peaks(
            R_s,
            prominence=dip_prom,
            distance=peak_dist
        )
        left_peaks  = peaks[peaks < dip_idx]
        right_peaks = peaks[peaks > dip_idx]

        # max gauche = celui juste avant le dip
        if left_peaks.size:
            lm = int(left_peaks.max())
        else:
            lm = 0

        # max droit = celui juste après le dip
        if right_peaks.size:
            rm = int(right_peaks.min())
        else:
            rm = len(R_s)-1

        lam_max_l, R_max_l = lam[lm], R[lm]
        lam_max_r, R_max_r = lam[rm], R[rm]

        # listes debug
        lam_max_ls.append(lam_max_l); R_max_ls.append(R_max_l)
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

        # 3e) calcul du FWHM manuel entre dip_idx et son symétrique
        half_level = R_dip + 0.5 * (y_small - R_dip)
        if R_max_l < R_max_r:
            # gauche = dip→rm, droite = lm→dip
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

    # 4) niveau à mi-hauteur pour la barre graphique
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
        lam_syms, R_syms
    )

def minmax(x):
    a = np.array(x, dtype=float)
    mn, mx = a.min(), a.max()
    return (a - mn)/(mx - mn) if mx > mn else np.zeros_like(a)
