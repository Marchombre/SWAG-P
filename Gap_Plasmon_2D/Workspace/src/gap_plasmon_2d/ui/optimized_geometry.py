import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from copy import deepcopy

# --- réutilise exactement les aides du module interactif --------------------
from gap_plasmon_2d.ui.geometry_settings import displayed_thickness, draw_layer, geometry_config



def _adjust_sub_super(d_sub, d_sup, p, *, min_central_ratio=0.70):
    max_subsuper = p * (1.0 - min_central_ratio)
    total = d_sub + d_sup
    if total <= 0 or total <= max_subsuper:          # rien à corriger
        return d_sub, d_sup
    k = max_subsuper / total                         # facteur < 1
    return d_sub * k, d_sup * k




# --------------------------------------------------------------------------- 
def plot_geometry_static_from_run(ax,
                                  keys, best_vec,
                                  fixed_vals=None,
                                  *, default_geom: dict = geometry_config,
                                  ax_offset=(0.0, 0.0)):

    
    # ---------- 1) reconstitution complète ----------------------------------
    # 1) on part de l’ordre exact du JSON (default_geom)
    base = {k: 0.0 for k in (default_geom or {}).keys()}   # ← ordre préservé

    # 2) on complète avec les clés manquantes (celles de geometry_config)
    for k in geometry_config:
        base.setdefault(k, 0.0)    # n’ajoute que si absent → pas de casse d’ordre

    
    
    for k, v in (fixed_vals or {}).items():
        base[k] = float(v)
    
    for k, v in zip(keys, best_vec):
        base[k] = float(v)

    geom = base        # on travaille ensuite avec geom …


    for k, v in (fixed_vals or {}).items():
        geom[k] = float(v)
    for k, v in zip(keys, best_vec):
        geom[k] = float(v)

    # ---------- 2) extraction des valeurs -----------------------------------
    p        = geom.get("period", 100.0)
    t_sub    = geom.get("thick_sub", 0.0)
    t_super  = geom.get("thick_super", 0.0)

    # (t_acc, t_XIAOYI … identiques)
    t_acc    = geom.get("thick_accroche", 0.0)
    t_XIAOYI = geom.get("thick_XIAOYI",    0.0)
    t_metal  = geom.get("thick_metalliclayer", 0.0)
    t_gap    = geom.get("thick_gap", 0.0)
    t_reso   = geom.get("thick_reso", 0.0)
    t_diel   = geom.get("thick_diel", 0.0)
    t_func   = geom.get("thick_func", 0.0)
    t_mol    = geom.get("thick_mol", 0.0)
    w_reso   = geom.get("width_reso", 0.0)

    extra_keys = [
        k for k in geom                 # l'ordre du dict est conservé
        if k.startswith("thick_homo_") and geom[k] > 0.0
    ]

    t_extras   = [geom[k] for k in extra_keys]

    # ---------- 3) Substrate / Superstrate (70 % mini pour le reste) --------
    disp_sub_raw   = displayed_thickness(t_sub)
    disp_super_raw = displayed_thickness(t_super)
    disp_sub, disp_super = _adjust_sub_super(
        disp_sub_raw, disp_super_raw, p, min_central_ratio=0.70
    )

    # ---------- 4) constitution de la liste SANS couches à 0 nm ------------
    layer_pairs = [
        ("thick_accroche",    t_acc),
        ("thick_XIAOYI",      t_XIAOYI),
        *[(k, geom[k]) for k in extra_keys],      # homo_*
        ("thick_metalliclayer", t_metal),
        ("thick_gap",           t_gap),
        ("thick_reso",          t_reso),
        ("thick_diel",          t_diel),
        ("thick_func",          t_func),
        ("thick_mol",           t_mol),
    ]
    # ne garde que celles > 0
    layer_pairs = [(k, v) for k, v in layer_pairs if v > 0]

    layer_keys, layer_real = zip(*layer_pairs) if layer_pairs else ([], [])

    # ---------- 5) rescale global -------------------------------------------
    H_avail = p - (disp_sub + disp_super)
    h_min   = max(0.5, 0.02 * H_avail)
    disp_layers = _rescale_with_min(layer_real, H_avail, h_min)


    # dépaquetage dynamique
    disp_dict = dict(zip(layer_keys, disp_layers))

    disp_acc    = disp_dict.get("thick_accroche",     0.0)
    disp_x      = disp_dict.get("thick_XIAOYI",       0.0)
    disp_metal  = disp_dict.get("thick_metalliclayer",0.0)
    disp_gap    = disp_dict.get("thick_gap",          0.0)
    disp_reso   = disp_dict.get("thick_reso",         0.0)
    disp_dielectric = disp_dict.get("thick_diel",     0.0)
    disp_func   = disp_dict.get("thick_func",         0.0)
    disp_mol    = disp_dict.get("thick_mol",          0.0)
    disp_extra = [disp_dict[k] for k in extra_keys]


    # ───────── Ajuste la largeur affichée du nanocube pour qu’il reste carré ──────
    if t_reso > 0:
        scale_h      = disp_reso / t_reso        # même facteur que l’axe vertical
        w_reso_disp  = w_reso * scale_h          # largeur affichée
    else:
        w_reso_disp  = w_reso                    # cube absent → on garde w_reso

    cx      = (p - w_reso_disp) / 2              # nouvelle abscisse du cube
    lat_w   = cx                                 # largeur des zones latérales



    # ---------- 6) dessin : schéma de la géométrie -----------------------------
    ax.clear()

    # ─── ❶ agrandit SEULEMENT cet axe ------------------------------------------
    box   = ax.get_position()            # [x0, y0, w, h] en coordonnées figure
    scale = 1.5                      # facteur d’agrandissement (≥1)
    new_w = min(box.width  * scale, 1.0 - box.x0 - 0.01)   # reste dans la figure
    new_h = min(box.height * scale, 1.0 - box.y0 - 0.01)
    ax.set_position([box.x0, box.y0, new_w, new_h])

    # déplacement personnalisable
    dx, dy = ax_offset
    move_axes(ax, dx, dy)


    # ─── ❷ petit utilitaire d’étiquette ----------------------------------------
    def _name(label: str, t: float) -> str:
        """Retourne 'Nom : (123.4 nm)' ou '' si t == 0."""
        return f"{label} : ({t:.1f} nm)" if t > 0 else ""

    ax.set_title("Schematics (static)", fontsize=10, pad=4)

    # ---------------------------------------------------------------------------
    #  Substrate
    # ---------------------------------------------------------------------------
    y = 0
    if disp_sub > 0:
        draw_layer(ax, 0, y, p, disp_sub, "brown",  _name("Substrate", t_sub))
        draw_layer(ax, 0, 0, p, min(0.05*p, disp_sub), "none", "", hatch="///")
        y += disp_sub

    # ---------------------------------------------------------------------------
    #  Accroche et XIAOYI
    # ---------------------------------------------------------------------------
    if disp_acc > 0:
        draw_layer(ax, 0, y, p, disp_acc, "orange", _name("Accroche", t_acc))
        y += disp_acc
    if disp_x > 0:
        draw_layer(ax, 0, y, p, disp_x,  "purple", _name("XIAOYI", t_XIAOYI))
        y += disp_x

    # ---------------------------------------------------------------------------
    #  Couches homo_* (si présentes) — couleur unique par couche
    # ---------------------------------------------------------------------------
    color_cycle  = plt.rcParams['axes.prop_cycle'].by_key()['color']
    extra_colors = {k: color_cycle[i % len(color_cycle)]
                    for i, k in enumerate(extra_keys)}

    for k, h_i, t_i in zip(extra_keys, disp_extra, t_extras):
        if h_i <= 0:
            continue
        draw_layer(
            ax, 0, y, p, h_i,
            extra_colors[k],                          # ← couleur spécifique
            _name(k.replace("thick_homo_", "homo_"), t_i)
        )
        y += h_i


    # ---------------------------------------------------------------------------
    #  Couche métallique
    # ---------------------------------------------------------------------------
    if disp_metal > 0:
        draw_layer(ax, 0, y, p, disp_metal,
                "gold", _name("Metallic layer", t_metal))
    y_metal_top = y + disp_metal  # même si disp_metal == 0


    # ---------------------------------------------------------------------------
    #  Gap + Nanocube  (avec w_reso_disp)
    # ---------------------------------------------------------------------------
    if disp_gap > 0:
        draw_layer(ax, cx, y_metal_top,            w_reso_disp, disp_gap,
                "lightgreen", _name("Gap", t_gap))
    if disp_reso > 0:
        draw_layer(ax, cx, y_metal_top + disp_gap, w_reso_disp, disp_reso,
                "silver",     _name("Nanocube", t_reso))
    y_cube_top = y_metal_top + disp_gap + disp_reso


    # ---------------------------------------------------------------------------
    #  Parois latérales : Polymer / Func / Mol
    # ---------------------------------------------------------------------------
    y_lat = y_metal_top

    if disp_dielectric > 0:
        draw_layer(ax, 0, y_lat, lat_w, disp_dielectric,
                "green", _name("Photopolymer", t_diel))
        draw_layer(ax, cx+w_reso_disp, y_lat, lat_w, disp_dielectric,
                "green", "")
        y_lat += disp_dielectric

    if disp_func > 0:
        draw_layer(ax, 0, y_lat, lat_w, disp_func,
                "pink", _name("Functionalisation", t_func))
        draw_layer(ax, cx+w_reso_disp, y_lat, lat_w, disp_func,
                "pink", "")
        y_lat += disp_func

    if disp_mol > 0:
        draw_layer(ax, 0, y_lat, lat_w, disp_mol,
                "violet", _name("Molecule", t_mol))
        draw_layer(ax, cx+w_reso_disp, y_lat, lat_w, disp_mol,
                "violet", "")
        y_lat += disp_mol

    # Environnement latéral éventuel
    lat_fill = y_cube_top - y_lat
    if lat_fill > 0:
        draw_layer(ax, 0,           y_lat, lat_w, lat_fill, "lightblue", "")
        draw_layer(ax, cx + w_reso_disp, y_lat, lat_w, lat_fill, "lightblue", "")

    # ---------------------------------------------------------------------------
    #  Superstrate
    # ---------------------------------------------------------------------------
    sup_h = p - y_cube_top
    if sup_h > 0:
        draw_layer(ax, 0, y_cube_top, p, sup_h,
                "lightblue", _name("Superstrate", t_super))
        draw_layer(ax, 0, p - min(0.05*p, disp_super), p,
                min(0.05*p, disp_super), "none", "", hatch="///")

    # ---------------------------------------------------------------------------
    #  Axes
    # ---------------------------------------------------------------------------
    ax.set_xlim(0, p)
    ax.set_ylim(0, p)
    ax.set_aspect("equal")
    ax.axis("off")



def move_axes(ax, dx=0.0, dy=0.0):
    """
    Décale l'axe d'un offset (dx, dy) en coordonnées figure.
    dx > 0 → vers la droite,  dx < 0 → vers la gauche  
    dy > 0 → vers le haut,    dy < 0 → vers le bas
    """
    x0, y0, w, h = ax.get_position().bounds
    ax.set_position([x0 + dx, y0 + dy, w, h])





def _rescale_with_min(thicknesses, H_avail, h_min):
    """Renvoie les hauteurs d'affichage > h_min tout en respectant Σh = H_avail."""
    h = np.asarray(thicknesses, float)
    if h.sum() == 0:
        return np.zeros_like(h)

    # étape 1 : linéaire
    h = h * (H_avail / h.sum())

    # étape 2 : contrainte h_min
    small = h < h_min
    if not small.any():
        return h                    # rien à faire

    deficit = (h_min - h[small]).sum()
    large   = ~small
    surplus = (h[large] - h_min).sum()

    if surplus <= 0:                # cas pathologique : tout est min
        h[small] = h_min
        h[large] = h_min
        return h * (H_avail / h.sum())

    factor = 1 - deficit / surplus
    h[large] = h_min + (h[large] - h_min) * factor
    h[small] = h_min
    return h    