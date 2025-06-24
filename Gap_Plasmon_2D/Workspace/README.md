# SWAG-P / Gap_Plasmon_2D

Modélisation – optimisation – caractérisation de structures gap-plasmon 2D
(interfaces métal / diélectrique) — simulations RCWA, post-traitement et GUI
intéractive sous Jupyter/Voila.





## Fonctionnalités principales
| Bloc | Description |
|------|-------------|
| **Simulation** | RCWA mono / multi-mode, spectres Rup/Rdown, Δn. |
| **Optimisation** | Recherche de géométrie : max (ΔR / Δn) ou max (Δλ / Δn). |
| **Analyse** | Extraction FWHM, λ<sub>0</sub>, Q-factor, tableaux comparatifs. |
| **Interface** | Widgets *ipywidgets* (onglets Simulation, Optimisation, Materials). |
| **Base matériaux** | Catalogues *n,k* / *n²* YAML + fetch dynamique sur refractiveindex.info. |
| **Export** | PNG, CSV, HDF5 (`simulation_results.h5`). |


