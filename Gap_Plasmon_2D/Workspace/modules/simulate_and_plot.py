# simulate_and_plot.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from simulate_reflectance import simulate_reflectance

# Path Workspace
workspace_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))

# Path to the workspace directory
figures_dir = os.path.join(workspace_dir, "Figures")

# Path to save the figure
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
fig_path = os.path.join(figures_dir, f"reflectance_simulation_{timestamp}.png")



def run_simulation(lambda_range, n_mod, geometry, wave, materials_config, json_path):
    """
    Executes the reflectance simulation over a range of wavelengths and displays the result,
    along with a summary table of the geometric parameters and the materials configuration.
    
    Parameters
    ----------
    lambda_range : array_like
        Range of wavelengths (in nm).
    n_mod : int
        Number of RCWA modes.
    geometry : dict
        Dictionary defining the system's geometry.
    wave : dict
        Dictionary of wave parameters (angle, polarization, etc.).
    materials_config : DataFrame
        Materials configuration (from the dropdowns, MATERIALS_CONFIG).
    json_path : str
        Path to the JSON file containing ExpData.
    
    Returns
    -------
    Rup_values, Rdown_values : lists
        Reflectance values computed for each wavelength.
    """
    # Run the simulation using the existing function
    Rup_values, Rdown_values = simulate_reflectance(lambda_range, geometry, wave, materials_config, json_path, n_mod)
    
    # Create the reflectance plot
    plt.figure(figsize=(10, 6))
    plt.plot(lambda_range, Rup_values, 'o-', label='Rup')
    plt.plot(lambda_range, Rdown_values, 's-', label='Rdown')
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.legend()
    plt.title("Reflectance Simulation")
    plt.grid(True)
    
    # Prepare the summary table for the geometry parameters
    geom_df = pd.DataFrame(list(geometry.items()), columns=['Geometric Parameter', 'Value'])
    # Prepare the table for the materials configuration
    mat_df = materials_config.copy()  # columns: key and material
    
    cellText_geom = geom_df.values.tolist()
    cellText_mat = mat_df.values.tolist()
    
    # Display the geometry table at the bottom left
    table_geom = plt.table(cellText=cellText_geom, colLabels=geom_df.columns,
                           loc='bottom', bbox=[0, -0.45, 0.5, 0.3])
    table_geom.auto_set_font_size(False)
    table_geom.set_fontsize(8)
    
    # Display the materials configuration table at the bottom right
    table_mat = plt.table(cellText=cellText_mat, colLabels=mat_df.columns,
                          loc='bottom', bbox=[0.5, -0.45, 0.5, 0.3])
    table_mat.auto_set_font_size(False)
    table_mat.set_fontsize(8)
    
    # Add titles for the tables
    plt.text(0.25, -0.5, 'Geometric Parameters', ha='center', fontsize=13, transform=plt.gca().transAxes)
    plt.text(0.75, -0.5, 'Materials Configuration', ha='center', fontsize=13, transform=plt.gca().transAxes)
    
    plt.subplots_adjust(bottom=0.3)


    plt.show()
    plt.savefig(fig_path)
    return Rup_values, Rdown_values
