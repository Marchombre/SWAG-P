# material_selector.py
import ipywidgets as widgets
import pandas as pd
from IPython.display import display
from material_list_provider import get_available_materials

# Default list of roles used in your simulation
DEFAULT_ROLES = [
    "perm_env", "perm_dielec", "perm_sub", "perm_reso",
    "perm_metalliclayer", "perm_accroche", "perm_func", "perm_mol"
]

def create_material_selector(json_path, roles=DEFAULT_ROLES):
    """
    Creates and displays an interactive interface (dropdowns + custom text fields)
    for selecting materials. The list of available materials is dynamically generated
    from the combined JSON file.
    
    Parameters
    ----------
    json_path : str
        Path to the combined JSON file.
    roles : list, optional
        List of roles to use (default is DEFAULT_ROLES).
    
    Returns
    -------
    widget_container : VBox
        A container (VBox) holding all the widgets,
        including the validation button and an output area for the summary.
        The final configuration will be stored in the global variable MATERIALS_CONFIG.
    """
    # Retrieve the dynamic list of available materials
    all_materials = get_available_materials(json_path)
    # Add the options "None" and "Custom"
    all_materials_with_options = ["None", "Custom"] + all_materials

    dropdowns = {}
    text_inputs = {}
    widget_boxes = []
    
    for role in roles:
        dropdown = widgets.Dropdown(
            options=all_materials_with_options,
            description=role,
            style={'description_width': 'initial'}
        )
        text_input = widgets.Text(
            value="",
            description="Custom:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(visibility='hidden')
        )
        
        def on_dropdown_change(change, txt=text_input):
            if change['new'] == "Custom":
                txt.layout.visibility = 'visible'
            else:
                txt.layout.visibility = 'hidden'
                txt.value = ""
                
        dropdown.observe(lambda change, t=text_input: on_dropdown_change(change, t), names='value')
        
        dropdowns[role] = dropdown
        text_inputs[role] = text_input
        
        widget_boxes.append(widgets.HBox([dropdown, text_input]))
    
    button_create_df = widgets.Button(description="Validate Materials Configuration")
    output_df = widgets.Output()
    
    def on_create_df(b):
        config = {"key": [], "material": []}
        for role in roles:
            if dropdowns[role].value == "Custom":
                mat_value = text_inputs[role].value.strip()
                if mat_value == "":
                    mat_value = "None"
            else:
                mat_value = dropdowns[role].value
            config["key"].append(role)
            config["material"].append(mat_value)
        df_config = pd.DataFrame(config)
        with output_df:
            output_df.clear_output()
            print("Selected materials configuration:")
            display(df_config)
        import __main__
        __main__.MATERIALS_CONFIG = df_config
        print("MATERIALS_CONFIG has been defined in the global __main__ space")
    
    button_create_df.on_click(on_create_df)
    
    widget_container = widgets.VBox(widget_boxes + [button_create_df, output_df])
    return widget_container
