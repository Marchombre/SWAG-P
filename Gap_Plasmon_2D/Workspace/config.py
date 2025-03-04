import PyMoosh as pm
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.linalg import toeplitz, inv

import json


from scipy.special import wofz


### Chargement des matériaux depuis le fichier JSON
with open('/home/chardon-grossard/Bureau/SWAG-P/Gap_Plasmon_2D/Workspace/data/material_data.json') as file:
    data = json.load(file)