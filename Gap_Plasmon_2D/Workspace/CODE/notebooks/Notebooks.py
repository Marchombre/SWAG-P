import sys
import os

# Ajoute le chemin du dossier 'modules'
module_path = os.path.abspath(os.path.join('..', 'modules'))
if module_path not in sys.path:
    sys.path.append(module_path)

from config import *      # ou import config
from calculs import *     # ou import calculs
from affichage import *   # ou import affichage

# Exemple d'utilisation
resultat = calculs.effectuer_calcul(42)
affichage.afficher_resultat(resultat)
