
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Chemins (relatifs au dossier du projet par défaut)
# ---------------------------------------------------------------------------
PLY_PATH = Path("data/splat1.ply")
POSES_BOUNDS_PATH = "YOUR PATH "
IMAGES_DIR = "YOUR PATH "
OUTPUT_DIR = Path("outputs")

# ---------------------------------------------------------------------------
# Transformation appliquée par nerfstudio (dataparser_transforms.json)
# lors de l'entraînement splatfacto de la scène "room".
# Elle permet d'exprimer les poses COLMAP dans le repère du fichier .ply.
# ---------------------------------------------------------------------------
NERFSTUDIO_TRANSFORM = np.array([
    [0.9988114833831787, 0.04734036698937416, 0.011596380732953548, -0.07592017948627472],
    [0.011596380732953548, -0.46190759539604187, 0.8868522644042969, 0.12564101815223694],
    [0.04734036698937416, -0.8856637477874756, -0.46190759539604187, -0.01871594227850437],
])
SCALE_FACTOR = 0.1720844588717592

# ---------------------------------------------------------------------------
# Caméra virtuelle utilisée pour le rendu gsplat
# ---------------------------------------------------------------------------
WIDTH, HEIGHT = 512, 512
FX, FY = 400.0, 400.0

# ---------------------------------------------------------------------------
# Bornes des angles d'Euler (degrés) tirés dans les environnements 6 DOF
# ---------------------------------------------------------------------------
PITCH_BOUNDS = (-60, 45)
ROLL_BOUNDS = (-120, 100)
