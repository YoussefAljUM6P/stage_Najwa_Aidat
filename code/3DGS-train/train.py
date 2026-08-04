"""Entraînement 3D Gaussian Splatting (splatfacto) sur la scène `room` de Mip-NeRF 360.

    python train.py
    python train.py /autre/chemin/vers/room
"""

import os
import shutil
import subprocess
import sys

from PIL import Image

DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/360_v2/room"
ITERATIONS = 30000
DOWNSCALE = 4


os.environ["MAX_JOBS"] = "2" # Pour éviter de sature la RAM 

# --- 1. Layout COLMAP : Nerfstudio attend colmap/sparse, le dataset livre sparse ---
if os.path.isdir(os.path.join(DATA_PATH, "sparse")):
    os.makedirs(os.path.join(DATA_PATH, "colmap"), exist_ok=True)
    shutil.move(os.path.join(DATA_PATH, "sparse"),
                os.path.join(DATA_PATH, "colmap", "sparse"))
    print("✅ COLMAP déplacé vers colmap/sparse")

# --- 2. Images : 1 px de trop par rapport aux intrinsèques COLMAP / 4 ---
# Le recadrage écrase les fichiers : le marqueur évite de rogner deux fois.
marker = os.path.join(DATA_PATH, ".images_cropped")
if os.path.exists(marker):
    print("✅ Images_4 déjà corrigées")
else:
    img_dir = os.path.join(DATA_PATH, f"images_{DOWNSCALE}")
    for fname in os.listdir(img_dir):
        path = os.path.join(img_dir, fname)
        img = Image.open(path)
        img = img.crop((0, 0, img.width - 1, img.height - 1))
        img.save(path)
    open(marker, "w").close()
    print("✅ Images_4 corrigées (-1 pixel)")

# --- 3. Entraînement ---
subprocess.run(
    f'yes | ns-train splatfacto '
    f'--pipeline.datamanager.cache-images cpu '
    f'--data "{DATA_PATH}" '
    f'--max-num-iterations {ITERATIONS} '
    f'--vis tensorboard '
    f'colmap '
    f'--downscale-factor {DOWNSCALE}',
    shell=True, check=True,
)
