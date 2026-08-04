"""Export du nuage de gaussiennes (.ply) et calcul du PSNR sur le split test.

    python eval.py
"""

import glob
import os
import subprocess

import numpy as np
from PIL import Image

OUTPUT_DIR = "outputs"      # sorties de ns-train
EXPORT_DIR = "exports/room"  # destination du .ply
RENDER_DIR = "renders"      # rendus du split test

os.environ["MAX_JOBS"] = "2"


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))


def load_data(path):
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32)


# Dernier run entraîné (le dossier créé par ns-train est un timestamp)
configs = sorted(glob.glob(f"{OUTPUT_DIR}/*/splatfacto/*/config.yml"))
if not configs:
    raise SystemExit(f"Aucun run dans {OUTPUT_DIR}/ — lance train.py d'abord")
config = configs[-1]
print(f"Run évalué : {config}\n")

# --- Export du nuage de gaussiennes ---
subprocess.run(
    f'ns-export gaussian-splat --load-config "{config}" --output-dir "{EXPORT_DIR}"',
    shell=True, check=True,
)
print(f"✅ .ply exporté dans {EXPORT_DIR}/\n")

# --- Rendu du split test (nécessaire au PSNR) ---
if not os.path.isdir(f"{RENDER_DIR}/test/gt-rgb"):
    subprocess.run(
        f'ns-render dataset --load-config "{config}" --output-path "{RENDER_DIR}"',
        shell=True, check=True,
    )

# --- PSNR ---
gt_dir = f"{RENDER_DIR}/test/gt-rgb"
render_dir = f"{RENDER_DIR}/test/rgb"

psnrs = []
for name in sorted(os.listdir(gt_dir)):
    gt = load_data(os.path.join(gt_dir, name))
    render = load_data(os.path.join(render_dir, name))
    psnrs.append(calculate_psnr(gt, render))

print("\n--- RÉSULTATS ---")
print(f"Images       : {len(psnrs)}")
print(f"Moyenne PSNR : {np.mean(psnrs):.2f} dB")
