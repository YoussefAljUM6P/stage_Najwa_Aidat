import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

gt_dir = "YOUR PATH/renders/test/gt-rgb"
render_dir = "YOUR PATH/renders/test/rgb"

image_names = sorted(os.listdir(gt_dir))[28:39]

for name in image_names:
    gt_path = os.path.join(gt_dir, name)
    render_path = os.path.join(render_dir, name)

    if not os.path.exists(render_path):
        continue

    gt_img = Image.open(gt_path)
    render_img = Image.open(render_path)

    gt_arr = np.array(gt_img).astype(np.float32)
    render_arr = np.array(render_img).astype(np.float32)

    diff_display = np.mean(np.abs(gt_arr - render_arr) * 5, axis=2)
    mse = np.mean((gt_arr - render_arr) ** 2)
    psnr = 100 if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))

    h, w = gt_arr.shape[:2]
    ratio = w / h  # ex: 1.5 pour une image landscape

    fig = plt.figure(figsize=(ratio * 15, 5))

    # width_ratios basé sur le vrai ratio de l'image pour les 3 colonnes
    gs = gridspec.GridSpec(
        1, 4,
        width_ratios=[ratio, ratio, ratio, 0.08],
        wspace=0.05
    )

    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(gt_img)
    ax1.set_title(f"RÉALITÉ (GT)\n{name}")
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(render_img)
    ax2.set_title(f"RENDU (Gaussian Splatting)\nPSNR: {psnr:.2f} dB")
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[2])
    im_diff = ax3.imshow(diff_display, cmap='hot', aspect='equal')
    ax3.set_title("CARTE DE DIFFÉRENCE\n(Zones claires = Erreurs)")
    ax3.axis('off')

    cax = fig.add_subplot(gs[3])
    plt.colorbar(im_diff, cax=cax)

    plt.show()