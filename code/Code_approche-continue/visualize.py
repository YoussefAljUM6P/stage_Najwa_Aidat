"""
Visualisations hors entraînement :
  - grille des images d'origine du jeu de données (scène « room » de Mip-NeRF 360)
  - rendu Gaussian Splatting d'une pose donnée

Exemples :
    python visualize.py dataset --images-dir data/360_v2/room/images --n 24
    python visualize.py render --ply data/splat1.ply --pose 0.1 0.0 0.2 --yaw 90 --pitch -10 --roll 0
"""
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

import config


def show_dataset_images(images_dir, n_display=24, n_cols=4, save_path=None, show=False):
    files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    print(f"{len(files)} images trouvées dans {images_dir}")

    n_display = min(n_display, len(files))
    n_rows = (n_display + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n_display:
            ax.imshow(Image.open(os.path.join(images_dir, files[i])))
            ax.set_title(files[i], fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📊 Figure sauvegardée : {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dataset", help="Grille des images d'origine")
    d.add_argument("--images-dir", type=Path, default=config.IMAGES_DIR)
    d.add_argument("--n", type=int, default=24)

    r = sub.add_parser("render", help="Rendu gsplat d'une pose")
    r.add_argument("--ply", type=Path, default=config.PLY_PATH)
    r.add_argument("--pose", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    r.add_argument("--yaw", type=float, default=0.0)
    r.add_argument("--pitch", type=float, default=0.0)
    r.add_argument("--roll", type=float, default=0.0)

    for s in (d, r):
        s.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR)
        s.add_argument("--show", action="store_true")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.cmd == "dataset":
        show_dataset_images(args.images_dir, args.n, save_path=args.output_dir / "dataset_images.png",
                            show=args.show)
    else:
        from gs_scene import GaussianScene
        from plots import show_image
        scene = GaussianScene(args.ply)
        img = scene.render(*args.pose, yaw_deg=args.yaw, pitch_deg=args.pitch, roll_deg=args.roll)
        show_image(img, f"pos={args.pose}, yaw={args.yaw}°, pitch={args.pitch}°, roll={args.roll}°",
                   args.output_dir / "render.png", args.show)


if __name__ == "__main__":
    main()
