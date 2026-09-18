"""
Chargement de la scène Gaussian Splatting (.ply) et rendu de vues avec gsplat.

Contient aussi les conversions de poses :
    angles d'Euler -> matrice de rotation (convention du projet)
    pose COLMAP (R, t)  -> c2w dans le repère du splat (nerfstudio)
    c2w splat           -> viewmat attendue par gsplat.rasterization
"""
from pathlib import Path

import numpy as np
import torch

import config


# ---------------------------------------------------------------------------
# Rotations
# ---------------------------------------------------------------------------
def euler_to_matrix(yaw_deg, pitch_deg=0.0, roll_deg=0.0):
    """R = R_yaw (axe Y) @ R_pitch (axe X) @ R_roll (axe Z), angles en degrés.

    Avec pitch = roll = 0, on retrouve la rotation « yaw seul » utilisée
    dans la première version de l'environnement (4 DOF).
    """
    yaw, pitch, roll = np.radians([yaw_deg, pitch_deg, roll_deg])

    R_yaw = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)],
    ])
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)],
    ])
    R_roll = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1],
    ])
    return R_yaw @ R_pitch @ R_roll


def build_pose_from_position_euler(x, y, z, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
    """Construit une pose (R, t) au format COLMAP à partir d'une position et d'angles d'Euler."""
    R = euler_to_matrix(yaw_deg, pitch_deg, roll_deg)
    t = np.array([x, y, z])
    return R, t


# ---------------------------------------------------------------------------
# Changements de repère
# ---------------------------------------------------------------------------
def colmap_pose_to_splat_c2w(R_colmap, t_colmap,
                             transform=config.NERFSTUDIO_TRANSFORM,
                             scale=config.SCALE_FACTOR):
    """Pose COLMAP (w2c) -> matrice c2w 4x4 dans le repère du splat nerfstudio."""
    # w2c -> c2w
    c2w = np.eye(4)
    c2w[:3, :3] = R_colmap.T
    c2w[:3, 3] = -R_colmap.T @ t_colmap
    # Convention OpenCV -> OpenGL (axes Y et Z inversés)
    c2w[0:3, 1:3] *= -1

    # Transformation + mise à l'échelle appliquées par nerfstudio
    R_trans = transform[:, :3]
    t_trans = transform[:, 3]
    c2w_splat = np.eye(4)
    c2w_splat[:3, :3] = R_trans @ c2w[:3, :3]
    c2w_splat[:3, 3] = (R_trans @ c2w[:3, 3] + t_trans) * scale
    return c2w_splat


def splat_c2w_to_gsplat_viewmat(c2w_splat, device="cuda"):
    """c2w (convention OpenGL) -> viewmat w2c (convention OpenCV) pour gsplat."""
    c2w_cv = c2w_splat.copy()
    c2w_cv[0:3, 1:3] *= -1
    c2w_t = torch.tensor(c2w_cv, dtype=torch.float32, device=device)
    return torch.inverse(c2w_t)


def transform_colmap_to_nerfstudio(pos_colmap,
                                   transform=config.NERFSTUDIO_TRANSFORM,
                                   scale=config.SCALE_FACTOR):
    """Position 3D COLMAP -> position dans le repère du splat."""
    pos_homo = np.array([*pos_colmap, 1.0])
    return (transform @ pos_homo) * scale


def load_camera_positions(poses_bounds_path):
    """Charge les positions des caméras d'origine (format LLFF poses_bounds.npy)
    et les exprime dans le repère du splat.

    Ces positions définissent l'espace de travail des environnements
    (tirage des positions initiales / cibles et bornes de déplacement).
    """
    poses_bounds = np.load(poses_bounds_path)
    colmap_positions = poses_bounds[:, [3, 8, 13]]  # colonne translation de la matrice 3x5
    return np.array([transform_colmap_to_nerfstudio(p) for p in colmap_positions])


# ---------------------------------------------------------------------------
# Scène Gaussian Splatting
# ---------------------------------------------------------------------------
class GaussianScene:
    """Charge un fichier .ply (export splatfacto) et rend des vues avec gsplat."""

    def __init__(self, ply_path, device=None,
                 width=config.WIDTH, height=config.HEIGHT, fx=config.FX, fy=config.FY):
        from plyfile import PlyData  # import local : seulement nécessaire pour le rendu

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.W, self.H = width, height

        vertex = PlyData.read(str(ply_path))["vertex"]

        means = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1)
        scales = np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=-1)
        opacities = vertex["opacity"]
        quats = np.stack([vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]], axis=-1)
        features_dc = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=-1)[:, None, :]

        sh_names = sorted([f for f in vertex.data.dtype.names if f.startswith("f_rest_")],
                          key=lambda s: int(s.split("_")[-1]))
        features_rest = np.stack([vertex[f] for f in sh_names], axis=-1).reshape(len(means), 15, 3)
        shs = np.concatenate([features_dc, features_rest], axis=1)  # harmoniques sphériques degré 3

        d = self.device
        self.means = torch.tensor(means, dtype=torch.float32, device=d)
        self.scales = torch.tensor(np.exp(scales), dtype=torch.float32, device=d)            # log-échelle -> échelle
        self.quats = torch.tensor(quats, dtype=torch.float32, device=d)
        self.opacities = torch.tensor(1.0 / (1.0 + np.exp(-opacities)), dtype=torch.float32, device=d)  # sigmoïde
        self.shs = torch.tensor(shs, dtype=torch.float32, device=d)
        self.K = torch.tensor([[fx, 0.0, width / 2],
                               [0.0, fy, height / 2],
                               [0.0, 0.0, 1.0]], dtype=torch.float32, device=d)

        print(f"✅ PLY chargé : {len(means)} Gaussiennes sur {d}")

    def render(self, x, y, z, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
        """Rend la vue depuis la position (x, y, z) et l'orientation donnée.
        Retourne une image uint8 de forme (H, W, 3)."""
        import gsplat

        R, t = build_pose_from_position_euler(x, y, z, yaw_deg, pitch_deg, roll_deg)
        c2w_splat = colmap_pose_to_splat_c2w(R, t)
        viewmat = splat_c2w_to_gsplat_viewmat(c2w_splat, device=self.device).unsqueeze(0)

        with torch.no_grad():
            render_colors, _, _ = gsplat.rasterization(
                means=self.means, quats=self.quats, scales=self.scales, opacities=self.opacities,
                colors=self.shs, viewmats=viewmat, Ks=self.K.unsqueeze(0),
                width=self.W, height=self.H, sh_degree=3,
            )

        img = render_colors[0].detach().cpu().numpy()
        if img.shape[0] == 3 and img.shape[1] == self.H and img.shape[2] == self.W:
            img = np.transpose(img, (1, 2, 0))
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)

    # Permet d'utiliser l'objet directement comme render_fn
    __call__ = render
