"""
Environnements Gymnasium d'asservissement visuel, dans l'ordre des expériences.

  1. VisualServoingEnvYaw          état = poses (position + yaw), 4 DOF
  2. VisualServoingEnvEuler        état = poses (position + yaw/pitch/roll), 6 DOF
  3. OrientationOnlyEnv            test isolé : position figée, 3 rotations, distance géodésique
  4. OrientationOnlyEnvDecoupled   idem, distance angulaire découplée (yaw, pitch, roll séparément)
  5. SimpleVisualServoingEnv       environnement jouet : erreur locale + quaternion relatif
  6. QuaternionVisualServoingEnv   idem 5 mais dans l'espace de travail réel de la scène
  7. GSplatVisualServoingEnv       état = différence de features ResNet entre rendu courant et cible

Dans le notebook, les environnements 1 et 2 s'appelaient tous deux
« VisualServoingEnvMLP » (la seconde définition remplaçait la première).

Toutes les classes renvoient dans `info["mse"]` la distance à la cible
utilisée pour la récompense (le nom « mse » est hérité du notebook ; ce
n'est pas une erreur quadratique moyenne).
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial.transform import Rotation as Rsc

import config
from gs_scene import euler_to_matrix


# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------
def angular_diff(a, b):
    """Écart angulaire absolu entre deux angles en degrés, ramené dans [0, 180]."""
    return abs((a - b + 180) % 360 - 180)


def geodesic_distance_deg(R1, R2):
    """Angle (degrés) de la rotation relative R1^T R2 (distance géodésique sur SO(3))."""
    trace = np.clip(np.trace(R1.T @ R2), -1.0, 3.0)
    return np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))


def wrap180(angle):
    """Ramène un angle en degrés dans [-180, 180)."""
    return ((angle + 180) % 360) - 180


# ===========================================================================
# 1. Pose -> pose, position + yaw (4 DOF)
# ===========================================================================
class VisualServoingEnvYaw(gym.Env):
    """Action (4) : [dx, dy, dz, dyaw] dans [-1, 1].
    État (8)     : pos courante (3) + pos cible (3) + yaw courant / 180 + yaw cible / 180.
    Distance     : ||p - p*|| / diag(espace de travail) + |yaw - yaw*| / 180.
    """

    def __init__(self, camera_positions, max_steps=200, action_scale=0.08, yaw_scale=5.0):
        super().__init__()
        self.camera_positions = camera_positions
        self.max_steps = max_steps
        self.action_scale = action_scale
        self.yaw_scale = yaw_scale

        self.current_pos = self.target_pos = None
        self.current_yaw = self.target_yaw = None
        self.step_count = 0
        self.prev_distance = None

        self.pos_min = camera_positions.min(axis=0)
        self.pos_max = camera_positions.max(axis=0)
        self.max_pos_dist = np.linalg.norm(self.pos_max - self.pos_min)

        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

    def _compute_distance(self):
        pos_dist_norm = np.linalg.norm(self.current_pos - self.target_pos) / self.max_pos_dist
        yaw_dist_norm = angular_diff(self.current_yaw, self.target_yaw) / 180.0
        return pos_dist_norm + yaw_dist_norm

    def _get_state(self):
        return np.concatenate([
            self.current_pos,
            self.target_pos,
            [wrap180(self.current_yaw) / 180.0],
            [wrap180(self.target_yaw) / 180.0],
        ]).astype(np.float32)

    def reset(self, seed=None, options=None, target_pos=None, target_yaw=None):
        super().reset(seed=seed)
        n = len(self.camera_positions)
        self.current_pos = self.camera_positions[np.random.randint(n)].copy()
        if target_pos is not None:
            self.target_pos = np.array(target_pos, dtype=np.float32)
        else:
            self.target_pos = self.camera_positions[np.random.randint(n)].copy()

        self.current_yaw = np.random.uniform(0, 360)
        self.target_yaw = target_yaw if target_yaw is not None else np.random.uniform(0, 360)

        self.step_count = 0
        self.prev_distance = self._compute_distance()
        return self._get_state(), {}

    def step(self, action):
        self.step_count += 1
        self.current_pos = np.clip(self.current_pos + action[:3] * self.action_scale,
                                   self.pos_min, self.pos_max)
        self.current_yaw = (self.current_yaw + action[3] * self.yaw_scale) % 360

        current_distance = self._compute_distance()
        reward = (self.prev_distance - current_distance) * 100.0   # récompense de progrès

        terminated = False
        if current_distance < 0.2:
            reward += 10.0
            terminated = True

        truncated = self.step_count >= self.max_steps
        self.prev_distance = current_distance
        return self._get_state(), reward, terminated, truncated, {"mse": current_distance}


# ===========================================================================
# 2. Pose -> pose, position + yaw/pitch/roll (6 DOF)
# ===========================================================================
class VisualServoingEnvEuler(gym.Env):
    """Action (6) : [dx, dy, dz, dyaw, dpitch, droll] dans [-1, 1].
    État (12)    : pos courante (3) + pos cible (3) + Euler courants / 180 (3) + Euler cibles / 180 (3).
    Distance     : ||p - p*|| / diag + distance géodésique(R, R*) / 180.
    """

    def __init__(self, camera_positions, max_steps=200, action_scale=0.08, rotation_scale=5.0,
                 pitch_bounds=config.PITCH_BOUNDS, roll_bounds=config.ROLL_BOUNDS):
        super().__init__()
        self.camera_positions = camera_positions
        self.max_steps = max_steps
        self.action_scale = action_scale
        self.rotation_scale = rotation_scale
        self.pitch_bounds = pitch_bounds
        self.roll_bounds = roll_bounds

        self.current_pos = self.target_pos = None
        self.current_yaw = self.current_pitch = self.current_roll = None
        self.target_yaw = self.target_pitch = self.target_roll = None
        self.step_count = 0
        self.prev_distance = None

        self.pos_min = camera_positions.min(axis=0)
        self.pos_max = camera_positions.max(axis=0)
        self.max_pos_dist = np.linalg.norm(self.pos_max - self.pos_min)

        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

    def _orientation_distance(self):
        R_current = euler_to_matrix(self.current_yaw, self.current_pitch, self.current_roll)
        R_target = euler_to_matrix(self.target_yaw, self.target_pitch, self.target_roll)
        return geodesic_distance_deg(R_current, R_target) / 180.0

    def _compute_distance(self):
        pos_dist_norm = np.linalg.norm(self.current_pos - self.target_pos) / self.max_pos_dist
        return pos_dist_norm + self._orientation_distance()

    def _get_state(self):
        return np.concatenate([
            self.current_pos,
            self.target_pos,
            np.array([wrap180(self.current_yaw), self.current_pitch, self.current_roll]) / 180.0,
            np.array([wrap180(self.target_yaw), self.target_pitch, self.target_roll]) / 180.0,
        ]).astype(np.float32)

    def _sample_orientations(self, target_yaw, target_pitch, target_roll):
        self.current_yaw = np.random.uniform(0, 360)
        self.current_pitch = np.random.uniform(*self.pitch_bounds)
        self.current_roll = np.random.uniform(*self.roll_bounds)
        self.target_yaw = target_yaw if target_yaw is not None else np.random.uniform(0, 360)
        self.target_pitch = target_pitch if target_pitch is not None else np.random.uniform(*self.pitch_bounds)
        self.target_roll = target_roll if target_roll is not None else np.random.uniform(*self.roll_bounds)

    def reset(self, seed=None, options=None, target_pos=None,
              target_yaw=None, target_pitch=None, target_roll=None):
        super().reset(seed=seed)
        n = len(self.camera_positions)
        self.current_pos = self.camera_positions[np.random.randint(n)].copy()
        if target_pos is not None:
            self.target_pos = np.array(target_pos, dtype=np.float32)
        else:
            self.target_pos = self.camera_positions[np.random.randint(n)].copy()

        self._sample_orientations(target_yaw, target_pitch, target_roll)

        self.step_count = 0
        self.prev_distance = self._compute_distance()
        return self._get_state(), {}

    def step(self, action):
        self.step_count += 1
        self.current_pos = np.clip(self.current_pos + action[:3] * self.action_scale,
                                   self.pos_min, self.pos_max)
        self.current_yaw = (self.current_yaw + action[3] * self.rotation_scale) % 360
        self.current_pitch = np.clip(self.current_pitch + action[4] * self.rotation_scale, *self.pitch_bounds)
        self.current_roll = np.clip(self.current_roll + action[5] * self.rotation_scale, *self.roll_bounds)

        current_distance = self._compute_distance()
        reward = (self.prev_distance - current_distance) * 100.0

        terminated = False
        if current_distance < 0.2:
            reward += 10.0
            terminated = True

        truncated = self.step_count >= self.max_steps
        self.prev_distance = current_distance
        return self._get_state(), reward, terminated, truncated, {"mse": current_distance}


# ===========================================================================
# 3. Test isolé : orientation seule, distance géodésique
# ===========================================================================
class OrientationOnlyEnv(VisualServoingEnvEuler):
    """La position courante est fixée sur la position cible : seules les
    3 rotations sont apprises.
    Action (3) : [dyaw, dpitch, droll].  État (12) : identique à l'env 6 DOF.
    Distance   : distance géodésique / 180.  Succès si distance < 0.1.
    """

    success_threshold = 0.1

    def __init__(self, camera_positions, max_steps=200, rotation_scale=5.0,
                 pitch_bounds=config.PITCH_BOUNDS, roll_bounds=config.ROLL_BOUNDS):
        super().__init__(camera_positions, max_steps=max_steps, action_scale=0.0,
                         rotation_scale=rotation_scale,
                         pitch_bounds=pitch_bounds, roll_bounds=roll_bounds)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

    def _compute_distance(self):
        return self._orientation_distance()

    def reset(self, seed=None, options=None, target_pos=None,
              target_yaw=None, target_pitch=None, target_roll=None):
        gym.Env.reset(self, seed=seed)
        if target_pos is not None:
            self.target_pos = np.array(target_pos, dtype=np.float32)
        else:
            self.target_pos = self.camera_positions[np.random.randint(len(self.camera_positions))].copy()
        self.current_pos = self.target_pos.copy()   # position figée sur la cible

        self._sample_orientations(target_yaw, target_pitch, target_roll)

        self.step_count = 0
        self.prev_distance = self._compute_distance()
        return self._get_state(), {}

    def step(self, action):
        self.step_count += 1
        self.current_yaw = (self.current_yaw + action[0] * self.rotation_scale) % 360
        self.current_pitch = np.clip(self.current_pitch + action[1] * self.rotation_scale, *self.pitch_bounds)
        self.current_roll = np.clip(self.current_roll + action[2] * self.rotation_scale, *self.roll_bounds)

        current_distance = self._compute_distance()
        reward = (self.prev_distance - current_distance) * 100.0

        terminated = False
        if current_distance < self.success_threshold:
            reward += 10.0
            terminated = True

        truncated = self.step_count >= self.max_steps
        self.prev_distance = current_distance
        return self._get_state(), reward, terminated, truncated, {"mse": current_distance}


# ===========================================================================
# 4. Test isolé : orientation seule, distance découplée
# ===========================================================================
class OrientationOnlyEnvDecoupled(OrientationOnlyEnv):
    """Identique à OrientationOnlyEnv, mais la distance est la moyenne des
    écarts angulaires sur yaw, pitch et roll pris séparément (chacun / 180)."""

    def _compute_distance(self):
        d_yaw = angular_diff(self.current_yaw, self.target_yaw) / 180.0
        d_pitch = angular_diff(self.current_pitch, self.target_pitch) / 180.0
        d_roll = angular_diff(self.current_roll, self.target_roll) / 180.0
        return (d_yaw + d_pitch + d_roll) / 3.0


# ===========================================================================
# 5. Environnement jouet : erreur exprimée dans le repère caméra + quaternion
# ===========================================================================
class SimpleVisualServoingEnv(gym.Env):
    """Cible fixe à l'origine (position nulle, rotation identité).
    Départ : position uniforme dans [-0.5, 0.5]^3, rotation aléatoire (rotvec dans [-0.3, 0.3]^3).
    Action (6) : translation (3) dans le repère caméra + rotvec (3), composée à droite.
    État (7)   : erreur de position dans le repère caméra (3) + quaternion relatif (4).
    Succès     : erreur position < 0.02 et erreur rotation < 0.05 rad.
    """

    def __init__(self, max_steps=100, pos_scale=0.05, rot_scale=0.05):
        super().__init__()
        self.max_steps = max_steps
        self.pos_scale = pos_scale
        self.rot_scale = rot_scale

        self.current_pos = self.target_pos = None
        self.current_rot = self.target_rot = None
        self.step_count = 0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

    def _get_distances(self):
        pos_dist = np.linalg.norm(self.target_pos - self.current_pos)
        rot_dist = (self.current_rot.inv() * self.target_rot).magnitude()   # angle en radians
        return pos_dist, rot_dist

    def _get_state(self):
        pos_err_local = self.current_rot.inv().apply(self.target_pos - self.current_pos)
        quat_rel = (self.current_rot.inv() * self.target_rot).as_quat()
        return np.concatenate([pos_err_local, quat_rel]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.target_pos = np.zeros(3, dtype=np.float32)
        self.target_rot = Rsc.identity()
        self.current_pos = np.random.uniform(-0.5, 0.5, size=(3,)).astype(np.float32)
        self.current_rot = Rsc.from_rotvec(np.random.uniform(-0.3, 0.3, size=(3,)))

        self.step_count = 0
        self.prev_pos_d, self.prev_rot_d = self._get_distances()
        return self._get_state(), {}

    def _apply_action(self, action, pos_min, pos_max):
        delta_pos_world = self.current_rot.apply(action[:3] * self.pos_scale)
        self.current_pos = np.clip(self.current_pos + delta_pos_world, pos_min, pos_max)
        self.current_rot = self.current_rot * Rsc.from_rotvec(action[3:] * self.rot_scale)

    def _reward_and_termination(self):
        pos_d, rot_d = self._get_distances()
        reward = (self.prev_pos_d - pos_d) * 10.0 + (self.prev_rot_d - rot_d) * 5.0 - 0.01

        terminated = False
        if pos_d < 0.02 and rot_d < 0.05:   # < 2 cm et < ~2.8°
            reward += 20.0
            terminated = True

        truncated = self.step_count >= self.max_steps
        self.prev_pos_d, self.prev_rot_d = pos_d, rot_d
        return reward, terminated, truncated, {"mse": pos_d + rot_d}

    def step(self, action):
        self.step_count += 1
        self._apply_action(action, -2.0, 2.0)
        reward, terminated, truncated, info = self._reward_and_termination()
        return self._get_state(), reward, terminated, truncated, info


# Séquence d'Euler utilisée par scipy dans l'env 6, conservée telle quelle depuis le notebook.
# ATTENTION : en minuscules, scipy interprète "yxz" en rotations EXTRINSÈQUES, ce qui donne
# R_z @ R_x @ R_y et NON R_y @ R_x @ R_z comme euler_to_matrix. La séquence cohérente avec
# euler_to_matrix serait "YXZ" (intrinsèque). Sans effet sur l'apprentissage (l'état est un
# quaternion relatif), mais render_current / render_target ne rendent pas exactement
# l'orientation de l'agent. Voir le README.
EULER_SEQ = "yxz"


# ===========================================================================
# 6. Même formulation, dans l'espace de travail réel de la scène
# ===========================================================================
class QuaternionVisualServoingEnv(SimpleVisualServoingEnv):
    """Positions de départ et cibles tirées parmi les caméras réelles de la scène,
    orientations tirées dans les bornes d'Euler du projet, déplacements bornés
    par l'espace de travail des caméras."""

    def __init__(self, camera_positions, max_steps=200, pos_scale=0.08, rot_scale=0.08,
                 pitch_bounds=config.PITCH_BOUNDS, roll_bounds=config.ROLL_BOUNDS):
        super().__init__(max_steps=max_steps, pos_scale=pos_scale, rot_scale=rot_scale)
        self.camera_positions = camera_positions
        self.pitch_bounds = pitch_bounds
        self.roll_bounds = roll_bounds
        self.pos_min = camera_positions.min(axis=0)
        self.pos_max = camera_positions.max(axis=0)

    def _sample_random_rotation(self):
        yaw = np.random.uniform(0, 360)
        pitch = np.random.uniform(*self.pitch_bounds)
        roll = np.random.uniform(*self.roll_bounds)
        return Rsc.from_euler(EULER_SEQ, [yaw, pitch, roll], degrees=True)

    def reset(self, seed=None, options=None, target_pos=None,
              target_yaw=None, target_pitch=None, target_roll=None):
        gym.Env.reset(self, seed=seed)
        n = len(self.camera_positions)
        self.current_pos = self.camera_positions[np.random.randint(n)].copy()
        if target_pos is not None:
            self.target_pos = np.array(target_pos, dtype=np.float32)
        else:
            self.target_pos = self.camera_positions[np.random.randint(n)].copy()

        self.current_rot = self._sample_random_rotation()
        if target_yaw is not None:
            self.target_rot = Rsc.from_euler(EULER_SEQ, [target_yaw, target_pitch, target_roll], degrees=True)
        else:
            self.target_rot = self._sample_random_rotation()

        self.step_count = 0
        self.prev_pos_d, self.prev_rot_d = self._get_distances()
        return self._get_state(), {}

    def step(self, action):
        self.step_count += 1
        self._apply_action(action, self.pos_min, self.pos_max)
        reward, terminated, truncated, info = self._reward_and_termination()
        return self._get_state(), reward, terminated, truncated, info

    # Rendus pour visualisation uniquement (non utilisés pendant l'entraînement)
    def render_current(self, render_fn):
        yaw, pitch, roll = self.current_rot.as_euler(EULER_SEQ, degrees=True)
        return render_fn(*self.current_pos, yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll)

    def render_target(self, render_fn):
        yaw, pitch, roll = self.target_rot.as_euler(EULER_SEQ, degrees=True)
        return render_fn(*self.target_pos, yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll)


# ===========================================================================
# 7. Asservissement dans l'espace latent ResNet, rendus Gaussian Splatting
# ===========================================================================
class GSplatVisualServoingEnv(gym.Env):
    """État (512)  : z* - z, différence entre les features ResNet-18 (normalisées L2)
                     de la vue cible et de la vue courante rendue par gsplat.
    Action (6)     : [dx, dy, dz, dyaw, dpitch, droll].
    Distance       : distance cosinus (1 - <z*, z>) x 100.
    Départ         : pose cible + bruit (±0.3 en position, ±15° par angle).
    Succès         : distance < 1.0 (bonus +100) ; bonus partiel si distance < 3.0 en fin d'épisode.
    """

    def __init__(self, render_fn, camera_positions, feature_extractor, device, max_steps=100,
                 pitch_bounds=config.PITCH_BOUNDS, roll_bounds=config.ROLL_BOUNDS):
        super().__init__()
        self.render_fn = render_fn
        self.camera_positions = camera_positions
        self.feature_extractor = feature_extractor
        self.device = device
        self.max_steps = max_steps
        self.pitch_bounds = pitch_bounds
        self.roll_bounds = roll_bounds

        self.pos_min = camera_positions.min(axis=0)
        self.pos_max = camera_positions.max(axis=0)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(512,), dtype=np.float32)

        self.pos_scale = 0.05
        self.rot_scale = 5.0   # degrés, appliqué aux 3 rotations

        self.current_pos = None
        self.current_yaw = self.current_pitch = self.current_roll = 0.0
        self.target_pos = None
        self.target_yaw = self.target_pitch = self.target_roll = 0.0
        self.z_target = None
        self.prev_feat_dist = None
        self.step_count = 0

    def _get_feature_vector(self, img_np):
        import torch
        import torch.nn.functional as F
        from models import resnet_transform

        img_tensor = resnet_transform(img_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.feature_extractor(img_tensor)
        feat = F.normalize(feat, p=2, dim=1)
        return feat.squeeze(0).cpu().numpy()

    def _get_state(self):
        img_curr = self.render_fn(*self.current_pos, yaw_deg=self.current_yaw,
                                  pitch_deg=self.current_pitch, roll_deg=self.current_roll)
        z_curr = self._get_feature_vector(img_curr)
        cosine_dist = (1.0 - np.dot(self.z_target, z_curr)) * 100.0
        return (self.z_target - z_curr).astype(np.float32), cosine_dist

    def reset(self, seed=None, options=None, target_pos=None,
              target_yaw=None, target_pitch=None, target_roll=None):
        super().reset(seed=seed)
        if target_pos is not None:
            self.target_pos = np.array(target_pos, dtype=np.float32)
        else:
            self.target_pos = self.camera_positions[np.random.choice(len(self.camera_positions))].copy()

        self.target_yaw = target_yaw if target_yaw is not None else np.random.uniform(0, 360)
        self.target_pitch = target_pitch if target_pitch is not None else np.random.uniform(*self.pitch_bounds)
        self.target_roll = target_roll if target_roll is not None else np.random.uniform(*self.roll_bounds)

        img_target = self.render_fn(*self.target_pos, yaw_deg=self.target_yaw,
                                    pitch_deg=self.target_pitch, roll_deg=self.target_roll)
        self.z_target = self._get_feature_vector(img_target)

        # Pose de départ = cible + bruit
        self.current_pos = np.clip(self.target_pos + np.random.uniform(-0.3, 0.3, size=(3,)),
                                   self.pos_min, self.pos_max)
        self.current_yaw = (self.target_yaw + np.random.uniform(-15.0, 15.0)) % 360
        self.current_pitch = np.clip(self.target_pitch + np.random.uniform(-15.0, 15.0), *self.pitch_bounds)
        self.current_roll = np.clip(self.target_roll + np.random.uniform(-15.0, 15.0), *self.roll_bounds)

        self.step_count = 0
        state, feat_dist = self._get_state()
        self.prev_feat_dist = feat_dist
        return state, {}

    def step(self, action):
        self.step_count += 1
        self.current_pos = np.clip(self.current_pos + action[:3] * self.pos_scale, self.pos_min, self.pos_max)
        self.current_yaw = (self.current_yaw + action[3] * self.rot_scale) % 360
        self.current_pitch = np.clip(self.current_pitch + action[4] * self.rot_scale, *self.pitch_bounds)
        self.current_roll = np.clip(self.current_roll + action[5] * self.rot_scale, *self.roll_bounds)

        state, feat_dist = self._get_state()
        reward = (self.prev_feat_dist - feat_dist) * 20.0

        terminated = False
        if feat_dist < 1.0:
            reward += 100.0
            terminated = True

        truncated = self.step_count >= self.max_steps
        if truncated and not terminated and feat_dist < 3.0:
            reward += 10.0 * (3.0 - feat_dist)   # bonus partiel si proche de la cible

        self.prev_feat_dist = feat_dist
        return state, reward, terminated, truncated, {"mse": feat_dist}
