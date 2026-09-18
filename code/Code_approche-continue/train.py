"""
Entraînement des agents d'asservissement visuel (A3C et PPO).

Chaque expérience correspond à une section du notebook Kaggle d'origine :

  --exp yaw                    Pose -> pose, position + yaw (4 DOF)            A3C + PPO
  --exp euler                  Pose -> pose, position + 3 rotations (6 DOF)    A3C + PPO
  --exp orientation            Orientation seule, distance géodésique          PPO
  --exp orientation_decoupled  Orientation seule, distance découplée + StepLR  PPO
  --exp quat_toy               Env jouet, erreur locale + quaternion relatif   PPO (+ évaluation)
  --exp quat                   Même formulation dans la scène réelle           A3C + PPO
  --exp resnet                 Features ResNet-18 sur rendus Gaussian Splatting A3C + PPO (+ évaluation)

Exemples :
    python train.py --exp euler --algo both
    python train.py --exp resnet --algo ppo --ply data/splat1.ply --poses-bounds data/360_v2/room/poses_bounds.npy

Sorties (dans --output-dir) : checkpoints .pt, historiques .npz et figures .png.
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch

import config
import algorithms as algo
import envs
import plots
from models import ActorCritic, ActorCriticMLP

EXPERIMENTS = {
    # nom : algorithmes disponibles dans le notebook
    "yaw": ("a3c", "ppo"),
    "euler": ("a3c", "ppo"),
    "orientation": ("ppo",),
    "orientation_decoupled": ("ppo",),
    "quat_toy": ("ppo",),
    "quat": ("a3c", "ppo"),
    "resnet": ("a3c", "ppo"),
}
NEEDS_SCENE = {"resnet"}                      # rendu indispensable à l'entraînement
PREVIEW = {"yaw", "euler", "resnet"}          # aperçu de la vue cible affiché dans le notebook


# ---------------------------------------------------------------------------
# Construction des expériences
# ---------------------------------------------------------------------------
def sample_target(camera_positions, with_pitch_roll=True):
    """Cible fixe pour tout l'entraînement : une des positions de caméra réelles
    et une orientation aléatoire."""
    idx = np.random.randint(len(camera_positions))
    return {
        "idx": int(idx),
        "pos": camera_positions[idx].copy(),
        "yaw": float(np.random.uniform(0, 360)),
        "pitch": float(np.random.uniform(*config.PITCH_BOUNDS)) if with_pitch_roll else 0.0,
        "roll": float(np.random.uniform(*config.ROLL_BOUNDS)) if with_pitch_roll else 0.0,
    }


def build_experiment(exp, args, device, target=None, scene=None):
    """Renvoie (env, make_model, ctx).

    ctx contient la cible, les arguments passés à env.reset pendant
    l'entraînement et, si elle a été chargée, la scène Gaussian Splatting.
    """
    positions = None
    if exp != "quat_toy":
        from gs_scene import load_camera_positions
        positions = load_camera_positions(args.poses_bounds)
        print(f"✅ {len(positions)} positions de caméra chargées et transformées")
        if target is None:
            target = sample_target(positions, with_pitch_roll=(exp != "yaw"))

    if exp in NEEDS_SCENE and scene is None:
        from gs_scene import GaussianScene
        scene = GaussianScene(args.ply, device=device)

    # Pendant l'entraînement, seule la position cible est fixée, sauf pour
    # l'env ResNet où la pose cible complète est fixée (comme dans le notebook).
    reset_kwargs = {"target_pos": target["pos"]} if target is not None else {}

    if exp == "yaw":
        env = envs.VisualServoingEnvYaw(positions, max_steps=200, action_scale=0.08, yaw_scale=5.0)
        make_model = lambda: ActorCriticMLP(input_dim=8, action_dim=4)
    elif exp == "euler":
        env = envs.VisualServoingEnvEuler(positions, max_steps=200, action_scale=0.08, rotation_scale=10.0)
        make_model = lambda: ActorCriticMLP(input_dim=12, action_dim=6)
    elif exp == "orientation":
        env = envs.OrientationOnlyEnv(positions, max_steps=200, rotation_scale=5.0)
        make_model = lambda: ActorCriticMLP(input_dim=12, action_dim=3)
    elif exp == "orientation_decoupled":
        env = envs.OrientationOnlyEnvDecoupled(positions, max_steps=200, rotation_scale=5.0)
        make_model = lambda: ActorCriticMLP(input_dim=12, action_dim=3, max_std=0.15)
    elif exp == "quat_toy":
        env = envs.SimpleVisualServoingEnv(max_steps=100)
        make_model = lambda: ActorCritic(state_dim=7, action_dim=6)
    elif exp == "quat":
        env = envs.QuaternionVisualServoingEnv(positions, max_steps=200, pos_scale=0.08, rot_scale=0.08)
        make_model = lambda: ActorCritic(state_dim=7, action_dim=6)
    elif exp == "resnet":
        from models import ResNetFeatureExtractor
        feature_extractor = ResNetFeatureExtractor().to(device)
        env = envs.GSplatVisualServoingEnv(scene, positions, feature_extractor, device, max_steps=100)
        make_model = lambda: ActorCritic(state_dim=512, action_dim=6, hidden=(128, 64), log_std_init=-1.8)
        reset_kwargs = {"target_pos": target["pos"], "target_yaw": target["yaw"],
                        "target_pitch": target["pitch"], "target_roll": target["roll"]}
    else:
        raise ValueError(f"Expérience inconnue : {exp}")

    return env, make_model, {"target": target, "reset_kwargs": reset_kwargs, "scene": scene}


def run_algorithm(exp, name, env, model, reset_kwargs, episodes, device):
    """Hyperparamètres de chaque run, repris du notebook."""
    common = dict(reset_kwargs=reset_kwargs, device=device)

    if exp in ("yaw", "euler"):
        if name == "a3c":
            return algo.train_a3c_mlp(env, model, episodes, lr=5e-5 if exp == "yaw" else 1e-4, **common)
        return algo.train_ppo_mlp(env, model, episodes, lr=1e-4, **common)

    if exp == "orientation":
        return algo.train_ppo_mlp(env, model, episodes, lr=1e-4, **common)

    if exp == "orientation_decoupled":
        return algo.train_ppo_mlp(env, model, episodes, lr=1e-4, step_lr=(400, 0.8), log_every=50, **common)

    if exp == "quat_toy":
        return algo.train_ppo_batched(env, model, episodes, lr=1e-3, use_std_scale=False,
                                      value_clipping=False, entropy="fixed", **common)

    if exp == "quat":
        if name == "a3c":
            return algo.train_a3c_std_scale(env, model, episodes, lr=1e-3, min_lr=1e-5, **common)
        return algo.train_ppo_batched(env, model, episodes, lr=1e-3, use_std_scale=True,
                                      value_clipping=True, entropy="annealed", **common)

    if exp == "resnet":
        if name == "a3c":
            return algo.train_a3c_std_scale(env, model, episodes, lr=3e-4, min_lr=1e-5, **common)
        return algo.train_ppo_batched(env, model, episodes, lr=3e-4, use_std_scale=True,
                                      value_clipping=False, entropy="fixed", min_lr=1e-5, **common)

    raise ValueError(exp)


# ---------------------------------------------------------------------------
# Figures et évaluations de fin d'entraînement
# ---------------------------------------------------------------------------
def make_figures(exp, results, models, env, ctx, out, show, device):
    if exp in ("yaw", "euler") and {"a3c", "ppo"} <= results.keys():
        suffix = "side_by_side" if exp == "yaw" else "3rot"
        label = "" if exp == "yaw" else " (3 rotations)"
        plots.plot_reward_comparison(results["a3c"]["rewards"], results["ppo"]["rewards"],
                                     f"A3C vs PPO{label} : évolution de la reward par épisode",
                                     out / f"reward_a3c_vs_ppo_{suffix}.png", show)
    if exp == "euler" and "ppo" in results:
        plots.plot_reward_and_distance(results["ppo"]["rewards"], results["ppo"]["distances"],
                                       "PPO (3 rotations) : reward et distance au fil de l'entraînement",
                                       out / "reward_and_distance_ppo_3rot.png", show)

    if exp == "orientation":
        plots.plot_distance_curve(results["ppo"]["distances"],
                                  "Test isolé — orientation seule : évolution de l'erreur",
                                  "Erreur d'orientation (distance géodésique normalisée)",
                                  "Orientation seule (moyenne mobile)",
                                  out / "orientation_only_distance.png", show)
    if exp == "orientation_decoupled":
        plots.plot_distance_curve(results["ppo"]["distances"],
                                  "Test isolé — orientation découplée : évolution de l'erreur",
                                  "Erreur d'orientation (distance découplée)",
                                  "Orientation découplée (moyenne mobile)",
                                  out / "orientation_decoupled_distance.png", show)
        print("std final :", models["ppo"].get_std().detach().cpu().numpy())

    if exp == "quat_toy":
        from evaluate import evaluate_agent
        evaluate_agent(models["ppo"], env, num_episodes=50, device=device)

    if exp == "quat":
        for name, res in results.items():
            plots.plot_training_results(res["rewards"], res["distances"],
                                        f"Visual Servoing {name.upper()} (6 DOF)",
                                        out / f"{name}_convergence_curves.png", show,
                                        distance_label="Distance finale (position + rotation)")
        if "ppo" in models:
            print("std final (par dimension d'action) :",
                  torch.exp(models["ppo"].log_std).detach().cpu().numpy())

    if exp == "resnet":
        for name, res in results.items():
            plots.plot_training_results(res["rewards"], res["distances"],
                                        f"Visual Servoing {name.upper()} — features ResNet (6 DOF)",
                                        out / f"{name}_features_convergence_curves.png", show,
                                        thresholds=((1.0, "green", ":", "Seuil de succès (1.0)"),),
                                        distance_label="Distance cosinus (x100)")
        if "ppo" in models:
            from evaluate import evaluate_and_plot
            t = ctx["target"]
            evaluate_and_plot(env, models["ppo"], device, target_pos=t["pos"], target_yaw=t["yaw"],
                              target_pitch=t["pitch"], target_roll=t["roll"],
                              save_path=out / "resnet_ppo_trajectory.png", show=show)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", choices=EXPERIMENTS.keys(), required=True)
    p.add_argument("--algo", choices=["a3c", "ppo", "both"], default="both")
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--seed", type=int, default=None, help="Aucune graine dans le notebook (runs non reproductibles).")
    p.add_argument("--ply", type=Path, default=config.PLY_PATH)
    p.add_argument("--poses-bounds", type=Path, default=config.POSES_BOUNDS_PATH)
    p.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR)
    p.add_argument("--no-preview", action="store_true", help="Ne pas rendre l'aperçu de la vue cible.")
    p.add_argument("--show", action="store_true", help="Afficher les figures en plus de les sauvegarder.")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    available = EXPERIMENTS[args.exp]
    algos = available if args.algo == "both" else (args.algo,)
    missing = [a for a in algos if a not in available]
    if missing:
        raise SystemExit(f"L'expérience « {args.exp} » n'existe qu'avec : {', '.join(available)}")

    env, make_model, ctx = build_experiment(args.exp, args, device)
    target = ctx["target"]

    if target is not None:
        print(f"Cible (idx={target['idx']}) : {target['pos']}")
        print(f"Orientation cible : yaw={target['yaw']:.1f}°, pitch={target['pitch']:.1f}°, "
              f"roll={target['roll']:.1f}°")
        if args.exp in PREVIEW and not args.no_preview:
            scene = ctx["scene"]
            if scene is None:
                from gs_scene import GaussianScene
                scene = GaussianScene(args.ply, device=device)
            img = scene.render(*target["pos"], yaw_deg=target["yaw"],
                               pitch_deg=target["pitch"], roll_deg=target["roll"])
            plots.show_image(img, f"Cible (idx={target['idx']})\nyaw={target['yaw']:.1f}°, "
                                  f"pitch={target['pitch']:.1f}°, roll={target['roll']:.1f}°",
                             args.output_dir / f"{args.exp}_target.png", args.show)

    results, models = {}, {}
    for name in algos:
        model = make_model().to(device)
        res = run_algorithm(args.exp, name, env, model, ctx["reset_kwargs"], args.episodes, device)
        results[name], models[name] = res, model

        stem = args.output_dir / f"{args.exp}_{name}"
        np.savez(f"{stem}_history.npz", rewards=res["rewards"], distances=res["distances"], time=res["time"])
        torch.save({"experiment": args.exp, "algo": name, "model": model.state_dict(), "target": target},
                   f"{stem}.pt")
        print(f"💾 {stem}.pt et {stem}_history.npz sauvegardés")

    make_figures(args.exp, results, models, env, ctx, args.output_dir, args.show, device)


if __name__ == "__main__":
    main()
