"""
Évaluation des politiques entraînées.

- evaluate_agent    : statistiques sur N épisodes (envs 5 et 6, formulation quaternion)
- evaluate_and_plot : un épisode avec l'action moyenne, trajectoire 3D + distance visuelle (env 7)

Utilisation en ligne de commande, sur un checkpoint produit par train.py :
    python evaluate.py --checkpoint outputs/quat_toy_ppo.pt
    python evaluate.py --checkpoint outputs/resnet_ppo.pt --ply data/splat1.ply
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt


def evaluate_agent(model, env, num_episodes=50, device="cpu", deterministic=False):
    
    model.eval()
    model.to(device)
    pos_errors, rot_errors, successes, lengths = [], [], [], []
    mode = "déterministe (moyenne)" if deterministic else "stochastique (échantillonnée)"
    print(f"🟦 Évaluation {mode} sur {num_episodes} épisodes...")

    for _ in range(num_episodes):
        state = torch.from_numpy(env.reset()[0]).unsqueeze(0).to(device)
        done, step, success = False, 0, False
        while not done:
            step += 1
            with torch.no_grad():
                if deterministic:
                    action = torch.clamp(model(state)[0], -1.0, 1.0)
                else:
                    action = model.get_action(state)[0]
            next_state_np, _, terminated, truncated, _ = env.step(action[0].cpu().numpy())
            done = terminated or truncated
            success = success or terminated
            state = torch.from_numpy(next_state_np).unsqueeze(0).to(device)

        pos_err, rot_err = env._get_distances()
        pos_errors.append(pos_err)
        rot_errors.append(rot_err)
        successes.append(int(success))
        lengths.append(step)

    results = {
        "mean_pos": float(np.mean(pos_errors)),
        "std_pos": float(np.std(pos_errors)),
        "mean_rot": float(np.mean(rot_errors)),
        "success_rate": 100.0 * np.mean(successes),
        "avg_steps": float(np.mean(lengths)),
    }
    print("\n" + "=" * 45)
    print("📊 RÉSULTATS DE L'ÉVALUATION")
    print("=" * 45)
    print(f"🎯 Erreur de position finale (moyenne) : {results['mean_pos']:.4f} (± {results['std_pos']:.4f})")
    print(f"🔄 Erreur de rotation finale (moyenne) : {results['mean_rot']:.4f} rad "
          f"({np.degrees(results['mean_rot']):.2f}°)")
    print(f"🏁 Taux de succès                      : {results['success_rate']:.1f} %")
    print(f"⏱️  Nombre moyen de pas par épisode     : {results['avg_steps']:.1f} / {env.max_steps}")
    print("=" * 45)
    return results


def evaluate_and_plot(env, model, device, target_pos=None, target_yaw=None, target_pitch=None,
                      target_roll=None, save_path=None, show=False):
    
    model.eval()
    state, _ = env.reset(target_pos=target_pos, target_yaw=target_yaw,
                         target_pitch=target_pitch, target_roll=target_roll)
    positions = [env.current_pos.copy()]
    distances = []
    target = env.target_pos.copy()
    done = False

    print("🎬 Début de l'évaluation...")
    with torch.no_grad():
        while not done:
            action_mean, _ = model(torch.FloatTensor(state).unsqueeze(0).to(device))
            state, _, terminated, truncated, info = env.step(action_mean.squeeze(0).cpu().numpy())
            done = terminated or truncated
            positions.append(env.current_pos.copy())
            distances.append(info["mse"])

    positions, distances = np.array(positions), np.array(distances)
    print(f"✅ Évaluation terminée en {len(distances)} étapes.")
    print(f"   Distance finale de features : {distances[-1]:.4f}")

    fig = plt.figure(figsize=(15, 6))
    ax_3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_3d.plot(positions[:, 0], positions[:, 1], positions[:, 2],
               color="blue", marker="o", markersize=3, label="Trajectoire caméra")
    ax_3d.scatter(*positions[0], color="green", s=100, label="Départ")
    ax_3d.scatter(*target, color="red", s=100, label="Cible")
    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    ax_3d.set_zlabel("Z")
    ax_3d.set_title("Trajectoire 3D de l'asservissement")
    ax_3d.legend()

    ax_d = fig.add_subplot(1, 2, 2)
    ax_d.plot(range(1, len(distances) + 1), distances, color="purple", linewidth=2)
    ax_d.axhline(y=1.0, color="r", linestyle="--", label="Seuil de succès (< 1.0)")
    ax_d.set_xlabel("Étapes")
    ax_d.set_ylabel("Distance cosinus (x100)")
    ax_d.set_title("Évolution de la distance visuelle")
    ax_d.grid(True)
    ax_d.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📊 Figure sauvegardée : {save_path}")
    if show:
        plt.show()
    plt.close(fig)
    return positions, distances


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    import config
    from train import build_experiment

    parser = argparse.ArgumentParser(description="Évaluation d'un checkpoint produit par train.py")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ply", type=Path, default=config.PLY_PATH)
    parser.add_argument("--poses-bounds", type=Path, default=config.POSES_BOUNDS_PATH)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    exp = ckpt["experiment"]
    env, make_model, _ = build_experiment(exp, args, device, target=ckpt.get("target"))
    model = make_model().to(device)
    model.load_state_dict(ckpt["model"])

    if exp in ("quat_toy", "quat"):
        evaluate_agent(model, env, num_episodes=args.episodes, device=device,
                       deterministic=args.deterministic)
    elif exp == "resnet":
        t = ckpt["target"]
        evaluate_and_plot(env, model, device, target_pos=t["pos"], target_yaw=t["yaw"],
                          target_pitch=t["pitch"], target_roll=t["roll"],
                          save_path=args.output_dir / f"{args.checkpoint.stem}_eval.png", show=args.show)
    else:
        raise SystemExit(f"Pas d'évaluation dédiée pour l'expérience « {exp} » dans le notebook.")


if __name__ == "__main__":
    main()
