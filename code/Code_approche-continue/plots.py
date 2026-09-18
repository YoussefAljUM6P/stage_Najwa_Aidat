"""Figures de suivi d'entraînement (sauvegardées en PNG dans le dossier de sortie)."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d


def _finish(fig, save_path, show):
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📊 Figure sauvegardée : {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_reward_comparison(rewards_a3c, rewards_ppo, title, save_path=None, show=False, window=20):
    """A3C vs PPO côte à côte : reward cumulée par épisode (brute + moyenne mobile)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, rewards, name, color in [(ax1, rewards_a3c, "A3C", "#FF6B6B"),
                                     (ax2, rewards_ppo, "PPO", "#4ECDC4")]:
        episodes = np.arange(1, len(rewards) + 1)
        ax.plot(episodes, rewards, linewidth=1.5, alpha=0.3, color=color)
        ax.plot(episodes, uniform_filter1d(rewards, size=window), linewidth=2.5,
                label=f"{name} (moyenne mobile)", color=color)
        ax.set_xlabel("Episode", fontsize=13, fontweight="bold")
        ax.set_title(name, fontsize=15, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
    ax1.set_ylabel("Reward cumulée par épisode", fontsize=13, fontweight="bold")
    fig.suptitle(title, fontsize=16, fontweight="bold")
    _finish(fig, save_path, show)

    print(f"A3C — reward moyenne (100 derniers épisodes) : {np.mean(rewards_a3c[-100:]):.2f}")
    print(f"PPO — reward moyenne (100 derniers épisodes) : {np.mean(rewards_ppo[-100:]):.2f}")


def plot_reward_and_distance(rewards, distances, title, save_path=None, show=False, window=20):
    """Reward et distance finale d'un même run, l'une au-dessus de l'autre."""
    episodes = np.arange(1, len(rewards) + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    ax1.plot(episodes, rewards, linewidth=1.5, alpha=0.3, color="#4ECDC4")
    ax1.plot(episodes, uniform_filter1d(rewards, size=window), linewidth=2.5,
             label="Reward (moyenne mobile)", color="#4ECDC4")
    ax1.set_ylabel("Reward cumulée", fontsize=13, fontweight="bold")
    ax1.set_title(title, fontsize=15, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(episodes, distances, linewidth=1.5, alpha=0.3, color="#FF6B6B")
    ax2.plot(episodes, uniform_filter1d(distances, size=window), linewidth=2.5,
             label="Distance (moyenne mobile)", color="#FF6B6B")
    ax2.set_xlabel("Episode", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Distance à la cible", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    _finish(fig, save_path, show)


def plot_distance_curve(distances, title, ylabel, label, save_path=None, show=False, window=20):
    """Évolution de l'erreur finale par épisode (tests isolés d'orientation)."""
    episodes = np.arange(1, len(distances) + 1)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(episodes, distances, linewidth=1.5, alpha=0.3, color="#FF6B6B")
    ax.plot(episodes, uniform_filter1d(distances, size=window), linewidth=2.5, label=label, color="#FF6B6B")
    ax.set_xlabel("Episode", fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    _finish(fig, save_path, show)

    print(f"Erreur moyenne (100 derniers épisodes) : {np.mean(distances[-100:]):.4f}")
    print(f"Erreur minimale atteinte : {np.min(distances):.4f}")


def plot_training_results(rewards, distances, title, save_path=None, show=False, window_size=20,
                          thresholds=((0.08, "green", ":", "Seuil 0.08"),
                                      (0.02, "orange", "--", "Seuil 0.02")),
                          distance_label="Distance finale"):
    """Courbes de convergence reward / distance avec moyennes mobiles et lignes de seuil.

    Les seuils affichés sont indicatifs : la « distance » dépend de l'environnement
    (somme position + angle en rad pour les envs quaternion, distance cosinus x100
    pour l'env ResNet). Ils sont donc passés en paramètre.
    """
    episodes = np.arange(1, len(rewards) + 1)

    def moving_average(data, w):
        return np.convolve(data, np.ones(w) / w, mode="valid")

    ma_episodes = episodes[window_size - 1:]
    style = "seaborn-v0_8-whitegrid"
    with plt.style.context(style if style in plt.style.available else "default"):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        ax1.plot(episodes, rewards, alpha=0.25, color="dodgerblue", label="Reward brut")
        ax1.plot(ma_episodes, moving_average(rewards, window_size), color="navy", linewidth=2,
                 label=f"Moyenne mobile ({window_size} ép.)")
        ax1.set_ylabel("Récompense totale", fontsize=12, fontweight="bold")
        ax1.set_title(f"Courbes de convergence - {title}", fontsize=14, fontweight="bold", pad=12)
        ax1.legend(loc="upper left")
        ax1.grid(True, linestyle="--", alpha=0.6)

        ax2.plot(episodes, distances, alpha=0.25, color="crimson", label="Distance brute")
        ax2.plot(ma_episodes, moving_average(distances, window_size), color="darkred", linewidth=2,
                 label=f"Moyenne mobile ({window_size} ép.)")
        for y, color, ls, label in thresholds:
            ax2.axhline(y=y, color=color, linestyle=ls, linewidth=1.5, label=label)
        ax2.set_xlabel("Épisodes", fontsize=12, fontweight="bold")
        ax2.set_ylabel(distance_label, fontsize=12, fontweight="bold")
        ax2.legend(loc="upper right")
        ax2.grid(True, linestyle="--", alpha=0.6)
        _finish(fig, save_path, show)


def show_image(img, title, save_path=None, show=False):
    """Affiche une vue rendue (ex. aperçu de la cible)."""
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    _finish(fig, save_path, show)
