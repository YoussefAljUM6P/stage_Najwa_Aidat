import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def animate_episode(locations, path_history, obs_history, 
                    target_id, scene_name, save_path,
                    target_rotation=0, start_rotation=0):
    
    # Toutes les positions de la scène
    all_x = [loc[0] for loc in locations]
    all_y = [loc[1] for loc in locations]
    
    # Position cible
    target_x = locations[target_id][0]
    target_y = locations[target_id][1]
    
    # Position départ
    start_x = path_history[0]['x']
    start_y = path_history[0]['y']
    
    # Créer la figure avec 2 sous-graphes
    fig, (ax_obs, ax_map) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Scene: {scene_name} | Target: {target_id}', 
                 fontsize=14, fontweight='bold')
    
    # ── Sous-graphe gauche : vue first-person ──
    ax_obs.set_title("Vue First-Person")
    ax_obs.axis('off')
    img_display = ax_obs.imshow(
        np.zeros((300, 400, 3), dtype=np.uint8)
    )
    step_text = ax_obs.text(
        0.5, -0.05, '', transform=ax_obs.transAxes,
        ha='center', fontsize=11
    )
    
    # ── Sous-graphe droit : carte 2D ──
    ax_map.set_title("Carte 2D de la Scène")
    
    # Toutes les positions possibles
    ax_map.scatter(all_x, all_y, c='lightgray', 
                   s=80, zorder=1, label='Positions')
    
    # Cible
    ax_map.scatter(target_x, target_y, c='red', 
                   s=300, marker='*', zorder=4, label='Cible')
    
    # Départ
    ax_map.scatter(start_x, start_y, c='teal', 
                   s=200, marker='o', zorder=3, label='Départ')
    
    # Chemin (ligne bleue)
    line, = ax_map.plot([], [], 'b-', linewidth=2, zorder=2)
    
    # Agent (point bleu)
    agent, = ax_map.plot([], [], 'bo', markersize=8, zorder=5)

    # Conteneur pour la flèche agent (recréée à chaque frame)
    arrow_artist = [None]

    # ── Longueur des flèches adaptée à l'échelle de la carte ──
    x_range = max(all_x) - min(all_x)
    y_range = max(all_y) - min(all_y)
    arrow_len = 0.08 * max(x_range, y_range)

    def _rotation_to_rad(rotation):
        """Convertit la rotation (degrés, convention AI2-THOR/yaw) en radians
        pour matplotlib.
        AI2-THOR : 0° = Nord (+Y), sens horaire.
        Mapping : angle_mpl = 90° - rotation (sens trigonométrique)."""
        rot_deg = float(np.squeeze(rotation))
        return np.deg2rad(90.0 - rot_deg)

    # ── Flèche fixe : orientation initiale de l'agent au départ ──
    start_angle_rad = _rotation_to_rad(start_rotation)
    dx_start = arrow_len * np.cos(start_angle_rad)
    dy_start = arrow_len * np.sin(start_angle_rad)
    ax_map.annotate(
        '',
        xy=(start_x + dx_start, start_y + dy_start),
        xytext=(start_x, start_y),
        arrowprops=dict(arrowstyle='->', color='teal', lw=2.5),
        zorder=6
    )

    # ── Flèche fixe : orientation d'arrivée à la cible ──
    target_angle_rad = _rotation_to_rad(target_rotation)
    dx_target = arrow_len * np.cos(target_angle_rad)
    dy_target = arrow_len * np.sin(target_angle_rad)
    ax_map.annotate(
        '',
        xy=(target_x + dx_target, target_y + dy_target),
        xytext=(target_x, target_y),
        arrowprops=dict(arrowstyle='->', color='red', lw=2.5),
        zorder=6
    )

    ax_map.legend(loc='upper right')
    ax_map.grid(True, alpha=0.3)
    ax_map.set_xlabel("X")
    ax_map.set_ylabel("Y")
    
    plt.tight_layout()

    # ── Animation ──
    cumulative_reward = [0.0]

    def init():
        img_display.set_data(np.zeros((300, 400, 3), dtype=np.uint8))
        line.set_data([], [])
        agent.set_data([], [])
        step_text.set_text('')
        return img_display, line, agent, step_text

    def update(frame):
        if frame == 0:
            cumulative_reward[0] = 0.0

        # ── Vue first-person ──
        img_display.set_data(obs_history[frame])
        
        # ── Chemin et position agent ──
        xs = [path_history[i]['x'] for i in range(frame + 1)]
        ys = [path_history[i]['y'] for i in range(frame + 1)]
        line.set_data(xs, ys)
        agent.set_data([xs[-1]], [ys[-1]])
        
        # ── Reward cumulé ──
        is_terminal = path_history[frame]['is_terminal']
        step_reward = 10.0 if is_terminal else -0.01
        cumulative_reward[0] += step_reward
        step_text.set_text(
            f'Pas {frame + 1}  |  '
            f'Reward étape: {step_reward:+.2f}  |  '
            f'Reward cumulé: {cumulative_reward[0]:+.2f}'
        )
        
        # ── Couleur agent ──
        arrow_color = 'red' if is_terminal else 'dodgerblue'
        agent.set_color(arrow_color)
        agent.set_markersize(15 if is_terminal else 8)

        # ── Flèche agent (orientation courante, recréée à chaque frame) ──
        if arrow_artist[0] is not None:
            arrow_artist[0].remove()
            arrow_artist[0] = None

        cx, cy = xs[-1], ys[-1]
        rotation = path_history[frame].get('rotation', 0)
        angle_rad = _rotation_to_rad(rotation)
        dx = arrow_len * np.cos(angle_rad)
        dy = arrow_len * np.sin(angle_rad)

        arrow_artist[0] = ax_map.annotate(
            '',
            xy=(cx + dx, cy + dy),
            xytext=(cx, cy),
            arrowprops=dict(
                arrowstyle='->',
                color=arrow_color,
                lw=2.5
            ),
            zorder=7
        )
        
        return img_display, line, agent, step_text
    
    ani = animation.FuncAnimation(
        fig, update,
        frames=len(path_history),
        init_func=init,
        interval=600,
        blit=False
    )
    
    ani.save(save_path, writer='pillow', fps=2)
    plt.close()
    print(f"Animation sauvegardée : {save_path}")