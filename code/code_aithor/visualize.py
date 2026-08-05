# -*- coding: utf-8 -*-


import matplotlib
matplotlib.use('Agg')          

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def rotation_to_vector(rotation_deg):
    """0 = Nord (+y), 90 = Est (+x), 180 = Sud (-y), 270 = Ouest (-x)."""
    rad = np.radians(float(rotation_deg))
    return np.sin(rad), np.cos(rad)


def animate_episode(locations, path_history, obs_history,
                    target_id, scene_name, save_path,
                    target_rotation=0, start_rotation=0):

    if not path_history:
        return

    all_x = [loc[0] for loc in locations]
    all_y = [loc[1] for loc in locations]

    target_x, target_y = locations[target_id][0], locations[target_id][1]
    start_x,  start_y  = path_history[0]['x'], path_history[0]['y']

    fig, (ax_obs, ax_map) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Scene: {scene_name} | Target: {target_id}',
                 fontsize=14, fontweight='bold')

    # --- Vue first-person ---
    first_img = np.asarray(obs_history[0])
    ax_obs.set_title("Vue First-Person")
    ax_obs.axis('off')
    img_display = ax_obs.imshow(np.zeros_like(first_img))
    step_text = ax_obs.text(0.5, -0.05, '', transform=ax_obs.transAxes,
                            ha='center', fontsize=11)

    # --- Carte 2D ---
    ax_map.set_title("Carte 2D de la Scène")
    ax_map.scatter(all_x, all_y, c='lightgray', s=80, zorder=1, label='Positions')
    ax_map.scatter(target_x, target_y, c='red', s=300, marker='*',
                   zorder=4, label='Cible')

    tdx, tdy = rotation_to_vector(target_rotation)
    ax_map.quiver(target_x, target_y, tdx, tdy, color='red', scale=8,
                  zorder=4, width=0.008)

    ax_map.scatter(start_x, start_y, c='green', s=200, marker='o',
                   zorder=3, label='Départ')

    line, = ax_map.plot([], [], 'b-', linewidth=2, zorder=2)

    sdx, sdy = rotation_to_vector(start_rotation)
    agent_quiver = ax_map.quiver(start_x, start_y, sdx, sdy, color='blue',
                                 scale=8, zorder=5, width=0.012)

    ax_map.legend(loc='upper right')
    ax_map.grid(True, alpha=0.3)
    ax_map.set_xlabel("X")
    ax_map.set_ylabel("Y")
    plt.tight_layout()

    n_frames = min(len(path_history), len(obs_history))

    def init():
        img_display.set_data(np.zeros_like(first_img))
        line.set_data([], [])
        agent_quiver.set_offsets([[start_x, start_y]])
        agent_quiver.set_UVC(sdx, sdy)
        step_text.set_text('')
        return img_display, line, agent_quiver, step_text

    def update(frame):
        # obs_history contient directement les images issues du HDF5
        img_display.set_data(np.asarray(obs_history[frame]))

        xs = [path_history[i]['x'] for i in range(frame + 1)]
        ys = [path_history[i]['y'] for i in range(frame + 1)]
        line.set_data(xs, ys)

        rotation = path_history[frame].get('rotation', 0)
        dx, dy = rotation_to_vector(rotation)
        agent_quiver.set_offsets([[xs[-1], ys[-1]]])
        agent_quiver.set_UVC(dx, dy)

        is_terminal = path_history[frame]['is_terminal']
        reward = 10.0 if is_terminal else -0.01
        step_text.set_text(
            f'Pas {frame + 1} | Reward: {reward} | Rotation: {rotation}°'
        )
        agent_quiver.set_color('red' if is_terminal else 'blue')

        return img_display, line, agent_quiver, step_text

    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                  init_func=init, interval=600, blit=True)
    ani.save(save_path, writer='pillow', fps=2)
    plt.close(fig)
    print(f"Animation sauvegardée : {save_path}")