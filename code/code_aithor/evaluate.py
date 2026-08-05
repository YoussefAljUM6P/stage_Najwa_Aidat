# -*- coding: utf-8 -*-
import torch
import numpy as np
import random

from network import ActorCriticFFNetwork
from scene_loader import THORDiscreteEnvironment as Environment

from constants import (
    ACTION_SIZE, CHECKPOINT_DIR, NUM_EVAL_EPISODES,
    VERBOSE, TASK_TYPE, TASK_LIST, USE_GPU
)

import os


def sample_action(pi_values):
    """Échantillonne une action selon la distribution pi."""
    cumsum = 0.0
    r = random.random()
    for i, p in enumerate(pi_values):
        cumsum += p
        if cumsum >= r:
            return i
    return len(pi_values) - 1


if __name__ == '__main__':

    device = "cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu"
    print(f"Using device: {device}")

    network_scope = TASK_TYPE
    list_of_tasks = TASK_LIST
    scene_scopes  = list(list_of_tasks.keys())

    # Réseau global avec toutes les scènes
    global_network = ActorCriticFFNetwork(
        action_size=ACTION_SIZE,
        device=device,
        network_scope=network_scope,
        scene_scopes=scene_scopes
    )
    global_network.eval()

    # Chargement du checkpoint PyTorch
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'checkpoint.pt')
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        global_network.load_state_dict(ckpt['model_state_dict'])
        global_t = ckpt.get('global_t', 0)
        print(f"Checkpoint chargé : {checkpoint_path} | global_t = {global_t}")
    else:
        print("Aucun checkpoint trouvé — évaluation avec poids aléatoires.")

    scene_stats = dict()

    for scene_scope in scene_scopes:
        scene_stats[scene_scope] = []

        for task_scope in list_of_tasks[scene_scope]:
            env = Environment({
                'scene_name': scene_scope,
                'terminal_state_id': int(task_scope)
            })

            scopes = [network_scope, scene_scope, task_scope]

            ep_rewards    = []
            ep_lengths    = []
            ep_collisions = []

            for i_episode in range(NUM_EVAL_EPISODES):
                env.reset()
                terminal     = False
                ep_reward    = 0
                ep_collision = 0
                ep_t         = 0

                while not terminal:
                    pi_values = global_network.run_policy(env.s_t, env.target, scopes)
                    action    = sample_action(pi_values)

                    env.step(action)
                    env.update()

                    terminal = env.terminal
                    if ep_t >= 10000:
                        break
                    if env.collided:
                        ep_collision += 1
                    ep_reward += env.reward
                    ep_t += 1

                ep_lengths.append(ep_t)
                ep_rewards.append(ep_reward)
                ep_collisions.append(ep_collision)

                if VERBOSE:
                    print(f"  épisode #{i_episode:3d} | {ep_t} pas | "
                          f"reward={ep_reward:.2f} | collisions={ep_collision}")

            print(f"\nÉvaluation : {scene_scope} → cible #{task_scope}")
            print(f"  reward moyen    : {np.mean(ep_rewards):.2f}")
            print(f"  longueur moy.   : {np.mean(ep_lengths):.2f} pas")
            print(f"  collisions moy. : {np.mean(ep_collisions):.2f}")

            scene_stats[scene_scope].extend(ep_lengths)

    print('\n=== Résultats (longueur moyenne des trajectoires) ===')
    for scene_scope in scene_stats:
        print(f"  {scene_scope}: {np.mean(scene_stats[scene_scope]):.2f} pas")