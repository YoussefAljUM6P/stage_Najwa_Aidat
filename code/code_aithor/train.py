# -*- coding: utf-8 -*-
import torch
import threading
import numpy as np
import signal
import os

from network import ActorCriticFFNetwork
from training_thread import A3CTrainingThread
from utils.rmsprop_applier import RMSPropApplier
from constants import (
    ACTION_SIZE, PARALLEL_SIZE,
    INITIAL_ALPHA_LOW, INITIAL_ALPHA_HIGH, INITIAL_ALPHA_LOG_RATE,
    MAX_TIME_STEP, CHECKPOINT_DIR,
    RMSP_EPSILON, RMSP_ALPHA, GRAD_NORM_CLIP,
    USE_GPU, TASK_TYPE, TASK_LIST
)


def log_uniform(lo, hi, rate):
    return np.exp(np.log(lo) * (1 - rate) + np.log(hi) * rate)


if __name__ == '__main__':

    device = "cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu"
    print(f"Using device: {device}")

    scene_scopes = list(TASK_LIST.keys())

    global_t       = 0
    stop_requested = False

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    initial_learning_rate = log_uniform(
        INITIAL_ALPHA_LOW, INITIAL_ALPHA_HIGH, INITIAL_ALPHA_LOG_RATE
    )
    print(f"Initial learning rate: {initial_learning_rate:.6f}")

    # Réseau global avec TOUTES les scènes
    global_network = ActorCriticFFNetwork(
        action_size=ACTION_SIZE,
        device=device,
        network_scope=TASK_TYPE,
        scene_scopes=scene_scopes
    )
    global_network.share_memory()

    # Liste de toutes les branches (scène, cible)
    branches = []
    for scene in scene_scopes:
        for task in TASK_LIST[scene]:
            branches.append((scene, task))

    NUM_TASKS = len(branches)
    assert PARALLEL_SIZE >= NUM_TASKS, \
        f"Pas assez de threads : il faut au minimum {NUM_TASKS} threads."

    # Un seul optimizer RMSProp sur les paramètres du réseau global
    grad_applier = RMSPropApplier(
        var_list=list(global_network.parameters()),
        learning_rate=initial_learning_rate,
        decay=RMSP_ALPHA,
        momentum=0.0,
        epsilon=RMSP_EPSILON,
        clip_norm=GRAD_NORM_CLIP,
        device=device
    )

    # Création des threads d'entraînement
    training_threads = []
    for i in range(PARALLEL_SIZE):
        scene, task = branches[i % NUM_TASKS]
        t = A3CTrainingThread(
            thread_index=i,
            global_network=global_network,
            initial_learning_rate=initial_learning_rate,
            grad_applier=grad_applier,
            max_global_time_step=MAX_TIME_STEP,
            device=device,
            network_scope=f"thread-{i + 1}",
            scene_scope=scene,
            task_scope=task
        )
        training_threads.append(t)

    # TensorBoard (optionnel)
    try:
        from torch.utils.tensorboard import SummaryWriter
        summary_writer = SummaryWriter(log_dir='logs')
    except ImportError:
        summary_writer = None
        print("TensorBoard non disponible.")

    # Chargement checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'checkpoint.pt')
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        global_network.load_state_dict(ckpt['model_state_dict'])
        global_t = ckpt.get('global_t', 0)
        print(f"Checkpoint chargé : global_t = {global_t}")
    else:
        print("Démarrage from scratch.")

    lock = threading.Lock()

    def train_function(parallel_index):
        global global_t
        training_thread = training_threads[parallel_index]
        last_save_t = global_t

        while global_t < MAX_TIME_STEP and not stop_requested:
            diff = training_thread.process(global_t, summary_writer)
            with lock:
                global_t += diff
                cur_t = global_t

            if parallel_index == 0 and cur_t - last_save_t > 10_000:
                print(f"[t={cur_t}] Sauvegarde checkpoint...")
                torch.save({
                    'model_state_dict': global_network.state_dict(),
                    'global_t': cur_t
                }, checkpoint_path)
                last_save_t = cur_t

    def signal_handler(sig, frame):
        global stop_requested
        print('\nCtrl+C reçu — arrêt en cours...')
        stop_requested = True

    signal.signal(signal.SIGINT, signal_handler)

    # Lancement des threads
    threads = []
    for i in range(PARALLEL_SIZE):
        t = threading.Thread(target=train_function, args=(i,))
        threads.append(t)

    for t in threads:
        t.start()

    print(f'Entraînement lancé : {PARALLEL_SIZE} threads, {len(scene_scopes)} scènes.')

    for t in threads:
        t.join()

    print('Sauvegarde finale...')
    torch.save({
        'model_state_dict': global_network.state_dict(),
        'global_t': global_t
    }, checkpoint_path)

    if summary_writer:
        summary_writer.close()

    print('Terminé.')