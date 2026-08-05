# -*- coding: utf-8 -*-
import torch
import threading
import numpy as np
import signal
import random
import os

from network import ActorCriticFFNetwork
from training_thread import A3CTrainingThread
from utils.rmsprop_applier import RMSPropApplier
from constants import ACTION_SIZE
from constants import PARALLEL_SIZE
from constants import INITIAL_ALPHA_LOW
from constants import INITIAL_ALPHA_HIGH
from constants import INITIAL_ALPHA_LOG_RATE
from constants import MAX_TIME_STEP
from constants import CHECKPOINT_DIR
from constants import RMSP_EPSILON
from constants import RMSP_ALPHA
from constants import GRAD_NORM_CLIP
from constants import USE_GPU
from constants import TASK_TYPE
from constants import TASK_LIST


def log_uniform(lo, hi, rate):
    log_lo = np.log(lo)
    log_hi = np.log(hi)
    v = log_lo * (1 - rate) + log_hi * rate
    return np.exp(v)


if __name__ == '__main__':

    device = "cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu"
    print(f"Using device: {device}")

    network_scope = TASK_TYPE
    list_of_tasks = TASK_LIST
    scene_scopes  = list(list_of_tasks.keys())

    global_t       = 0
    stop_requested = False

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    initial_learning_rate = log_uniform(
        INITIAL_ALPHA_LOW, INITIAL_ALPHA_HIGH, INITIAL_ALPHA_LOG_RATE
    )
    print(f"Initial learning rate: {initial_learning_rate:.6f}")

    global_network = ActorCriticFFNetwork(
        action_size=ACTION_SIZE,
        device=device,
        network_scope=network_scope,
        scene_scopes=scene_scopes
    )
    global_network.share_memory()

    branches = []
    for scene in scene_scopes:
        for task in list_of_tasks[scene]:
            branches.append((scene, task))

    NUM_TASKS = len(branches)
    assert PARALLEL_SIZE >= NUM_TASKS, \
        "Not enough threads for multitasking: at least {} threads needed.".format(NUM_TASKS)

    grad_applier = RMSPropApplier(
        learning_rate=initial_learning_rate,
        decay=RMSP_ALPHA,
        momentum=0.0,
        epsilon=RMSP_EPSILON,
        clip_norm=GRAD_NORM_CLIP,
        device=device
    )

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
            network_scope="thread-%d" % (i + 1),
            scene_scope=scene,
            task_scope=task
        )
        training_threads.append(t)

    try:
        from torch.utils.tensorboard import SummaryWriter
        summary_writer = SummaryWriter(log_dir='logs')
    except ImportError:
        summary_writer = None
        print("TensorBoard non disponible, pas de logs.")

    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'checkpoint.pt')
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        global_network.load_state_dict(ckpt['model_state_dict'])
        global_t = ckpt.get('global_t', 0)
        print(f"Checkpoint chargé : {checkpoint_path} | global_t = {global_t}")
    else:
        print("Aucun checkpoint trouvé, démarrage from scratch.")

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

            # BUG FIX: MAX_TIME_STEP = 150 000, la sauvegarde tous les
            # 1 000 000 pas ne se déclenchait jamais. Seuil réduit à 10 000.
            if parallel_index == 0 and cur_t - last_save_t > 10_000:
                print(f"Sauvegarde checkpoint à t={cur_t}")
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

    train_threads = []
    for i in range(PARALLEL_SIZE):
        train_threads.append(threading.Thread(target=train_function, args=(i,)))

    for t in train_threads:
        t.start()

    print('Entraînement lancé. Appuyez sur Ctrl+C pour arrêter.')

    for t in train_threads:
        t.join()

    print('Sauvegarde finale...')
    torch.save({
        'model_state_dict': global_network.state_dict(),
        'global_t': global_t
    }, checkpoint_path)

    if summary_writer:
        summary_writer.close()

    print('Terminé.')
