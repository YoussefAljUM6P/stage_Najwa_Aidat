# -*- coding: utf-8 -*-
import torch
import numpy as np
import random
import time
import sys
import csv
import os

from utils.accum_trainer import AccumTrainer
from scene_loader import THORDiscreteEnvironment as Environment
from network import ActorCriticFFNetwork
from visualize import animate_episode

from constants import ACTION_SIZE
from constants import GAMMA
from constants import LOCAL_T_MAX
from constants import ENTROPY_BETA
from constants import VERBOSE

os.makedirs('animations', exist_ok=True)


class A3CTrainingThread(object):
    def __init__(self,
                 thread_index,
                 global_network,
                 initial_learning_rate,
                 grad_applier,
                 max_global_time_step,
                 device="cpu",
                 network_scope="network",
                 scene_scope="scene",
                 task_scope="task"):

        self.thread_index          = thread_index
        self.max_global_time_step  = max_global_time_step
        self.initial_learning_rate = initial_learning_rate

        self.network_scope = network_scope
        self.scene_scope   = scene_scope
        self.task_scope    = task_scope
        self.scopes        = [network_scope, scene_scope, task_scope]
        self._device       = device

        self.local_network = ActorCriticFFNetwork(
            action_size=ACTION_SIZE,
            device=device,
            network_scope=network_scope,
            scene_scopes=[scene_scope]
        )
        self.local_network.prepare_loss(ENTROPY_BETA, self.scopes)

        self.trainer = AccumTrainer(device=device)
        self.trainer.prepare_minimize(None, list(self.local_network.parameters()))

        self.grad_applier   = grad_applier
        self.global_network = global_network

        self.env            = None
        self.local_t        = 0
        self.episode_reward = 0
        self.episode_length = 0
        self.episode_max_q  = -np.inf
        # BUG FIX: episode_count était utilisé mais jamais déclaré dans __init__
        self.episode_count  = 0

    def _anneal_learning_rate(self, global_time_step):
        time_step_to_go = max(self.max_global_time_step - global_time_step, 0.0)
        return self.initial_learning_rate * time_step_to_go / self.max_global_time_step

    def choose_action(self, pi_values):
        cumsum = 0.0
        r = random.random()
        for i, rate in enumerate(pi_values):
            cumsum += rate
            if cumsum >= r:
                return i
        return len(pi_values) - 1

    def _record_score(self, writer, values, global_t):
        if writer is not None:
            for k, v in values.items():
                writer.add_scalar(
                    self.scene_scope + '/' + self.task_scope + '/' + k, v, global_t
                )

    def _sync_from_global(self):
        self.local_network.sync_from(self.global_network)

    def process(self, global_t, summary_writer=None):
        if self.env is None:
            time.sleep(self.thread_index * 1.0)
            self.env = Environment({
                'scene_name': self.scene_scope,
                'terminal_state_id': int(self.task_scope),
                'h5_file_path': '/kaggle/working/scene_3dgs.h5'
            })
            self.env.reset()

        states  = []
        actions = []
        rewards = []
        values  = []
        targets = []

        terminal_end = False

        self._sync_from_global()
        self.trainer.reset_gradients()

        start_local_t = self.local_t

        for i in range(LOCAL_T_MAX):
            pi_, value_ = self.local_network.run_policy_and_value(
                self.env.s_t, self.env.target, self.scopes
            )
            action = self.choose_action(pi_)

            states.append(self.env.s_t)
            actions.append(action)
            values.append(value_)
            targets.append(self.env.target)

            if VERBOSE and (self.thread_index == 0) and (self.local_t % 1000) == 0:
                sys.stdout.write("Pi = {0} V = {1}\n".format(pi_, value_))

            self.env.step(action)

            terminal = self.env.terminal
            reward   = 10.0 if terminal else -0.01
            if self.episode_length > 5e3:
                terminal = True

            self.episode_reward += reward
            self.episode_length += 1
            self.episode_max_q   = max(self.episode_max_q, np.max(value_))

            rewards.append(np.clip(reward, -1, 1))
            self.local_t += 1
            self.env.update()

            if terminal:
                terminal_end = True
                sys.stdout.write(
                    "time %d | thread #%d | scene %s | target #%s\n"
                    "%s %s episode reward = %.3f\n"
                    "%s %s episode length = %d\n"
                    "%s %s episode max Q  = %.3f\n" % (
                        global_t, self.thread_index, self.scene_scope, self.task_scope,
                        self.scene_scope, self.task_scope, self.episode_reward,
                        self.scene_scope, self.task_scope, self.episode_length,
                        self.scene_scope, self.task_scope, self.episode_max_q
                    )
                )
                summary_values = {
                    "episode_reward": self.episode_reward,
                    "episode_length": float(self.episode_length),
                    "episode_max_q":  self.episode_max_q,
                    "learning_rate":  self._anneal_learning_rate(global_t)
                }
                self._record_score(summary_writer, summary_values, global_t)

                # CSV logging
                with open('rewards_log.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        global_t,
                        self.thread_index,
                        self.task_scope,
                        self.episode_reward,
                        self.episode_length,
                        self.episode_max_q
                    ])

                # Animation tous les 20 épisodes
                self.episode_count += 1
                if self.episode_count % 20 == 0:
                    save_path = f'animations/episode_{global_t}.gif'
                    animate_episode(
                        self.env.locations,
                        self.env.path_history,
                        self.env.obs_history,
                        int(self.task_scope),
                        self.scene_scope,
                        save_path
                    )

                self.episode_reward = 0
                self.episode_length = 0
                self.episode_max_q  = -np.inf
                self.env.reset()
                break

        R = 0.0
        if not terminal_end:
            R = self.local_network.run_value(self.env.s_t, self.env.target, self.scopes)

        actions.reverse()
        states.reverse()
        rewards.reverse()
        values.reverse()
        targets.reverse()

        batch_si, batch_a, batch_td, batch_R, batch_t = [], [], [], [], []

        for ai, ri, si, Vi, ti in zip(actions, rewards, states, values, targets):
            R  = ri + GAMMA * R
            td = R - Vi
            a  = np.zeros(ACTION_SIZE)
            a[ai] = 1

            batch_si.append(si)
            batch_a.append(a)
            batch_td.append(td)
            batch_R.append(R)
            batch_t.append(ti)

        loss = self.local_network.compute_loss(
            batch_si, batch_t, batch_a, batch_td, batch_R, self.scopes
        )
        self.trainer.accumulate_gradients(loss)

        cur_lr = self._anneal_learning_rate(global_t)
        self.grad_applier.apply_gradients(
            list(self.global_network.parameters()),
            self.trainer.get_accum_grad_list(),
            lr_override=cur_lr
        )

        if VERBOSE and (self.thread_index == 0) and (self.local_t % 100) == 0:
            sys.stdout.write("Local timestep %d\n" % self.local_t)

        return self.local_t - start_local_t
