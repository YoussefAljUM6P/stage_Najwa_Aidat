# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActorCriticFFNetwork(nn.Module):
    def __init__(self, action_size, device="cpu", network_scope="network", scene_scopes=["scene"]):
        super(ActorCriticFFNetwork, self).__init__()
        self._device = device
        self._action_size = action_size
        self._network_scope = network_scope
        self._scene_scopes = scene_scopes

        # Shared layers
        self.W_fc1 = nn.Linear(8192, 512)
        self.W_fc2 = nn.Linear(1024, 512)

        # Scene-specific layers
        # BUG FIX: '/' est interdit dans les clés de nn.ModuleDict (PyTorch).
        # On utilise '__' comme séparateur à la place.
        self.W_fc3    = nn.ModuleDict()
        self.W_policy = nn.ModuleDict()
        self.W_value  = nn.ModuleDict()

        for scene_scope in scene_scopes:
            key = self._get_key([network_scope, scene_scope])
            self.W_fc3[key]    = nn.Linear(512, 512)
            self.W_policy[key] = nn.Linear(512, action_size)
            self.W_value[key]  = nn.Linear(512, 1)

        self._init_weights()
        self.to(device)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                d = 1.0 / np.sqrt(m.in_features)
                nn.init.uniform_(m.weight, -d, d)
                nn.init.uniform_(m.bias, -d, d)

    def _get_key(self, scopes):
        # BUG FIX: '__' au lieu de '/' (interdit dans les clés ModuleDict)
        return '__'.join(scopes)

    def forward(self, s, t, scene_scope):
        key = self._get_key([self._network_scope, scene_scope])

        s_flat = s.view(-1, 8192)
        t_flat = t.view(-1, 8192)

        h_s   = F.relu(self.W_fc1(s_flat))
        h_t   = F.relu(self.W_fc1(t_flat))
        h_fc1 = torch.cat([h_s, h_t], dim=1)

        h_fc2 = F.relu(self.W_fc2(h_fc1))
        h_fc3 = F.relu(self.W_fc3[key](h_fc2))

        pi = F.softmax(self.W_policy[key](h_fc3), dim=1)
        v  = self.W_value[key](h_fc3).squeeze(1)

        return pi, v

    def run_policy_and_value(self, s_t, target, scopes):
        scene_scope = scopes[1]
        s = torch.FloatTensor(s_t).unsqueeze(0).to(self._device)
        t = torch.FloatTensor(target).unsqueeze(0).to(self._device)
        with torch.no_grad():
            pi, v = self.forward(s, t, scene_scope)
        return pi[0].cpu().numpy(), v[0].cpu().item()

    def run_policy(self, s_t, target, scopes):
        scene_scope = scopes[1]
        s = torch.FloatTensor(s_t).unsqueeze(0).to(self._device)
        t = torch.FloatTensor(target).unsqueeze(0).to(self._device)
        with torch.no_grad():
            pi, _ = self.forward(s, t, scene_scope)
        return pi[0].cpu().numpy()

    def run_value(self, s_t, target, scopes):
        scene_scope = scopes[1]
        s = torch.FloatTensor(s_t).unsqueeze(0).to(self._device)
        t = torch.FloatTensor(target).unsqueeze(0).to(self._device)
        with torch.no_grad():
            _, v = self.forward(s, t, scene_scope)
        return v[0].cpu().item()

    def get_vars(self):
        return list(self.parameters())

    def sync_from(self, src_network):
        """
        BUG FIX: load_state_dict() direct échoue car le network_scope est
        différent entre global ('navigation') et local ('thread-1'), ce qui
        change les clés des ModuleDict dans le state_dict.
        On copie les poids couche par couche en normalisant les clés.
        """
        src_state = src_network.state_dict()
        dst_state = self.state_dict()
        src_scope = src_network._network_scope
        dst_scope = self._network_scope

        new_state = {}
        for dst_key, dst_val in dst_state.items():
            # Normaliser la clé destination (remplacer son scope par placeholder)
            normalized = dst_key.replace(dst_scope, 'SCOPE', 1)
            matched = None
            for src_key in src_state:
                if src_key.replace(src_scope, 'SCOPE', 1) == normalized:
                    matched = src_key
                    break
            new_state[dst_key] = src_state[matched] if matched else dst_val

        self.load_state_dict(new_state)

    def prepare_loss(self, entropy_beta, scopes):
        self._entropy_beta = entropy_beta

    def compute_loss(self, s_batch, t_batch, a_batch, td_batch, r_batch, scopes):
        scene_scope = scopes[1]
        s  = torch.FloatTensor(np.array(s_batch)).to(self._device)
        t  = torch.FloatTensor(np.array(t_batch)).to(self._device)
        a  = torch.FloatTensor(np.array(a_batch)).to(self._device)
        td = torch.FloatTensor(np.array(td_batch)).to(self._device)
        r  = torch.FloatTensor(np.array(r_batch)).to(self._device)

        pi, v = self.forward(s, t, scene_scope)

        log_pi      = torch.log(torch.clamp(pi, 1e-20, 1.0))
        entropy     = -torch.sum(pi * log_pi, dim=1)
        policy_loss = -torch.sum(
            torch.sum(log_pi * a, dim=1) * td + entropy * self._entropy_beta
        )
        value_loss  = 0.5 * F.mse_loss(v, r, reduction='sum')

        return policy_loss + value_loss
