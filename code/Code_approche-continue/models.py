"""
Réseaux acteur-critique et extracteur de features.

- ActorCriticMLP : tronc partagé (128-128-64), moyenne en tanh, écart-type appris
  (softplus, borné). Utilisé pour les environnements 1 à 4 (état = poses/angles).
- ActorCritic    : acteur et critique séparés, log-écart-type appris, avec un
  facteur `std_scale` pour réduire l'exploration au cours de l'entraînement.
  Utilisé pour les environnements 5 et 6 (hidden=(64, 64), log_std_init=-1.5)
  et 7 (state_dim=512, hidden=(128, 64), log_std_init=-1.8).
- ResNetFeatureExtractor : ResNet-18 pré-entraîné ImageNet, gelé, sans la couche
  de classification (vecteur de 512 features).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCriticMLP(nn.Module):
    def __init__(self, input_dim, action_dim, min_std=1e-3, max_std=0.5):
        super().__init__()
        self.min_std = min_std
        self.max_std = max_std

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.actor_mean = nn.Linear(64, action_dim)
        self.actor_std = nn.Parameter(torch.ones(action_dim) * 0.5)
        self.critic = nn.Linear(64, 1)

    def forward(self, state):
        x = self.fc(state)
        return torch.tanh(self.actor_mean(x)), self.critic(x)

    def get_std(self):
        return (F.softplus(self.actor_std) + self.min_std).clamp(max=self.max_std)

    def get_action_with_entropy(self, state):
        mean, value = self.forward(state)
        dist = torch.distributions.Normal(mean, self.get_std())
        action = torch.clamp(dist.sample(), -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1)   # log-proba de l'action bornée
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, value, entropy


class ActorCritic(nn.Module):
    def __init__(self, state_dim=7, action_dim=6, hidden=(64, 64), log_std_init=-1.5):
        super().__init__()
        h1, h2 = hidden
        self.actor = nn.Sequential(
            nn.Linear(state_dim, h1), nn.Tanh(),
            nn.Linear(h1, h2), nn.Tanh(),
            nn.Linear(h2, action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim) + log_std_init)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, h1), nn.Tanh(),
            nn.Linear(h1, h2), nn.Tanh(),
            nn.Linear(h2, 1),
        )

    def forward(self, state):
        return self.actor(state), self.critic(state)

    def get_action(self, state, std_scale=1.0):
        mean, value = self.forward(state)
        std = torch.exp(self.log_std) * std_scale
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)   # log-proba de l'action AVANT clamp (cf. README)
        return torch.clamp(action, -1.0, 1.0), log_prob, value


# ---------------------------------------------------------------------------
# Features visuelles
# ---------------------------------------------------------------------------
def _build_resnet_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


resnet_transform = _build_resnet_transform()


class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        import torchvision.models as models
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])   # sans la couche fc
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, x):
        return torch.flatten(self.backbone(x), start_dim=1)
