"""
Boucles d'entraînement A3C et PPO.



Deux familles de boucles, qui reprennent les deux styles du notebook :
  * *_mlp      : une mise à jour par épisode, modèle ActorCriticMLP (envs 1 à 4)
  * *_batched / *_std_scale : modèle ActorCritic avec std_scale (envs 5 à 7)

Chaque fonction renvoie un dict {"rewards": [...], "distances": [...], "time": s}.
"""
import time

import numpy as np
import torch
import torch.nn.functional as F


def discounted_returns(rewards, bootstrap_value, gamma):
    returns = []
    R = bootstrap_value
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


def normalize(advantages):
    if advantages.numel() > 1 and advantages.std() > 1e-6:
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages


def annealed_entropy_coef(episode, num_episodes):
    """Coefficient d'entropie décroissant linéairement de 0.02 à 0.005."""
    return max(0.005, 0.02 * (1 - episode / num_episodes))


def _to_tensor(state_np, device):
    return torch.from_numpy(state_np).unsqueeze(0).to(device)


# ===========================================================================
# Mise à jour par épisode (ActorCriticMLP)
# ===========================================================================
def train_a3c_mlp(env, model, num_episodes=3000, lr=1e-4, gamma=0.99,
                  reset_kwargs=None, log_every=10, device="cpu"):
    reset_kwargs = reset_kwargs or {}
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rewards_hist, dist_hist = [], []
    t0 = time.time()
    print(f"🔴 Entraînement A3C (device={device})...")

    for episode in range(1, num_episodes + 1):
        state = _to_tensor(env.reset(**reset_kwargs)[0], device)
        log_probs, values, rewards, entropies = [], [], [], []
        ep_reward, terminated = 0.0, False
        info = {"mse": env.prev_distance}

        for _ in range(env.max_steps):
            action, log_prob, value, entropy = model.get_action_with_entropy(state)
            state_np, reward, terminated, truncated, info = env.step(action[0].detach().cpu().numpy())
            state = _to_tensor(state_np, device)
            ep_reward += reward
            log_probs.append(log_prob.view(-1))
            values.append(value.view(-1))
            rewards.append(reward)
            entropies.append(entropy.view(-1))
            if terminated or truncated:
                break

        with torch.no_grad():
            bootstrap = 0.0 if terminated else model(state)[1].item()
        returns = torch.tensor(discounted_returns(rewards, bootstrap, gamma),
                               dtype=torch.float32, device=device)
        values = torch.cat(values)
        advantages = normalize(returns - values.detach())

        actor_loss = -(torch.cat(log_probs) * advantages).mean()
        critic_loss = F.mse_loss(values, returns)
        entropy_loss = torch.cat(entropies).mean()
        loss = actor_loss + 0.5 * critic_loss - annealed_entropy_coef(episode, num_episodes) * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        rewards_hist.append(ep_reward)
        dist_hist.append(info["mse"])
        if episode % log_every == 0:
            print(f"Episode {episode}/{num_episodes} | Avg Reward ({log_every} ep): "
                  f"{np.mean(rewards_hist[-log_every:]):.2f} | "
                  f"Final Distance: {np.mean(dist_hist[-log_every:]):.4f}")

    elapsed = time.time() - t0
    print(f"⏱️ Temps A3C : {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    return {"rewards": rewards_hist, "distances": dist_hist, "time": elapsed}


def train_ppo_mlp(env, model, num_episodes=3000, lr=1e-4, gamma=0.99, clip_eps=0.2, ppo_epochs=4,
                  step_lr=None, reset_kwargs=None, log_every=10, device="cpu"):
    """PPO avec une mise à jour (ppo_epochs passes) à la fin de chaque épisode.

    step_lr : None, ou (step_size, gamma) pour un StepLR avancé une fois par épisode.
    """
    reset_kwargs = reset_kwargs or {}
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, *step_lr) if step_lr else None
    rewards_hist, dist_hist = [], []
    t0 = time.time()
    print(f"🟦 Entraînement PPO (device={device})...")

    for episode in range(1, num_episodes + 1):
        state = _to_tensor(env.reset(**reset_kwargs)[0], device)
        states, actions, old_log_probs, old_values, rewards = [], [], [], [], []
        ep_reward, terminated = 0.0, False
        info = {"mse": env.prev_distance}

        for _ in range(env.max_steps):
            with torch.no_grad():
                action, log_prob, value, _ = model.get_action_with_entropy(state)
            state_np, reward, terminated, truncated, info = env.step(action[0].cpu().numpy())
            ep_reward += reward
            states.append(state)
            actions.append(action.view(-1))
            old_log_probs.append(log_prob.view(-1))
            old_values.append(value.view(-1))
            rewards.append(reward)
            state = _to_tensor(state_np, device)
            if terminated or truncated:
                break

        with torch.no_grad():
            bootstrap = 0.0 if terminated else model(state)[1].item()
        returns = torch.tensor(discounted_returns(rewards, bootstrap, gamma),
                               dtype=torch.float32, device=device)
        values_old = torch.cat(old_values).detach()
        log_probs_old = torch.cat(old_log_probs).detach()
        actions = torch.stack(actions).detach()
        states_batch = torch.cat(states).detach()
        advantages = normalize(returns - values_old)
        entropy_coef = annealed_entropy_coef(episode, num_episodes)

        for _ in range(ppo_epochs):
            mean, values_new = model(states_batch)
            dist = torch.distributions.Normal(mean, model.get_std())
            log_probs_new = dist.log_prob(actions).sum(dim=-1)

            ratio = torch.exp(log_probs_new - log_probs_old)
            actor_loss = -torch.min(ratio * advantages,
                                    torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages).mean()

            # Critique avec clipping de la valeur
            values_new = values_new.view(-1)
            values_clipped = values_old + torch.clamp(values_new - values_old, -clip_eps, clip_eps)
            critic_loss = torch.max(F.mse_loss(values_new, returns, reduction="none"),
                                    F.mse_loss(values_clipped, returns, reduction="none")).mean()

            loss = actor_loss + 0.5 * critic_loss - entropy_coef * dist.entropy().sum(dim=-1).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()   # une fois par épisode

        rewards_hist.append(ep_reward)
        dist_hist.append(info["mse"])
        if episode % log_every == 0:
            msg = (f"Episode {episode}/{num_episodes} | Avg Reward ({log_every} ep): "
                   f"{np.mean(rewards_hist[-log_every:]):.2f} | "
                   f"Final Distance: {np.mean(dist_hist[-log_every:]):.4f}")
            if scheduler is not None:
                msg += f" | LR: {optimizer.param_groups[0]['lr']:.2e}"
            print(msg)

    elapsed = time.time() - t0
    print(f"⏱️ Temps PPO : {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    return {"rewards": rewards_hist, "distances": dist_hist, "time": elapsed}


# ===========================================================================
# ActorCritic avec std_scale
# ===========================================================================
def train_ppo_batched(env, model, total_episodes=3000, lr=1e-3, gamma=0.95,
                      episodes_per_batch=10, ppo_epochs=5, clip_eps=0.2,
                      use_std_scale=True, value_clipping=True, entropy="annealed",
                      min_lr=0.0, reset_kwargs=None, log_every=20, device="cpu"):
    """PPO multi-rollouts : `episodes_per_batch` épisodes collectés, puis `ppo_epochs`
    passes d'optimisation sur le lot. LR réduit par ReduceLROnPlateau sur la distance finale.

    use_std_scale  : écart-type d'exploration multiplié par max(0.3, 1 - episode / total).
    value_clipping : clipping de la valeur pour la perte du critique.
    entropy        : "annealed" (0.02 -> 0.005) ou "fixed" (0.01).
    """
    reset_kwargs = reset_kwargs or {}
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15, min_lr=min_lr)
    rewards_hist, dist_hist = [], []
    t0 = time.time()
    print(f"🟩 Entraînement PPO multi-rollouts (device={device})...")

    episode = 0
    while episode < total_episodes:
        std_scale = max(0.3, 1.0 - episode / total_episodes) if use_std_scale else 1.0
        b_states, b_actions, b_log_probs, b_returns, b_values = [], [], [], [], []

        # --- Collecte ---
        for _ in range(episodes_per_batch):
            episode += 1
            state = _to_tensor(env.reset(**reset_kwargs)[0], device)
            states, actions, log_probs, values, rewards = [], [], [], [], []
            ep_reward, terminated = 0.0, False

            for _ in range(env.max_steps):
                with torch.no_grad():
                    action, log_prob, value = model.get_action(state, std_scale=std_scale)
                next_state_np, reward, terminated, truncated, info = env.step(action[0].cpu().numpy())
                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                ep_reward += reward
                state = _to_tensor(next_state_np, device)
                if terminated or truncated:
                    break

            with torch.no_grad():
                bootstrap = 0.0 if terminated else model(state)[1].item()
            b_returns.append(torch.tensor(discounted_returns(rewards, bootstrap, gamma),
                                          dtype=torch.float32, device=device))
            b_values.append(torch.cat(values).squeeze(-1))
            b_states.append(torch.cat(states))
            b_actions.append(torch.cat(actions))
            b_log_probs.append(torch.cat(log_probs))
            dist_hist.append(info["mse"])
            rewards_hist.append(ep_reward)

        # --- Optimisation ---
        states_b = torch.cat(b_states)
        actions_b = torch.cat(b_actions)
        log_probs_b = torch.cat(b_log_probs)
        returns_b = torch.cat(b_returns)
        values_b = torch.cat(b_values)
        advantages_b = normalize(returns_b - values_b)

        for _ in range(ppo_epochs):
            mean, vals_new = model(states_b)
            vals_new = vals_new.squeeze(-1)
            dist = torch.distributions.Normal(mean, torch.exp(model.log_std) * std_scale)
            new_log_probs = dist.log_prob(actions_b).sum(dim=-1)

            ratio = torch.exp(new_log_probs - log_probs_b)
            actor_loss = -torch.min(ratio * advantages_b,
                                    torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages_b).mean()

            if value_clipping:
                vals_clipped = values_b + torch.clamp(vals_new - values_b, -clip_eps, clip_eps)
                critic_loss = torch.max(F.mse_loss(vals_new, returns_b, reduction="none"),
                                        F.mse_loss(vals_clipped, returns_b, reduction="none")).mean()
            else:
                critic_loss = F.mse_loss(vals_new, returns_b)

            entropy_coef = annealed_entropy_coef(episode, total_episodes) if entropy == "annealed" else 0.01
            loss = actor_loss + 0.5 * critic_loss - entropy_coef * dist.entropy().sum(dim=-1).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

        avg_d = np.mean(dist_hist[-episodes_per_batch:])
        scheduler.step(avg_d)

        if episode % log_every == 0:
            print(f"Episode {episode}/{total_episodes} | Avg Reward: {np.mean(rewards_hist[-log_every:]):6.2f} | "
                  f"Final Distance: {avg_d:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"std_scale: {std_scale:.2f}")

    elapsed = time.time() - t0
    print(f"⏱️ Temps PPO : {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    return {"rewards": rewards_hist, "distances": dist_hist, "time": elapsed}


def train_a3c_std_scale(env, model, num_episodes=3000, lr=1e-3, gamma=0.95, min_lr=1e-5,
                        reset_kwargs=None, log_every=20, device="cpu"):
    """A3C (1 worker) avec ActorCritic : std_scale décroissant, pas de terme d'entropie,
    ReduceLROnPlateau sur la distance finale moyenne des 10 derniers épisodes."""
    reset_kwargs = reset_kwargs or {}
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15, min_lr=min_lr)
    rewards_hist, dist_hist = [], []
    t0 = time.time()
    print(f"🔴 Entraînement A3C (device={device})...")

    for episode in range(1, num_episodes + 1):
        std_scale = max(0.3, 1.0 - episode / num_episodes)
        state = _to_tensor(env.reset(**reset_kwargs)[0], device)
        log_probs, values, rewards = [], [], []
        ep_reward, terminated = 0.0, False

        for _ in range(env.max_steps):
            action, log_prob, value = model.get_action(state, std_scale=std_scale)
            next_state_np, reward, terminated, truncated, info = env.step(action[0].detach().cpu().numpy())
            log_probs.append(log_prob.view(-1))
            values.append(value.view(-1))
            rewards.append(reward)
            ep_reward += reward
            state = _to_tensor(next_state_np, device)
            if terminated or truncated:
                break

        with torch.no_grad():
            bootstrap = 0.0 if terminated else model(state)[1].item()
        returns = torch.tensor(discounted_returns(rewards, bootstrap, gamma),
                               dtype=torch.float32, device=device)
        values = torch.cat(values)
        advantages = normalize(returns - values.detach())

        loss = -(torch.cat(log_probs) * advantages).mean() + 0.5 * F.mse_loss(values, returns)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        dist_hist.append(info["mse"])
        rewards_hist.append(ep_reward)
        avg_d = np.mean(dist_hist[-10:])
        scheduler.step(avg_d)

        if episode % log_every == 0:
            print(f"Episode {episode}/{num_episodes} | Avg Reward: {np.mean(rewards_hist[-log_every:]):6.2f} | "
                  f"Final Distance: {avg_d:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"std_scale: {std_scale:.2f}")

    elapsed = time.time() - t0
    print(f"⏱️ Temps A3C : {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    return {"rewards": rewards_hist, "distances": dist_hist, "time": elapsed}
