import random
import statistics

import torch
import torch.nn as nn
import torch.optim as optim

from sre_environment import SREEnvironment


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 10_000) -> None:
        self.capacity = capacity
        self.buffer = []

    def add(self, state, action: int, reward: float, next_state, done: bool) -> None:
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)

    def len(self) -> int:
        return len(self.buffer)


class RewardNormalizer:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    def normalize(self, value: float) -> float:
        if self.count < 2:
            return value
        var = self.m2 / (self.count - 1)
        std = max(var**0.5, 1e-6)
        normalized = (value - self.mean) / std
        return max(-5.0, min(5.0, normalized))


def state_to_tensor(state) -> torch.Tensor:
    return torch.tensor(state, dtype=torch.float32)


def select_action(q_net: QNetwork, state, epsilon: float, action_dim: int) -> int:
    if random.random() < epsilon:
        return random.randrange(action_dim)

    with torch.no_grad():
        state_tensor = state_to_tensor(state).unsqueeze(0)
        q_values = q_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())


def train_step(
    q_net: QNetwork,
    target_net: QNetwork,
    optimizer: optim.Optimizer,
    buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
    tau: float,
) -> float:
    if len(buffer) < batch_size:
        return 0.0

    batch = buffer.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states_t = torch.tensor(states, dtype=torch.float32)
    actions_t = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
    rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states_t = torch.tensor(next_states, dtype=torch.float32)
    dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

    q_values = q_net(states_t).gather(1, actions_t)

    with torch.no_grad():
        # Double DQN target: select via online net, evaluate via target net.
        next_actions = q_net(next_states_t).argmax(dim=1, keepdim=True)
        next_q = target_net(next_states_t).gather(1, next_actions)
        target_q = rewards_t + gamma * next_q * (1.0 - dones_t)

    loss = nn.MSELoss()(q_values, target_q)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
    optimizer.step()

    # Soft target update.
    for target_param, param in zip(target_net.parameters(), q_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    return float(loss.item())


def evaluate_success_rate(q_net: QNetwork, seeds, max_steps: int, action_dim: int) -> float:
    q_net.eval()
    successes = 0
    with torch.no_grad():
        for seed in seeds:
            env = SREEnvironment(seed=seed)
            state = env.reset()
            done = False
            for _ in range(max_steps):
                action = select_action(q_net, state, 0.0, action_dim=action_dim)
                state, _, done, _ = env.step(action)
                if done:
                    break
            if done and env.global_error_rate < 0.02:
                successes += 1
    return successes * 100.0 / len(seeds)


def main() -> None:
    random.seed(42)
    torch.manual_seed(42)

    env = SREEnvironment(seed=42)

    state_dim = 10
    action_dim = 6

    episodes = 1100
    max_steps = 40
    batch_size = 32
    gamma = 0.99
    learning_rate = 0.001

    epsilon_start = 1.0
    epsilon_end = 0.05
    tau = 0.005

    q_net = QNetwork(state_dim=state_dim, action_dim=action_dim)
    target_net = QNetwork(state_dim=state_dim, action_dim=action_dim)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)
    buffer = ReplayBuffer(capacity=10_000)
    reward_normalizer = RewardNormalizer()
    episode_rewards = []
    success_count = 0
    best_val_success = float("-inf")
    best_val_reward = float("-inf")
    validation_seeds = [seed for seed in range(1000, 1050)]
    eval_every = 25
    loss_history = []

    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        if episodes > 1:
            progress = (episode - 1) / (episodes - 1)
        else:
            progress = 1.0
        epsilon = epsilon_start - (epsilon_start - epsilon_end) * (progress ** 0.8)
        epsilon = max(epsilon_end, epsilon)

        for _ in range(max_steps):
            if random.random() < epsilon:
                # Explore full 6-action space without rule-policy bias.
                action = random.randrange(action_dim)
            else:
                action = select_action(q_net, state, 0.0, action_dim)
            next_state, reward, done, _ = env.step(action)

            reward_normalizer.update(reward)
            normalized_reward = reward_normalizer.normalize(reward)

            buffer.add(state, action, normalized_reward, next_state, done)
            loss_value = train_step(
                q_net,
                target_net,
                optimizer,
                buffer,
                batch_size,
                gamma,
                tau,
            )
            if loss_value > 0.0:
                loss_history.append(loss_value)

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        success = bool(steps > 0 and done and env.global_error_rate < 0.02)
        if success:
            success_count += 1
        episode_rewards.append(total_reward)

        print(
            f"Episode {episode:03d} | Reward: {total_reward:8.3f} | "
            f"Steps: {steps:2d} | Success: {success} | Epsilon: {epsilon:.3f}"
        )

        if episode % eval_every == 0:
            val_success = evaluate_success_rate(q_net, validation_seeds, max_steps, action_dim)
            val_reward = statistics.mean(episode_rewards[-min(30, len(episode_rewards)):])
            print(f"Validation | episode={episode} success={val_success:.2f}% recent_reward={val_reward:.3f}")
            if val_success > best_val_success or (
                val_success == best_val_success and val_reward > best_val_reward
            ):
                best_val_success = val_success
                best_val_reward = val_reward
                torch.save(q_net.state_dict(), "dqn_model.pth")
                print(
                    f"New best model saved. val_success={best_val_success:.2f}% "
                    f"recent_reward={best_val_reward:.3f}"
                )

    last_n = min(20, len(episode_rewards))
    avg_last_20 = sum(episode_rewards[-last_n:]) / last_n if last_n > 0 else 0.0
    success_rate = (success_count / episodes) * 100.0 if episodes > 0 else 0.0

    print("\nTraining summary")
    print(f"Average reward (last {last_n} episodes): {avg_last_20:.3f}")
    print(f"Total success count: {success_count}")
    print(f"Success rate: {success_rate:.2f}%")
    if loss_history:
        print(f"Average training loss (last {min(100, len(loss_history))}): {statistics.mean(loss_history[-100:]):.4f}")
    print(f"Best validation success: {best_val_success:.2f}%")


if __name__ == "__main__":
    main()
