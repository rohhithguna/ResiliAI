import random

try:
    import numpy as np
except Exception:
    np = None

try:
    import torch
except Exception:
    torch = None

from src.env.sre_openenv import SREOpenEnv
from src.inference.inference import select_action


def run_episode(seed):
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)

    env = SREOpenEnv(seed=seed)
    state = env.reset()

    total_score = 0
    steps = 0

    for _ in range(40):
        action = select_action(state)
        state, reward, done, _ = env.step(action)

        err = float(state.get("error_rate", 1.0)) if isinstance(state, dict) else 1.0

        score = 1.0 if err < 0.1 else 0.0
        total_score += score
        steps += 1

        if done and err < 0.2:
            break

    return total_score / steps, steps


def benchmark(runs=10):
    random.seed(42)
    if np is not None:
        np.random.seed(42)
    if torch is not None:
        torch.manual_seed(42)

    scores = []
    steps_list = []

    fixed_seeds = [1000 + i for i in range(runs)]
    for seed in fixed_seeds:
        s, st = run_episode(seed)
        scores.append(s)
        steps_list.append(st)

    return {
        "avg_score": sum(scores) / len(scores),
        "avg_steps": sum(steps_list) / len(steps_list)
    }


if __name__ == "__main__":
    print("===== BENCHMARK =====")
    print(benchmark())
