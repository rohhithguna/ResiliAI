from sre_openenv import SREOpenEnv
from inference import select_action

import random


def run_episode(seed):
    env = SREOpenEnv(seed=seed)
    state = env.reset()

    total_score = 0
    steps = 0

    for _ in range(40):
        action = select_action(state)
        state, reward, done, _ = env.step(action)

        # scoring proxy
        try:
            err = state[6]
        except Exception:
            err = state.get("error_rate", 1.0)

        score = 1.0 if err < 0.1 else 0.0
        total_score += score
        steps += 1

        if done:
            break

    return total_score / steps, steps


def benchmark(runs=10):
    random.seed(42)

    scores = []
    steps_list = []

    for _ in range(runs):
        seed = random.randint(0, 100000)
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
