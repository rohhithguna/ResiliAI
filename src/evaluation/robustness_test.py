from sre_openenv import SREOpenEnv
from inference import select_action

import random


def robustness_test(runs=10):
    random.seed(123)

    success = 0

    for _ in range(runs):
        env = SREOpenEnv(seed=random.randint(0, 100000))
        state = env.reset()

        for _ in range(40):
            action = select_action(state)
            state, reward, done, _ = env.step(action)

            try:
                err = state[6]
            except Exception:
                err = state.get("error_rate", 1.0)

            if err < 0.1:
                success += 1
                break

            if done:
                break

    return success / runs


if __name__ == "__main__":
    print("===== ROBUSTNESS =====")
    print(robustness_test())
