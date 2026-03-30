from sre_openenv import SREOpenEnv
from inference import select_action, _state_to_obs

from graders.hard_grader import grade_hard

import random


def to_dict(state):
    return _state_to_obs(state)


def _obs_to_raw_state(obs):
    return [
        float(obs.get("frontend_status", 2)),
        float(obs.get("backend_status", 2)),
        float(obs.get("db_status", 2)),
        float(obs.get("frontend_latency", 100.0)),
        float(obs.get("backend_latency", 120.0)),
        float(obs.get("db_latency", 80.0)),
        float(obs.get("error_rate", 0.0)),
        float(obs.get("traffic_load", 0.5)),
        float(obs.get("traffic_balance", 0.5)),
        float(obs.get("request_queue", 100.0)),
    ]


# -------- EDGE CASE SCENARIOS --------
def inject_edge_case(env, case_id):
    obs = env.reset()
    raw = _obs_to_raw_state(obs)

    if case_id == 1:
        # Extreme traffic spike
        raw[7] = 0.99

    elif case_id == 2:
        # All services degraded
        raw[0] = 1
        raw[1] = 1
        raw[2] = 1

    elif case_id == 3:
        # High latency everywhere
        raw[3] = 1500
        raw[4] = 1500
        raw[5] = 1500

    elif case_id == 4:
        # Queue overload
        raw[9] = 1000

    elif case_id == 5:
        # Multi failure
        raw[0] = 0
        raw[2] = 0
        raw[7] = 0.95

    env.env.current_state = raw
    return env.state()


def run_case(case_id, seed=None):
    if seed is None:
        seed = 1000 + int(case_id)

    env = SREOpenEnv(seed=seed)
    state = inject_edge_case(env, case_id)

    max_steps = 40
    score = 0.0

    for step in range(max_steps):
        action = select_action(state)
        state, _, done, _ = env.step(action)

        score = grade_hard(to_dict(state))

        if done and score > 0.7:
            return score, step + 1

    return score, max_steps


def main():
    print("\n===== ROBUSTNESS TEST =====")

    passed = True

    for case in range(1, 6):
        scores = []

        for run_idx in range(5):
            score, _steps = run_case(case, seed=1000 + case * 10 + run_idx)
            scores.append(score)

        avg_score = sum(scores) / len(scores)

        print(f"\nCase {case}: Avg Score = {avg_score:.2f}")

        if avg_score < 0.4:
            print("FAIL")
            passed = False
        else:
            print("PASS")

    print("\n===== FINAL =====")

    if passed:
        print("ROBUSTNESS PASS")
    else:
        print("ROBUSTNESS FAIL")


if __name__ == "__main__":
    main()
