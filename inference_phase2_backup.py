import random

try:
    import numpy as np
except Exception:
    np = None

try:
    import torch
except Exception:
    torch = None

from sre_openenv import SREOpenEnv

from tasks.easy_task import get_easy_task
from tasks.medium_task import get_medium_task
from tasks.hard_task import get_hard_task

from graders.easy_grader import grade_easy
from graders.medium_grader import grade_medium
from graders.hard_grader import grade_hard


rl_used = 0
rule_used = 0


def _set_global_seed(seed=42):
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


_set_global_seed(42)


def _safe_get(state, idx, key, default):
    if isinstance(state, dict):
        return state.get(key, default)
    if isinstance(state, (list, tuple)) and idx < len(state):
        return state[idx]
    return default


def _state_to_obs(state):
    return {
        "frontend_status": int(_safe_get(state, 0, "frontend_status", 0)),
        "backend_status": int(_safe_get(state, 1, "backend_status", 0)),
        "db_status": int(_safe_get(state, 2, "db_status", 0)),
        "frontend_latency": float(_safe_get(state, 3, "frontend_latency", 0.0)),
        "backend_latency": float(_safe_get(state, 4, "backend_latency", 0.0)),
        "db_latency": float(_safe_get(state, 5, "db_latency", 0.0)),
        "error_rate": float(_safe_get(state, 6, "error_rate", 1.0)),
        "traffic_load": float(_safe_get(state, 7, "traffic_load", 0.0)),
        "traffic_balance": float(_safe_get(state, 8, "traffic_balance", 0.0)),
        "request_queue": float(_safe_get(state, 9, "request_queue", 0.0)),
    }


from multi_agent import CoordinatorAgent

coordinator = CoordinatorAgent()


def select_action(state):
    return coordinator.select_action(state)


def run_task(task, grader):
    env = SREOpenEnv(seed=42)
    state = env.reset()

    total_reward = 0.0
    steps = 0
    max_steps = task["max_steps"]

    print(f"\n=== Running: {task['name']} ===")

    for step in range(max_steps):
        action = select_action(state)

        next_state, reward, done, _ = env.step(action)

        obs = _state_to_obs(next_state)
        score = grader(obs)

        print(f"Step {step+1} | Action: {action} | Score: {score:.2f}")

        state = next_state
        total_reward += reward
        steps += 1

        # Do not terminate immediately on done; only exit early on high-quality recovery.
        if done and score > 0.8:
            break

    final_score = grader(_state_to_obs(state))

    print(f"Final Score: {final_score:.2f}")

    return {
        "task": task["name"],
        "score": final_score,
        "confidence": final_score,
        "steps": steps,
        "total_reward": total_reward,
        # Keep legacy key for compatibility with existing scripts.
        "final_score": final_score,
    }


def run_all():
    return [
        run_task(get_easy_task(), grade_easy),
        run_task(get_medium_task(), grade_medium),
        run_task(get_hard_task(), grade_hard),
    ]


if __name__ == "__main__":
    print("=== INFERENCE START ===")

    results = run_all()

    print("\n=== SUMMARY ===")
    for r in results:
        print(f"{r['task']} -> score: {r['score']:.2f}, steps: {r['steps']}")

    print("\n=== INFERENCE COMPLETE ===")
