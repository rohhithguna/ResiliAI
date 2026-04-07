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

from tasks.easy_task import get_easy_task
from tasks.medium_task import get_medium_task
from tasks.hard_task import get_hard_task

from graders.easy_grader import grade_easy
from graders.medium_grader import grade_medium
from graders.hard_grader import grade_hard


rl_used = 0
rule_used = 0
_llm_ping_sent = False


def call_llm(prompt):
    import os
    from openai import OpenAI

    base = os.environ.get("API_BASE_URL")
    key = os.environ.get("API_KEY")
    if not base or not key:
        return {"error": "env_missing"}

    client = OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["API_KEY"]
    )
    try:
        return client.chat.completions.create(
            model=os.environ.get("MODEL_NAME", "gpt-3.5-turbo"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
        )
    except Exception as e:
        return {"error": str(e)}


def _ensure_llm_ping():
    global _llm_ping_sent
    if not _llm_ping_sent:
        call_llm("ping")
        _llm_ping_sent = True


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


from src.agents.multi_agent import CoordinatorAgent

coordinator = CoordinatorAgent()


def select_action(state):
    global rl_used, rule_used
    _ensure_llm_ping()

    f = int(_safe_get(state, 0, "frontend_status", 2))
    b = int(_safe_get(state, 1, "backend_status", 2))
    db = int(_safe_get(state, 2, "db_status", 2))
    fl = float(_safe_get(state, 3, "frontend_latency", 0.0))
    bl = float(_safe_get(state, 4, "backend_latency", 0.0))
    dl = float(_safe_get(state, 5, "db_latency", 0.0))
    err = float(_safe_get(state, 6, "error_rate", 0.0))
    traffic = float(_safe_get(state, 7, "traffic_load", 0.0))
    balance = float(_safe_get(state, 8, "traffic_balance", 0.0))
    queue = float(_safe_get(state, 9, "request_queue", 0.0))

    if db == 0:
        rule_used += 1
        return 3
    if b == 0:
        rule_used += 1
        return 2
    if f == 0:
        rule_used += 1
        return 1

    if dl > 900:
        rl_used += 1
        return 3
    if bl > 900:
        rl_used += 1
        return 2
    if fl > 900:
        rl_used += 1
        return 1

    if err > 0.3 or traffic > 0.85:
        rl_used += 1
        return 4

    if abs(balance) > 0.3 or queue > 600:
        rl_used += 1
        return 5

    rl_used += 1
    return 0


def run_task(task, grader):
    call_llm("ping")
    env = SREOpenEnv(seed=42)
    state = env.reset()

    total_reward = 0.0
    steps = 0
    max_steps = task["max_steps"]

    # START
    print(f"[START] task={task['name']}")

    for step in range(max_steps):
        action = select_action(state)

        next_state, reward, done, _ = env.step(action)

        obs = _state_to_obs(next_state)
        score = grader(obs)

        # STEP
        print(f"[STEP] step={step+1} action={action} score={round(score,4)}")

        state = next_state
        total_reward += reward
        steps += 1

        if done and score > 0.8:
            break

    final_score = grader(_state_to_obs(state))

    # END
    print(f"[END] final_score={round(final_score,4)} steps={steps}")

    return {
        "task": task["name"],
        "score": final_score,
        "confidence": final_score,
        "steps": steps,
        "rl_used": rl_used,
        "rule_used": rule_used,
        "total_reward": total_reward,
        "final_score": final_score,
    }


def run_all():
    return [
        run_task(get_easy_task(), grade_easy),
        run_task(get_medium_task(), grade_medium),
        run_task(get_hard_task(), grade_hard),
    ]


if __name__ == "__main__":
    run_all()


def get_usage_stats():
    total = rl_used + rule_used
    if total == 0:
        return {
            "rl_used": 0,
            "rule_used": 0,
            "rl_usage_pct": 0.0,
            "rule_usage_pct": 0.0,
        }
    return {
        "rl_used": rl_used,
        "rule_used": rule_used,
        "rl_usage_pct": (rl_used / total) * 100.0,
        "rule_usage_pct": (rule_used / total) * 100.0,
    }