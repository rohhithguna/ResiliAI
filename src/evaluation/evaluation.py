from sre_openenv import SREOpenEnv
from inference import select_action
from agent_rule import RuleAgent

import random


def _to_rule_state(state):
    if isinstance(state, dict):
        return [
            state.get("frontend_status", 2),
            state.get("backend_status", 2),
            state.get("db_status", 2),
            state.get("frontend_latency", 100.0),
            state.get("backend_latency", 120.0),
            state.get("db_latency", 80.0),
            state.get("error_rate", 0.0),
            state.get("traffic_load", 0.5),
        ]
    return state


def run_episode(env, action_fn, max_steps=40):
    state = env.reset()
    total_reward = 0
    steps = 0
    done = False

    for _ in range(max_steps):
        action = action_fn(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        steps += 1

        if done:
            break

    success = bool(done and getattr(env, "global_error_rate", 1.0) < 0.05)
    return total_reward, steps, success


def evaluate(episodes=30):
    rl_success = 0
    rl_reward = 0
    rl_steps = 0

    rule_success = 0
    rule_reward = 0
    rule_steps = 0

    rule_agent = RuleAgent()

    seeds = [random.randint(0, 100000) for _ in range(episodes)]

    for s in seeds:
        env_rl = SREOpenEnv(seed=s)
        env_rule = SREOpenEnv(seed=s)

        r_reward, r_steps, r_success = run_episode(
            env_rl,
            lambda st: select_action(st)
        )

        b_reward, b_steps, b_success = run_episode(
            env_rule,
            lambda st: rule_agent.select_action(_to_rule_state(st))
        )

        rl_reward += r_reward
        rl_steps += r_steps
        rl_success += int(r_success)

        rule_reward += b_reward
        rule_steps += b_steps
        rule_success += int(b_success)

    return {
        "rl": {
            "success": rl_success / episodes,
            "reward": rl_reward / episodes,
            "steps": rl_steps / episodes
        },
        "rule": {
            "success": rule_success / episodes,
            "reward": rule_reward / episodes,
            "steps": rule_steps / episodes
        }
    }


if __name__ == "__main__":
    print("===== EVALUATION START =====")
    results = evaluate(episodes=30)

    print("\n===== RESULTS =====")
    print(results)

    print("\n===== FINAL VERDICT =====")

    rl = results["rl"]
    rule = results["rule"]

    if rl["success"] > rule["success"]:
        print("RL wins on success")
    elif rl["reward"] > rule["reward"]:
        print("RL wins on reward")
    elif rl["steps"] < rule["steps"]:
        print("RL wins on speed")
    else:
        print("Rule still stronger or needs tuning")
