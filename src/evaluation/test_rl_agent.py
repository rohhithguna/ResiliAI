import math
import random
import statistics
import sys

import torch

from agent_rule import RuleAgent
from multi_agent import CoordinatorAgent
from sre_environment import SREEnvironment


class QNetwork(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.model(x)


def run_episode(env, max_steps, action_fn):
    state = env.reset()
    total_reward = 0.0
    steps = 0
    done = False

    for _ in range(max_steps):
        action = action_fn(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        steps += 1
        if done:
            break

    success = bool(done and env.global_error_rate < 0.02)
    return total_reward, steps, success


def evaluate_run(model, run_seed, episodes, max_steps):
    rule_agent = RuleAgent()
    coordinator = CoordinatorAgent()
    rng = random.Random(run_seed)
    episode_seeds = [rng.randint(0, 1_000_000) for _ in range(episodes)]

    metrics = {
        "rule": {"success": 0, "reward": 0.0, "steps": 0.0},
        "rl": {"success": 0, "reward": 0.0, "steps": 0.0},
        "hybrid": {"success": 0, "reward": 0.0, "steps": 0.0},
    }

    rl_action_count = 0
    rule_override_count = 0

    for seed in episode_seeds:
        env_rule = SREEnvironment(seed=seed)
        r_reward, r_steps, r_success = run_episode(
            env_rule,
            max_steps,
            lambda s: rule_agent.select_action(s),
        )
        metrics["rule"]["success"] += int(r_success)
        metrics["rule"]["reward"] += r_reward
        metrics["rule"]["steps"] += r_steps

        env_rl = SREEnvironment(seed=seed)
        rl_reward, rl_steps, rl_success = run_episode(
            env_rl,
            max_steps,
            lambda s: int(
                torch.argmax(model(torch.tensor(s, dtype=torch.float32).unsqueeze(0)), dim=1).item()
            ),
        )
        metrics["rl"]["success"] += int(rl_success)
        metrics["rl"]["reward"] += rl_reward
        metrics["rl"]["steps"] += rl_steps

        env_hybrid = SREEnvironment(seed=seed)
        h_state = env_hybrid.reset()
        h_reward = 0.0
        h_steps = 0
        h_done = False
        for _ in range(max_steps):
            h_action, source = coordinator.select_action_with_source(h_state, model)
            if source == "rl":
                rl_action_count += 1
            else:
                rule_override_count += 1
            h_state, reward, h_done, _ = env_hybrid.step(h_action)
            h_reward += reward
            h_steps += 1
            if h_done:
                break
        h_success = bool(h_done and env_hybrid.global_error_rate < 0.02)
        metrics["hybrid"]["success"] += int(h_success)
        metrics["hybrid"]["reward"] += h_reward
        metrics["hybrid"]["steps"] += h_steps

    for method in metrics:
        metrics[method]["success_rate"] = metrics[method]["success"] * 100.0 / episodes
        metrics[method]["avg_reward"] = metrics[method]["reward"] / episodes
        metrics[method]["avg_steps"] = metrics[method]["steps"] / episodes

    total_hybrid_decisions = rl_action_count + rule_override_count
    rl_usage = (rl_action_count / total_hybrid_decisions * 100.0) if total_hybrid_decisions else 0.0
    override_usage = (rule_override_count / total_hybrid_decisions * 100.0) if total_hybrid_decisions else 0.0

    return metrics, rl_usage, override_usage


def mean_std_ci(values):
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    n = len(values)

    # Small-sample t critical values for 95% CI (two-tailed), df = n - 1.
    t_critical_95 = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
    }
    t_val = t_critical_95.get(max(1, n - 1), 1.96)
    ci95 = t_val * (s / math.sqrt(n)) if n > 1 else 0.0
    return m, s, ci95


def summarize_runs(run_results):
    methods = ["rule", "rl", "hybrid"]
    summary = {}

    for method in methods:
        success_vals = [r[0][method]["success_rate"] for r in run_results]
        reward_vals = [r[0][method]["avg_reward"] for r in run_results]
        steps_vals = [r[0][method]["avg_steps"] for r in run_results]

        sm, ss, sci = mean_std_ci(success_vals)
        rm, rs, rci = mean_std_ci(reward_vals)
        tm, ts, tci = mean_std_ci(steps_vals)

        summary[method] = {
            "success_mean": sm,
            "success_std": ss,
            "success_ci": sci,
            "reward_mean": rm,
            "reward_std": rs,
            "reward_ci": rci,
            "steps_mean": tm,
            "steps_std": ts,
            "steps_ci": tci,
        }

    rl_usage_vals = [r[1] for r in run_results]
    override_vals = [r[2] for r in run_results]
    rl_m, rl_s, rl_ci = mean_std_ci(rl_usage_vals)
    ov_m, ov_s, ov_ci = mean_std_ci(override_vals)
    usage_summary = {
        "rl_usage_mean": rl_m,
        "rl_usage_std": rl_s,
        "rl_usage_ci": rl_ci,
        "override_mean": ov_m,
        "override_std": ov_s,
        "override_ci": ov_ci,
    }

    return summary, usage_summary


def main():
    torch.manual_seed(123)

    state_dim = 10
    action_dim = 6
    episodes = 100
    max_steps = 40
    run_seeds = [101, 202, 303, 404, 505, 606, 707, 808]

    model = QNetwork(state_dim=state_dim, action_dim=action_dim)
    try:
        state_dict = torch.load("dqn_model.pth", map_location=torch.device("cpu"))
    except FileNotFoundError:
        print("ERROR: Trained model not found. Run train_dqn.py first.")
        sys.exit(1)

    model.load_state_dict(state_dict)
    model.eval()

    run_results = []
    for run_idx, run_seed in enumerate(run_seeds, start=1):
        metrics, rl_usage, override_usage = evaluate_run(model, run_seed, episodes, max_steps)
        run_results.append((metrics, rl_usage, override_usage))

        print(f"Run {run_idx} seed={run_seed}")
        print(
            "Rule   -> "
            f"success={metrics['rule']['success_rate']:.2f}% "
            f"reward={metrics['rule']['avg_reward']:.4f} "
            f"steps={metrics['rule']['avg_steps']:.2f}"
        )
        print(
            "RL     -> "
            f"success={metrics['rl']['success_rate']:.2f}% "
            f"reward={metrics['rl']['avg_reward']:.4f} "
            f"steps={metrics['rl']['avg_steps']:.2f}"
        )
        print(
            "Hybrid -> "
            f"success={metrics['hybrid']['success_rate']:.2f}% "
            f"reward={metrics['hybrid']['avg_reward']:.4f} "
            f"steps={metrics['hybrid']['avg_steps']:.2f}"
        )
        print(f"Hybrid RL usage={rl_usage:.2f}% | Rule override={override_usage:.2f}%\n")

    summary, usage_summary = summarize_runs(run_results)

    print("========== ABLATION (8 runs x 100 episodes) ==========")
    for method in ["rule", "rl", "hybrid"]:
        s = summary[method]
        print(
            f"{method.upper()}: "
            f"Success {s['success_mean']:.2f}% +- {s['success_std']:.2f} (95% CI +- {s['success_ci']:.2f}), "
            f"Reward {s['reward_mean']:.4f} +- {s['reward_std']:.4f} (95% CI +- {s['reward_ci']:.4f}), "
            f"Steps {s['steps_mean']:.2f} +- {s['steps_std']:.2f} (95% CI +- {s['steps_ci']:.2f})"
        )

    print("\n========== HYBRID POLICY USAGE ==========")
    print(
        f"RL usage: {usage_summary['rl_usage_mean']:.2f}% +- {usage_summary['rl_usage_std']:.2f} "
        f"(95% CI +- {usage_summary['rl_usage_ci']:.2f})"
    )
    print(
        f"Rule override: {usage_summary['override_mean']:.2f}% +- {usage_summary['override_std']:.2f} "
        f"(95% CI +- {usage_summary['override_ci']:.2f})"
    )

    print("\n| Method | Success | Reward | Steps |")
    print("|--------|--------:|-------:|------:|")
    for method, label in [("rule", "Rule"), ("rl", "RL"), ("hybrid", "Hybrid")]:
        s = summary[method]
        print(
            f"| {label} | {s['success_mean']:.2f}% +- {s['success_std']:.2f} | "
            f"{s['reward_mean']:.4f} +- {s['reward_std']:.4f} | "
            f"{s['steps_mean']:.2f} +- {s['steps_std']:.2f} |"
        )

    hybrid = summary["hybrid"]
    rule = summary["rule"]
    dominance = 0
    dominance += int(hybrid["success_mean"] > rule["success_mean"])
    dominance += int(hybrid["reward_mean"] > rule["reward_mean"])
    dominance += int(hybrid["steps_mean"] < rule["steps_mean"])
    print(f"\nHybrid dominance vs Rule: {dominance}/3 metrics")


if __name__ == "__main__":
    main()
