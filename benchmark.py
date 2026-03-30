from inference import select_action, _state_to_obs
from sre_openenv import SREOpenEnv

from tasks.easy_task import get_easy_task
from tasks.medium_task import get_medium_task
from tasks.hard_task import get_hard_task

from graders.easy_grader import grade_easy
from graders.medium_grader import grade_medium
from graders.hard_grader import grade_hard

import random
import statistics


TASKS = [
    ("easy", get_easy_task, grade_easy),
    ("medium", get_medium_task, grade_medium),
    ("hard", get_hard_task, grade_hard),
]


def run_single(task_fn, grader, seed):
    env = SREOpenEnv(seed=seed)
    state = env.reset()

    total_score = 0.0
    steps = 0
    max_steps = task_fn()["max_steps"]

    for _ in range(max_steps):
        action = select_action(state)
        state, reward, done, _ = env.step(action)

        obs = _state_to_obs(state)

        score = grader(obs)
        total_score = score
        steps += 1

        if done and score > 0.8:
            break

    return total_score, steps


def benchmark(runs=10):
    results = {}

    for name, task_fn, grader in TASKS:
        scores = []
        steps_list = []

        for i in range(runs):
            seed = random.randint(0, 100000)
            score, steps = run_single(task_fn, grader, seed)

            scores.append(score)
            steps_list.append(steps)

        avg_score = statistics.mean(scores)
        std_score = statistics.stdev(scores) if len(scores) > 1 else 0.0
        avg_steps = statistics.mean(steps_list)

        results[name] = {
            "avg_score": avg_score,
            "std_score": std_score,
            "avg_steps": avg_steps,
            "min_score": min(scores),
            "max_score": max(scores),
        }

    return results


def print_results(results):
    print("\n===== BENCHMARK RESULTS =====")

    for task, res in results.items():
        print(f"\nTask: {task}")
        print(f"Avg Score : {res['avg_score']:.3f}")
        print(f"Std Dev   : {res['std_score']:.3f}")
        print(f"Min Score : {res['min_score']:.3f}")
        print(f"Max Score : {res['max_score']:.3f}")
        print(f"Avg Steps : {res['avg_steps']:.2f}")


def validate(results):
    print("\n===== VALIDATION =====")

    passed = True

    for task, res in results.items():
        if res["avg_score"] < 0.5:
            print(f"[FAIL] {task} avg score too low")
            passed = False
        else:
            print(f"[PASS] {task} avg score OK")

        if res["std_score"] > 0.4:
            print(f"[WARN] {task} unstable behavior")

    if passed:
        print("\nFINAL STATUS: PASS ✅")
    else:
        print("\nFINAL STATUS: FAIL ❌")


if __name__ == "__main__":
    results = benchmark(runs=10)
    print_results(results)
    validate(results)
