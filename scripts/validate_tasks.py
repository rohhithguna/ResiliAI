from sre_openenv import SREOpenEnv

from tasks.easy_task import get_easy_task
from tasks.medium_task import get_medium_task
from tasks.hard_task import get_hard_task

from graders.easy_grader import grade_easy
from graders.medium_grader import grade_medium
from graders.hard_grader import grade_hard


def run_task(task, grader):
    env = SREOpenEnv(seed=42)
    state = env.reset()

    print(f"\n=== Running: {task['name']} ===")

    for step in range(task["max_steps"]):
        state, reward, done, _ = env.step(0)

        score = grader(state)

        print(f"Step {step+1} | Score: {score:.2f}")

        if done:
            break

    final_score = grader(state)
    print(f"Final Score: {final_score:.2f}")


if __name__ == "__main__":
    run_task(get_easy_task(), grade_easy)
    run_task(get_medium_task(), grade_medium)
    run_task(get_hard_task(), grade_hard)
