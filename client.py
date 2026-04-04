from src.env.sre_openenv import SREOpenEnv
from tasks.easy_task import get_easy_task
from tasks.medium_task import get_medium_task
from tasks.hard_task import get_hard_task
from src.inference.inference import run_task


class Client:
    def __init__(self):
        self.env = SREOpenEnv(seed=42)

    def reset(self, task_name="easy"):
        if task_name == "easy":
            self.task = get_easy_task()
        elif task_name == "medium":
            self.task = get_medium_task()
        else:
            self.task = get_hard_task()

        return self.task["initial_state"]

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        return state, reward, done, info

    def run(self, task_name="easy"):
        if task_name == "easy":
            task = get_easy_task()
        elif task_name == "medium":
            task = get_medium_task()
        else:
            task = get_hard_task()

        return run_task(task)