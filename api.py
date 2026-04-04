import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

# Ensure root path works
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env.sre_openenv import SREOpenEnv
from src.inference.inference import run_task
from tasks.easy_task import get_easy_task
from tasks.medium_task import get_medium_task
from tasks.hard_task import get_hard_task

app = FastAPI()

env = SREOpenEnv(seed=42)
current_state = None


class StepInput(BaseModel):
    action: int


class ResetInput(BaseModel):
    task: str = "easy"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(data: ResetInput):
    global current_state, env

    env = SREOpenEnv(seed=42)

    if data.task == "easy":
        task = get_easy_task()
    elif data.task == "medium":
        task = get_medium_task()
    else:
        task = get_hard_task()

    env.apply_initial_state(task["initial_state"])
    current_state = env.state()

    return {
        "state": current_state,
        "task": data.task
    }


@app.post("/step")
def step(data: StepInput):
    global current_state

    next_state, reward, done, info = env.step(data.action)
    current_state = next_state

    return {
        "state": next_state,
        "reward": reward,
        "done": done,
        "info": info
    }


@app.post("/run")
def run(data: ResetInput):
    if data.task == "easy":
        t = get_easy_task()
    elif data.task == "medium":
        t = get_medium_task()
    else:
        t = get_hard_task()

    result = run_task(t)
    return result