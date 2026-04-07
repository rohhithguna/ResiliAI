import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

# Ensure root path works
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env.sre_openenv import SREOpenEnv
from src.inference.inference import call_llm, run_task
from tasks.easy_task import get_easy_task
from tasks.medium_task import get_medium_task
from tasks.hard_task import get_hard_task

app = FastAPI()

# ✅ ROOT ENDPOINT
@app.get("/")
def root():
    return {"message": "resiliAI API is running"}

env = SREOpenEnv(seed=42)
current_state = None


class StepInput(BaseModel):
    action: int


class ResetInput(BaseModel):
    task: str = "easy"


@app.get("/health")
def health():
    return {"status": "ok"}


# ✅ FIXED: reset must work WITHOUT body
@app.post("/reset")
def reset(data: ResetInput = None):
    global current_state, env

    env = SREOpenEnv(seed=42)

    # default task
    task_name = "easy"
    if data and data.task:
        task_name = data.task

    if task_name == "easy":
        task = get_easy_task()
    elif task_name == "medium":
        task = get_medium_task()
    else:
        task = get_hard_task()

    env.apply_initial_state(task["initial_state"])
    current_state = env.state()

    return {
        "state": current_state,
        "task": task_name
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


# ✅ FIXED: run must also allow no body
@app.post("/run")
def run(data: ResetInput = None):

    call_llm()

    task_name = "easy"
    if data and data.task:
        task_name = data.task

    if task_name == "easy":
        t = get_easy_task()
    elif task_name == "medium":
        t = get_medium_task()
    else:
        t = get_hard_task()

    result = run_task(t)
    return result