from typing import Dict, Any


class TaskInput:
    def __init__(self, task_name: str = "easy"):
        self.task_name = task_name


class StepInput:
    def __init__(self, action: int):
        self.action = action


class StepOutput:
    def __init__(self, state: Dict[str, Any], reward: float, done: bool, info: Dict[str, Any]):
        self.state = state
        self.reward = reward
        self.done = done
        self.info = info


class RunOutput:
    def __init__(self, result: Dict[str, Any]):
        self.result = result