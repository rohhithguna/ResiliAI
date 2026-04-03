#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.easy_task import get_easy_task
from tasks.medium_task import get_medium_task
from tasks.hard_task import get_hard_task
from src.inference.inference import run_task

print("===== RUNNING ALL TASKS =====")

for fn in [get_easy_task, get_medium_task, get_hard_task]:
    result = run_task(fn())
    print(f"{result['task']} → score={result['score']:.3f}, steps={result['steps']}")

print("===== DONE =====")
