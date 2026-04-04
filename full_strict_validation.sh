#!/usr/bin/env bash

set +e

echo "===== FULL STRICT VALIDATION START ====="

FAIL=0
PY="/Users/rohhithg/Desktop/meta_project/.venv/bin/python"

# -----------------------------
# PHASE 1 CHECK
# -----------------------------
echo "Checking Phase 1..."

"$PY" - << 'EOF'
from sre_openenv import SREOpenEnv

env = SREOpenEnv(seed=42)

state = env.reset()
assert isinstance(state, (list, dict))

ns, r, d, i = env.step(0)
assert isinstance(ns, (list, dict))
assert isinstance(r, float)
assert isinstance(d, bool)
EOF

if [ $? -ne 0 ]; then echo "PHASE 1 FAIL"; FAIL=1; else echo "PHASE 1 PASS"; fi


# -----------------------------
# PHASE 2 CHECK
# -----------------------------
echo "Checking Phase 2..."

"$PY" - << 'EOF'
from sre_openenv import SREOpenEnv
from tasks.easy_task import get_easy_task
from tasks.medium_task import get_medium_task
from tasks.hard_task import get_hard_task
from graders.easy_grader import grade_easy
from graders.medium_grader import grade_medium
from graders.hard_grader import grade_hard

env = SREOpenEnv(seed=42)
state = env.reset()

def to_dict(s):
    if isinstance(s, dict):
        return {
            "frontend_status": s.get("frontend_status", 0),
            "backend_status": s.get("backend_status", 0),
            "db_status": s.get("db_status", 0),
            "frontend_latency": s.get("frontend_latency", 0),
            "backend_latency": s.get("backend_latency", 0),
            "db_latency": s.get("db_latency", 0),
            "error_rate": s.get("error_rate", 0),
            "traffic_load": s.get("traffic_load", 0),
            "request_queue": s.get("request_queue", 0),
        }
    return {
        "frontend_status": s[0],
        "backend_status": s[1],
        "db_status": s[2],
        "frontend_latency": s[3],
        "backend_latency": s[4],
        "db_latency": s[5],
        "error_rate": s[6],
        "traffic_load": s[7],
        "request_queue": s[8] if len(s)>8 else 0
    }

tasks = [
    (get_easy_task(), grade_easy),
    (get_medium_task(), grade_medium),
    (get_hard_task(), grade_hard)
]

for t,g in tasks:
    score = g(to_dict(state))
    assert 0.0 <= score <= 1.0
EOF

if [ $? -ne 0 ]; then echo "PHASE 2 FAIL"; FAIL=1; else echo "PHASE 2 PASS"; fi


# -----------------------------
# PHASE 3 CHECK
# -----------------------------
echo "Checking Phase 3..."

"$PY" - << 'EOF'
from inference import run_all

res = run_all()
assert isinstance(res, list)

for r in res:
    assert 0.0 <= r["final_score"] <= 1.0
    assert r["steps"] > 0
EOF

if [ $? -ne 0 ]; then echo "PHASE 3 FAIL"; FAIL=1; else echo "PHASE 3 PASS"; fi


# -----------------------------
# PHASE 4 CHECK
# -----------------------------
echo "Checking Phase 4..."

if [ ! -f openenv.yaml ]; then echo "Missing openenv.yaml"; FAIL=1; fi
if [ ! -f Dockerfile ]; then echo "Missing Dockerfile"; FAIL=1; fi
if [ ! -f requirements.txt ]; then echo "Missing requirements.txt"; FAIL=1; fi
if [ ! -f README.md ]; then echo "Missing README.md"; FAIL=1; fi

"$PY" - << 'EOF'
import yaml
with open("openenv.yaml") as f:
    data = yaml.safe_load(f)

assert "entry_point" in data
assert "tasks" in data
assert "grader" in data
EOF

if [ $? -ne 0 ]; then echo "PHASE 4 FAIL"; FAIL=1; else echo "PHASE 4 PASS"; fi


# -----------------------------
# PHASE 5 CHECK
# -----------------------------
echo "Checking Phase 5..."

"$PY" - << 'EOF'
from benchmark import benchmark

res = benchmark(runs=5)

for task, r in res.items():
    assert r["avg_score"] >= 0.5
    assert r["avg_steps"] > 0
EOF

if [ $? -ne 0 ]; then echo "PHASE 5 FAIL"; FAIL=1; else echo "PHASE 5 PASS"; fi


# -----------------------------
# FINAL RESULT
# -----------------------------
echo ""
echo "===== FINAL RESULT ====="

if [ $FAIL -eq 0 ]; then
    echo "FINAL STATUS: PASS ✅"
else
    echo "FINAL STATUS: FAIL ❌"
fi
