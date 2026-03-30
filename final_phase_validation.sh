#!/usr/bin/env bash

set +e

echo "===== FINAL PHASE VALIDATION START ====="

FAIL=0
PY="/Users/rohhithg/Desktop/meta_project/.venv/bin/python"

# -----------------------------
# PHASE 4 — FILES + YAML
# -----------------------------
echo "Checking Phase 4..."

[ -f openenv.yaml ] || FAIL=1
[ -f Dockerfile ] || FAIL=1
[ -f requirements.txt ] || FAIL=1
[ -f README.md ] || FAIL=1

"$PY" - << 'EOF'
import yaml

with open("openenv.yaml") as f:
    data = yaml.safe_load(f)

assert "entry_point" in data
assert "tasks" in data
assert "grader" in data
EOF

if [ $? -ne 0 ]; then FAIL=1; fi


# -----------------------------
# PHASE 5 — BENCHMARK QUALITY
# -----------------------------
echo "Checking Phase 5..."

"$PY" - << 'EOF'
from benchmark import benchmark

res = benchmark(runs=5)

for r in res.values():
    assert r["avg_score"] >= 0.5
    assert r["avg_steps"] > 0
EOF

if [ $? -ne 0 ]; then FAIL=1; fi


# -----------------------------
# PHASE 6 — ROBUSTNESS
# -----------------------------
echo "Checking Phase 6..."

"$PY" - << 'EOF'
from robustness_test import run_case

scores = []
for i in range(1, 6):
    s, _ = run_case(i)
    scores.append(s)

avg = sum(scores) / len(scores)
assert avg >= 0.4
EOF

if [ $? -ne 0 ]; then FAIL=1; fi


# -----------------------------
# PHASE 7 — MULTI-AGENT
# -----------------------------
echo "Checking Phase 7..."

[ -f multi_agent.py ] || FAIL=1

"$PY" - << 'EOF'
from multi_agent import CoordinatorAgent

c = CoordinatorAgent()
a = c.select_action([2,2,2,100,100,100,0.01,0.2,0,0])

assert isinstance(a, int)
EOF

if [ $? -ne 0 ]; then FAIL=1; fi


# -----------------------------
# PHASE 8 — DEMO + PRESENTATION
# -----------------------------
echo "Checking Phase 8..."

[ -f demo_runner.py ] || FAIL=1
[ -f demo_script.txt ] || FAIL=1

"$PY" - << 'EOF'
import subprocess

result = subprocess.run(
    ["/Users/rohhithg/Desktop/meta_project/.venv/bin/python", "demo_runner.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    timeout=10
)

assert result.returncode == 0
EOF

if [ $? -ne 0 ]; then FAIL=1; fi


# -----------------------------
# FINAL RESULT
# -----------------------------
echo ""
echo "===== FINAL RESULT ====="

if [ $FAIL -eq 0 ]; then
    echo "PASS ✅"
else
    echo "FAIL ❌"
fi
