---
title: resiliAI
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: docker
app_file: api.py
base_path: /web
---


# ResiliAI - Autonomous Incident Recovery System

## Overview
ResiliAI is an AI-driven system for automatic recovery of distributed systems using a hybrid approach:

- Reinforcement Learning (adaptive decisions)
- Rule-based safety (critical overrides)
- Multi-component system modeling

## Key Idea
Instead of relying on static rules, we allow an RL-based controller to adapt to dynamic system conditions while ensuring safety using minimal rule overrides.

## System Components
- Frontend
- Backend
- Database
- Traffic & Load Balancing

## Architecture
- `sre_openenv.py` -> OpenEnv wrapper
- `inference.py` -> decision engine
- `benchmark.py` -> performance evaluation
- `robustness_test.py` -> stability validation
- `tasks/` -> evaluation scenarios
- `graders/` -> scoring system
- `app.py` -> live demo dashboard

## Results

**Note:** Results shown below are averaged across multiple seeded runs for consistency and reproducibility.

### Benchmark
- Average Score: > 0.7 (across multiple runs)
- Average Steps: ~12-18 (averaged across scenarios)

### Robustness
- Success Rate: ~0.6 across random seeds (statistically averaged)

### RL Usage
- RL decisions dominate when appropriate (>60% on complex scenarios)
- Rule overrides used only in critical states (safety prioritized)

## Key Strengths
- Deterministic behavior
- Robust across multiple scenarios
- Hybrid architecture ensures safety + adaptability
- Fully interpretable decision process

## How to Run

### Quick Start (Recommended)
```bash
chmod +x quick_start.sh
./quick_start.sh
```
This will:
1. Create a virtual environment
2. Install all dependencies
3. Run validation suite
4. Launch the interactive demo

### Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run validation
python3 validate_tasks.py

# Run all tasks
python3 run_all.py

# Run inference module
python3 -m src.inference.inference

# Launch interactive demo
streamlit run frontend/app.py
```

## Demo Flow
1. Run `python3 validate_tasks.py` to verify all systems
2. Run `python3 run_all.py` to see task execution
3. Run `streamlit run frontend/app.py` for live dashboard
4. Observe system state and recovery actions
5. Watch automatic incident recovery in action

## Why This Matters
Real-world distributed systems require both:
- Adaptability (RL)
- Safety (rules)

ResiliAI combines both to create reliable autonomous recovery.

## Final Statement
This project demonstrates that combining reinforcement learning with structured safety constraints leads to robust and scalable system recovery.
