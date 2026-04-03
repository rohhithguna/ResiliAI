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

### 1. Install dependencies
pip install torch streamlit

### 2. Run evaluation
python benchmark.py
python robustness_test.py

### 3. Run demo
streamlit run app.py

## Demo Flow
1. Start simulation
2. Observe system state
3. Run auto steps
4. Watch system recover automatically

## Why This Matters
Real-world distributed systems require both:
- Adaptability (RL)
- Safety (rules)

ResiliAI combines both to create reliable autonomous recovery.

## Final Statement
This project demonstrates that combining reinforcement learning with structured safety constraints leads to robust and scalable system recovery.
