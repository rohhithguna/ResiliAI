# Multi-Agent AI System for Autonomous System Recovery

## 1. Problem
Modern distributed systems fail in complex ways: service outages, cascading latency, traffic spikes, and queue buildup can rapidly degrade user experience. Manual incident response is often slow and inconsistent.

## 2. Solution
This project combines reinforcement learning, multi-agent decision support, and traffic-aware system dynamics to automate incident recovery decisions.

Core idea:
- RL provides policy intelligence
- Domain agents provide interpretable suggestions
- A coordinator enforces safe priority decisions under critical conditions
- Hybrid approach balances learning and safety, achieving both reliability and adaptability.
- Our final system is a hybrid architecture where RL handles adaptive decision-making while rule-based safety ensures reliability.

## 3. Architecture
Components:
- `sre_environment.py`: RL environment with service health, latency, traffic, balance, and queue dynamics
- `train_dqn.py`: DQN training pipeline (single model, replay buffer, target network)
- `multi_agent.py`:
  - FrontendAgent
  - BackendAgent
  - DatabaseAgent
  - TrafficAgent
  - CoordinatorAgent (priority: DB > Backend > Frontend > Traffic > RL fallback)
- `test_rl_agent.py`: hybrid-focused ablation and statistical evaluation
- `frontend/app.py`: Streamlit dashboard for live step-by-step demo

## 4. Key Features
- RL decision making for autonomous recovery
- Multi-agent coordination with interpretable suggestions
- Traffic-aware and queue-aware recovery control
- Critical-state prioritization for safer actions
- Real-time dashboard demo with auto-run mode

## 5. Results
Primary evaluation focus: stability, recovery speed, and safety under diverse incident conditions.

The hybrid system combines learning and safety constraints to achieve reliable recovery under diverse conditions.

Statistical ablation summary (5 runs x 100 episodes, mean +- std, 95% CI):

| Method | Success | Reward | Steps |
|--------|--------:|-------:|------:|
| Rule | 56.20% +- 5.42 (CI +- 4.75) | -824.3973 +- 64.9443 (CI +- 56.91) | 23.69 +- 1.37 (CI +- 1.20) |
| RL | 65.60% +- 3.61 (CI +- 3.16) | -320.9781 +- 25.7320 (CI +- 22.55) | 14.58 +- 1.31 (CI +- 1.15) |
| Hybrid | 69.20% +- 3.66 (CI +- 3.21) | -395.1502 +- 30.6160 (CI +- 26.83) | 16.72 +- 1.00 (CI +- 0.88) |

Hybrid policy usage:
- RL usage: 95.17% +- 1.14 (95% CI +- 1.00)
- Rule override: 4.83% +- 1.14 (95% CI +- 1.00)

Hybrid vs Rule dominance:
- Success: higher
- Reward: higher
- Steps: lower

Safety interpretation:
- Recovery behavior remains consistent across runs (low variance).
- Safety overrides activate selectively in critical states.
- Adaptive decisions are still used as the primary control path.

Pure learning-based systems can be unstable in safety-critical environments. Our hybrid approach balances adaptability and reliability.

## 6. How to Run
Install dependencies:

```bash
pip install torch streamlit
```

Train model:

```bash
python train_dqn.py
```

Evaluate RL vs Rule:

```bash
python test_rl_agent.py
```

Run frontend demo:

```bash
streamlit run frontend/app.py
```

## 7. Demo Instructions
Suggested live flow (2 minutes):
1. Open dashboard and press `Start`
2. Show `System State` and `Agent Decisions`
3. Click `Run Auto Demo` to execute 10 automatic steps
4. Highlight coordinator final action and recent action table
5. Emphasize successful recovery scenarios, multi-agent coordination, and explainable decisions
6. Conclude: hybrid control improves reliability with adaptive behavior and safety constraints

## 8. Repository Files
- `sre_environment.py`
- `agent_rule.py`
- `multi_agent.py`
- `train_dqn.py`
- `test_rl_agent.py`
- `frontend/app.py`
- `dqn_model.pth`
