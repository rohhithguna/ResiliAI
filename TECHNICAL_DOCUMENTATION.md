# ResiliAI - Autonomous Incident Recovery System

## 1. Title Page

**Project Name:** ResiliAI - Autonomous Incident Recovery System

**Purpose:** Automated recovery of distributed service incidents using an OpenEnv-compliant environment, hybrid decision making, and FastAPI-based execution.

**Implementation Stack:** FastAPI, OpenEnv, task definitions, graders, Uvicorn, Docker, HuggingFace Spaces

**Primary Entry Point:** [api.py](/Users/rohhithg/Desktop/meta_project/api.py)

**OpenEnv Configuration:** [openenv.yaml](/Users/rohhithg/Desktop/meta_project/openenv.yaml)

## 2. Abstract

ResiliAI is a production-oriented incident recovery system for simulated distributed services. The project combines a deterministic OpenEnv environment wrapper, a task-driven execution model, and a hybrid rule-based plus RL inference layer. The system is exposed through a FastAPI API for automated evaluation and deployment. Validation is driven by task runners and graders, with the environment, tasks, and scoring behavior defined explicitly in the repository.

## 3. Problem Statement

Distributed systems fail in multiple ways at once: traffic spikes, backend degradation, database failure, and cascading latency. Manual response is slow and inconsistent. ResiliAI addresses this by standardizing incident handling into a reproducible environment where each scenario can be reset, stepped, and evaluated automatically. The goal is not to simulate theory, but to provide a clear execution path for recovery actions, scoring, and validation.

## 4. Proposed Solution

The project solves incident recovery by combining:

- A reproducible OpenEnv environment wrapper ([SREOpenEnv](src/env/sre_openenv.py))
- Fixed task scenarios for easy, medium, and hard recovery cases
- Graders that return bounded scores for validation
- A hybrid control layer that blends rules with RL-based selection
- A FastAPI service layer for automated API evaluation

This structure makes the system suitable for OpenEnv validation, deployment on HuggingFace Spaces, and scripted testing.

## 5. System Architecture

The system is organized into five runtime layers:

1. **API layer** - FastAPI app in [api.py](/Users/rohhithg/Desktop/meta_project/api.py)
2. **Environment layer** - [SREOpenEnv](src/env/sre_openenv.py) wraps the underlying simulation
3. **Task layer** - `easy`, `medium`, and `hard` tasks define scenarios and max steps
4. **Inference layer** - [src/inference/inference.py](src/inference/inference.py) runs the recovery policy
5. **Evaluation layer** - graders score outputs and validate behavior

Deployment is configured through [Dockerfile](/Users/rohhithg/Desktop/meta_project/Dockerfile) and HuggingFace metadata in [README.md](/Users/rohhithg/Desktop/meta_project/README.md).

## 6. Components Description

### 6.1 API Service

The FastAPI service exposes endpoints for health checks, environment reset, environment stepping, and task execution. It is the primary runtime for automated evaluation.

### 6.2 Environment Wrapper

[SREOpenEnv](src/env/sre_openenv.py) normalizes state formatting, applies scenarios, exposes `reset`, `step`, `state`, and `apply_initial_state`, and ensures the observation structure is consistent for downstream consumers.

### 6.3 Tasks

The `tasks/` package defines scenario presets:

- Easy: traffic spike recovery
- Medium: database failure recovery
- Hard: multi-failure recovery

Each task includes initial state data and a maximum step budget.

### 6.4 Graders

The `graders/` package evaluates outputs from task execution. Scores are bounded and deterministic enough for OpenEnv validation.

### 6.5 Inference Engine

[src/inference/inference.py](src/inference/inference.py) executes task runs and produces structured results. It is used by both the API and validation scripts.

### 6.6 Deployment Files

- [pyproject.toml](/Users/rohhithg/Desktop/meta_project/pyproject.toml) defines the package and script entry
- [requirements.txt](/Users/rohhithg/Desktop/meta_project/requirements.txt) lists runtime dependencies
- [uv.lock](/Users/rohhithg/Desktop/meta_project/uv.lock) locks dependency resolution
- [server/app.py](/Users/rohhithg/Desktop/meta_project/server/app.py) provides a lightweight import wrapper for the app

## 7. Workflow (Step-by-Step Execution)

1. The service starts through Uvicorn using `api:app`.
2. `/health` confirms the service is alive.
3. `/reset` loads the selected task and applies the initial scenario state.
4. `/step` sends an action into the environment and returns updated state, reward, done, and info.
5. `/run` executes a full task run using the existing inference pipeline.
6. The task result is passed into the corresponding grader.
7. Validation scripts compare results against expected thresholds and OpenEnv requirements.

This sequence is deterministic and intended for automated scoring.

## 8. API Design (FastAPI Endpoints)

### `/`

Root availability endpoint used by the deployed service. In the current implementation it returns a simple runtime message confirming the API is running.

### `GET /health`

Returns service status for liveness checks.

Response shape:

```json
{"status": "ok"}
```

### `POST /reset`

Resets the environment and loads a task scenario.

Request body:

```json
{"task": "easy"}
```

Supported values:

- `easy`
- `medium`
- `hard`

Response includes the current state and selected task name.

### `POST /step`

Executes one environment action.

Request body:

```json
{"action": 0}
```

Response includes:

- `state`
- `reward`
- `done`
- `info`

### `POST /run`

Runs an entire task through the existing inference pipeline and returns the task result.

Request body:

```json
{"task": "easy"}
```

## 9. Core Algorithm (RL + Rule-based Hybrid)

The inference layer uses a hybrid decision pattern:

- Rule-based behavior handles obvious recovery decisions when the system state is clearly degraded.
- RL-based selection supports adaptive response when the situation requires policy-driven action.
- The environment and task runner keep the execution deterministic for validation.

The project does not rely on open-ended autonomous learning during evaluation. Instead, it uses the existing inference path to generate reproducible outputs for task scoring.

## 10. Evaluation & Results

Evaluation is based on OpenEnv task execution plus grader scoring.

What is evaluated:

- Task completion behavior
- Step count and recovery progression
- Score validity and bounds
- Environment response to actions
- Endpoint availability for automated runners

Current validation posture:

- OpenEnv structure is defined in [openenv.yaml](/Users/rohhithg/Desktop/meta_project/openenv.yaml)
- Tasks and graders are explicitly mapped
- API endpoints respond successfully for `/health`, `/reset`, and `/step`
- The system is suitable for automated validation and Docker deployment

## 11. Deployment (OpenEnv + HuggingFace)

The deployment model uses OpenEnv metadata and HuggingFace Spaces conventions:

- `sdk: docker` in [README.md](/Users/rohhithg/Desktop/meta_project/README.md)
- `app_file: api.py` as the service entry point
- Docker runtime command: `uvicorn api:app --host 0.0.0.0 --port 7860`

This setup keeps the deployment layer separate from the environment logic and makes the project compatible with automated evaluation workflows.

## 12. How to Run

### Local validation

```bash
python3 validate_tasks.py
```

### Execute all tasks

```bash
python3 run_all.py
```

### Start the API service

```bash
uvicorn api:app --host 127.0.0.1 --port 7860
```

### Run the Docker container

```bash
docker build -t resiliai .
docker run -p 7860:7860 resiliai
```

### Service checks

```bash
curl http://127.0.0.1:7860/health
curl -X POST http://127.0.0.1:7860/reset -H 'Content-Type: application/json' -d '{"task":"easy"}'
curl -X POST http://127.0.0.1:7860/step -H 'Content-Type: application/json' -d '{"action":0}'
```

## 13. Strengths & Limitations

### Strengths

- Clear OpenEnv structure with explicit env, tasks, and graders
- Deterministic execution path for validation
- FastAPI service layer for automated API testing
- Docker-friendly deployment model
- Minimal runtime surface area, which reduces integration risk

### Limitations

- The task set is fixed to predefined scenarios
- The current API is evaluation-oriented rather than interactive enough for operator workflows
- Recovery behavior is constrained by the existing environment and inference design

## 14. Future Improvements

- Add richer incident scenario coverage
- Extend API responses with trace metadata for debugging
- Add per-step action explanations for inspection
- Add more granular evaluator metrics
- Add health and readiness separation for deployment orchestration

## 15. Conclusion

ResiliAI is a cleanly structured, OpenEnv-compliant incident recovery system built for automated evaluation. It combines a reproducible environment, fixed recovery tasks, scoring utilities, and a FastAPI service layer into a deployable package. The implementation is intentionally compact and focused on behavior, validation, and deployment compatibility.