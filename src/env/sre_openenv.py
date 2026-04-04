import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import random

try:
    import numpy as np
except Exception:
    np = None

try:
    import torch
except Exception:
    torch = None


ACTIONS = {
    0: "noop",
    1: "restart_frontend",
    2: "restart_backend",
    3: "restart_database",
    4: "throttle_traffic",
    5: "rebalance_load",
}


class SREOpenEnv:
    def __init__(self, seed=42):
        from src.env.sre_environment import SREEnvironment

        self.seed = int(seed if seed is not None else 42)
        self._set_seed(self.seed)
        self.rng = random.Random(self.seed)
        self.env = SREEnvironment(seed=self.seed, max_steps=60)
        self.step_count = 0
        self.min_recovery_steps = 6
        if not hasattr(self.env, "current_state"):
            self.env.current_state = self.env._get_state()

    def force_seed(self):
        import random

        random.seed(self.seed)

        try:
            import numpy as np

            np.random.seed(self.seed)
        except Exception:
            pass

        try:
            import torch

            torch.manual_seed(self.seed)
        except Exception:
            pass

    def reset(self):
        self.force_seed()
        from src.env.sre_environment import SREEnvironment

        self.env = SREEnvironment(seed=self.seed, max_steps=60)
        state = self.env.reset()
        self.step_count = 0
        self.env.current_state = state
        return self.format_obs(state)

    def step(self, action):
        self.force_seed()
        safe_action = int(action) if action in ACTIONS else 0
        next_state, reward, raw_done, info = self.env.step(safe_action)
        self.step_count += 1
        self.env.current_state = next_state

        obs = self._format_obs(next_state)
        recovered = self._is_recovered(obs)
        done = bool(raw_done) and (self.step_count >= self.min_recovery_steps or recovered)

        return obs, float(reward), bool(done), self._format_info(obs, info, recovered, done)

    def state(self):
        if hasattr(self.env, "current_state"):
            return self._format_obs(self.env.current_state)
        if hasattr(self.env, "_get_state"):
            return self._format_obs(self.env._get_state())
        return self.reset()

    def inject_scenario(self, scenario=None):
        scenario = scenario or "traffic_spike"

        if scenario == "traffic_spike":
            self.env.global_traffic_load = 0.88
            self.env.request_queue = 520.0
            self.env.a_latency = max(self.env.a_latency, 560.0)
            self.env.b_latency = max(self.env.b_latency, 780.0)
        elif scenario == "db_failure":
            self.env.db_status = self.env.STATUS_DOWN
            self.env.db_latency = 1600.0
            self.env.request_queue = max(self.env.request_queue, 480.0)
        elif scenario == "multi_failure":
            self.env.a_status = self.env.STATUS_DEGRADED
            self.env.b_status = self.env.STATUS_DOWN
            self.env.db_status = self.env.STATUS_DEGRADED
            self.env.a_latency = max(self.env.a_latency, 880.0)
            self.env.b_latency = max(self.env.b_latency, 1300.0)
            self.env.db_latency = max(self.env.db_latency, 1200.0)
            self.env.global_traffic_load = max(self.env.global_traffic_load, 0.85)
            self.env.request_queue = max(self.env.request_queue, 680.0)

        self.env._update_error_rate()
        self.env.current_state = self.env._get_state()
        return str(scenario)

    def apply_initial_state(self, initial_state):
        if not isinstance(initial_state, dict):
            return

        scenario = initial_state.get("scenario")
        if scenario:
            self.inject_scenario(scenario)

        status_map = {"down": 0, "degraded": 1, "healthy": 2}

        def status_value(key, current):
            value = initial_state.get(key, current)
            if isinstance(value, str):
                return status_map.get(value.lower(), current)
            try:
                value = int(value)
            except Exception:
                return current
            return 0 if value < 0 else 2 if value > 2 else value

        self.env.a_status = status_value("frontend_status", self.env.a_status)
        self.env.b_status = status_value("backend_status", self.env.b_status)
        self.env.db_status = status_value("db_status", self.env.db_status)

        self.env.a_latency = float(initial_state.get("frontend_latency", self.env.a_latency))
        self.env.b_latency = float(initial_state.get("backend_latency", self.env.b_latency))
        self.env.db_latency = float(initial_state.get("db_latency", self.env.db_latency))
        self.env.global_traffic_load = float(initial_state.get("traffic_load", self.env.global_traffic_load))
        self.env.request_queue = float(initial_state.get("request_queue", self.env.request_queue))
        self.env.traffic_balance = float(initial_state.get("traffic_balance", self.env.traffic_balance))
        self.env._update_error_rate()
        self.env.current_state = self.env._get_state()

    def _safe_get(self, state, index, key, default):
        if isinstance(state, dict):
            return state.get(key, default)
        if isinstance(state, (list, tuple)) and index < len(state):
            return state[index]
        return default

    def _format_obs(self, state):
        frontend_status = int(self._safe_get(state, 0, "frontend_status", 2))
        backend_status = int(self._safe_get(state, 1, "backend_status", 2))
        db_status = int(self._safe_get(state, 2, "db_status", 2))
        frontend_latency = float(self._safe_get(state, 3, "frontend_latency", 100.0))
        backend_latency = float(self._safe_get(state, 4, "backend_latency", 100.0))
        db_latency = float(self._safe_get(state, 5, "db_latency", 100.0))
        error_rate = float(self._safe_get(state, 6, "error_rate", 1.0))
        traffic_load = float(self._safe_get(state, 7, "traffic_load", 0.5))
        traffic_balance = float(self._safe_get(state, 8, "traffic_balance", 0.5))
        request_queue = float(self._safe_get(state, 9, "request_queue", 0.0))

        load_balancer_status = 2
        if abs(traffic_balance - 0.5) > 0.3 or request_queue > 650.0:
            load_balancer_status = 0
        elif abs(traffic_balance - 0.5) > 0.15 or request_queue > 350.0:
            load_balancer_status = 1

        return {
            "frontend_status": frontend_status,
            "backend_status": backend_status,
            "db_status": db_status,
            "frontend_latency": frontend_latency,
            "backend_latency": backend_latency,
            "db_latency": db_latency,
            "latency_values": {
                "frontend": frontend_latency,
                "backend": backend_latency,
                "database": db_latency,
            },
            "error_rate": max(0.0, min(1.0, error_rate)),
            "traffic_load": max(0.0, min(1.0, traffic_load)),
            "traffic": max(0.0, min(1.0, traffic_load)),
            "traffic_balance": traffic_balance,
            "request_queue": max(0.0, request_queue),
            "load_balancer_status": load_balancer_status,
        }

    def format_obs(self, state):
        return self._format_obs(state)

    def _is_recovered(self, obs):
        return (
            int(obs.get("frontend_status", 0)) == 2
            and int(obs.get("backend_status", 0)) == 2
            and int(obs.get("db_status", 0)) == 2
            and float(obs.get("error_rate", 1.0)) < 0.08
            and int(obs.get("load_balancer_status", 0)) >= 1
        )

    def _format_info(self, obs, base_info, recovered, done):
        info = dict(base_info if isinstance(base_info, dict) else {})
        info.update(
            {
                "health_score": self._health_score(obs),
                "recovered": bool(recovered),
                "step": int(self.step_count),
                "done": bool(done),
            }
        )
        return info

    def _health_score(self, obs):
        healthy = 0
        healthy += 1 if int(obs.get("frontend_status", 0)) == 2 else 0
        healthy += 1 if int(obs.get("backend_status", 0)) == 2 else 0
        healthy += 1 if int(obs.get("db_status", 0)) == 2 else 0
        lb_bonus = 1 if int(obs.get("load_balancer_status", 0)) >= 1 else 0
        return (healthy + lb_bonus) / 4.0

    def _set_seed(self, seed):
        random.seed(seed)
        if np is not None:
            np.random.seed(seed)
        if torch is not None:
            torch.manual_seed(seed)
